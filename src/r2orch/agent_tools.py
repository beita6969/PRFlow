from __future__ import annotations

import ast
import io
import json
import subprocess
import sys
import tempfile
import time
import tokenize
from dataclasses import dataclass
from functools import reduce
from operator import mul
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .deadline import RolloutDeadline


class AgentTool(Protocol):
    name: str
    description: str
    parameters: dict[str, Any]
    stateful: bool

    def execute(self, arguments: dict[str, Any]) -> str: ...


_ALLOWED_PYTHON_MODULES = {
    "cmath",
    "collections",
    "decimal",
    "fractions",
    "functools",
    "itertools",
    "math",
    "statistics",
}
# R2Orch MBPP+ checks: the AIME list plus standard-library modules that
# MBPP solutions import. No file, process or network modules.
_STDLIB_PYTHON_MODULES = _ALLOWED_PYTHON_MODULES | {
    "array",
    "bisect",
    "calendar",
    "copy",
    "dataclasses",
    "datetime",
    "enum",
    "heapq",
    "numbers",
    "operator",
    "random",
    "re",
    "string",
    "textwrap",
    "typing",
    "unicodedata",
}
STDLIB_PYTHON_DESCRIPTION = (
    "Execute stateless sandboxed Python to compute, test or verify code and values. "
    "Allowed imports: " + ", ".join(sorted(_STDLIB_PYTHON_MODULES)) + ". No files, "
    "processes or network. Arguments: code and purpose; purpose must be compute, explore, "
    "verify, or repair. Print every value needed in the observation."
)
_FORBIDDEN_PYTHON_NAMES = {
    "__builtins__",
    "__import__",
    "breakpoint",
    "compile",
    "eval",
    "exec",
    "exit",
    "getattr",
    "globals",
    "help",
    "input",
    "locals",
    "open",
    "quit",
    "setattr",
    "vars",
}

_PYTHON_MATH_OPERATORS = {"×": "*", "÷": "/", "−": "-", "–": "-"}


def _normalize_python_math_syntax(code: str) -> tuple[str, list[str]]:
    """Normalize presentation operators in code tokens, never strings or comments."""

    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(code).readline))
    except (IndentationError, tokenize.TokenError):
        return code, []
    observed: set[str] = set()
    normalized = []
    for token in tokens:
        replacement = (
            _PYTHON_MATH_OPERATORS.get(token.string) if token.type == tokenize.ERRORTOKEN else None
        )
        if replacement is None:
            normalized.append(token)
            continue
        observed.add(token.string)
        normalized.append(token._replace(string=replacement))
    return tokenize.untokenize(normalized), [
        f"{source}->{target}"
        for source, target in _PYTHON_MATH_OPERATORS.items()
        if source in observed
    ]


@dataclass(frozen=True)
class PythonExecutionTool:
    """Stateless, process-isolated Python calculator for AIME worker actions."""

    timeout_s: float = 5.0
    max_output_chars: int = 8000
    max_code_chars: int = 12000
    memory_limit_mb: int = 1024
    name: str = "python_exec"
    description: str = STDLIB_PYTHON_DESCRIPTION

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "code": {"type": "string"},
                "purpose": {
                    "type": "string",
                    "enum": ["compute", "explore", "verify", "repair"],
                },
            },
            "required": ["code", "purpose"],
            "additionalProperties": False,
        }

    def execute(self, arguments: dict[str, Any]) -> str:
        code = arguments.get("code")
        purpose = str(arguments.get("purpose", "")).strip().casefold()
        if not isinstance(code, str) or not code.strip():
            raise ValueError("python_exec requires non-empty code")
        code, input_normalizations = _normalize_python_math_syntax(code)
        if len(code) > self.max_code_chars:
            raise ValueError(f"python_exec code exceeds {self.max_code_chars} characters")
        if purpose not in {"compute", "explore", "verify", "repair"}:
            raise ValueError("python_exec purpose must be compute, explore, verify, or repair")
        validation_error = _validate_python(code, frozenset(_STDLIB_PYTHON_MODULES))
        if validation_error:
            return json.dumps(
                {
                    "status": "error",
                    "purpose": purpose,
                    "stdout": "",
                    "stderr": validation_error,
                    "elapsed_ms": 0,
                    "truncated": False,
                    "input_normalizations": input_normalizations,
                },
                ensure_ascii=False,
            )
        started = time.monotonic()
        try:
            with tempfile.TemporaryDirectory(prefix="aime-python-") as workdir:
                completed = subprocess.run(
                    [sys.executable, "-I", "-c", code],
                    cwd=workdir,
                    env={"PYTHONIOENCODING": "utf-8"},
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_s,
                    check=False,
                    preexec_fn=self._limit_resources,
                )
        except subprocess.TimeoutExpired as exc:
            stdout = _stream_text(exc.stdout)
            stderr = _stream_text(exc.stderr)
            return self._result(
                status="timeout",
                purpose=purpose,
                stdout=stdout,
                stderr=stderr or f"execution exceeded {self.timeout_s:g} seconds",
                started=started,
                input_normalizations=input_normalizations,
            )
        return self._result(
            status="ok" if completed.returncode == 0 else "error",
            purpose=purpose,
            stdout=completed.stdout,
            stderr=completed.stderr,
            started=started,
            input_normalizations=input_normalizations,
        )

    def _result(
        self,
        *,
        status: str,
        purpose: str,
        stdout: str,
        stderr: str,
        started: float,
        input_normalizations: list[str] | None = None,
    ) -> str:
        combined_length = len(stdout) + len(stderr)
        remaining = self.max_output_chars
        bounded_stdout = stdout[:remaining]
        remaining -= len(bounded_stdout)
        bounded_stderr = stderr[: max(0, remaining)]
        return json.dumps(
            {
                "status": status,
                "purpose": purpose,
                "stdout": bounded_stdout,
                "stderr": bounded_stderr,
                "elapsed_ms": round((time.monotonic() - started) * 1000),
                "truncated": combined_length > self.max_output_chars,
                "input_normalizations": list(input_normalizations or []),
            },
            ensure_ascii=False,
        )

    def _limit_resources(self) -> None:
        _limit_resources(self.timeout_s, self.memory_limit_mb)


_SYMBOLIC_FUNCTIONS = {
    "Abs",
    "Mod",
    "binomial",
    "ceiling",
    "cos",
    "factorial",
    "floor",
    "gcd",
    "lcm",
    "log",
    "sin",
    "sqrt",
    "tan",
}
_SYMBOLIC_CONSTANTS = {"E", "I", "pi"}
_SYMBOLIC_OPERATIONS = {"evaluate", "simplify", "expand", "factor", "solve"}
_SYMBOLIC_DOMAINS = {"integer", "rational", "real", "complex"}


@dataclass(frozen=True)
class SymbolicComputeTool:
    """Exact, bounded symbolic computation for AIME workers."""

    timeout_s: float = 3.0
    max_output_chars: int = 8000
    memory_limit_mb: int = 1024
    max_expression_chars: int = 2000
    max_expressions: int = 4
    max_variables: int = 4
    name: str = "symbolic_compute"
    description: str = (
        "Perform exact symbolic mathematics in a restricted SymPy process. Operations: "
        "evaluate, simplify, expand, factor, solve. Supply 1 to 4 pure expressions, never "
        "assignment statements such as x=2. Declare simple symbol names in variables and use "
        "substitutions for known values. Expressions use ** for powers; solve interprets every "
        "expression as equal to zero. Schema-invalid calls may be corrected without consuming "
        "the configured Action budget."
    )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "operation": {"type": "string", "enum": sorted(_SYMBOLIC_OPERATIONS)},
                "expressions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                    "maxItems": self.max_expressions,
                },
                "variables": {
                    "type": "array",
                    "items": {"type": "string"},
                    "maxItems": self.max_variables,
                },
                "substitutions": {
                    "type": "object",
                    "additionalProperties": {"type": "string"},
                },
                "domain": {"type": "string", "enum": sorted(_SYMBOLIC_DOMAINS)},
            },
            "required": ["operation", "expressions"],
            "additionalProperties": False,
        }

    def execute(self, arguments: dict[str, Any]) -> str:
        operation = str(arguments.get("operation", "")).strip().casefold()
        if operation not in _SYMBOLIC_OPERATIONS:
            raise ValueError("symbolic_compute operation is not supported")
        expressions = arguments.get("expressions")
        if not isinstance(expressions, list) or not 1 <= len(expressions) <= self.max_expressions:
            raise ValueError(f"symbolic_compute requires 1..{self.max_expressions} expressions")
        if not all(isinstance(value, str) and value.strip() for value in expressions):
            raise ValueError("symbolic_compute expressions must be non-empty strings")
        variables = arguments.get("variables", [])
        if not isinstance(variables, list) or len(variables) > self.max_variables:
            raise ValueError(f"symbolic_compute accepts at most {self.max_variables} variables")
        if not all(_valid_identifier(value) for value in variables):
            raise ValueError("symbolic_compute variables must be simple identifiers")
        if len(set(variables)) != len(variables):
            raise ValueError("symbolic_compute variables must be unique")
        domain = str(arguments.get("domain", "complex")).strip().casefold()
        if domain not in _SYMBOLIC_DOMAINS:
            raise ValueError("symbolic_compute domain is not supported")
        substitutions = arguments.get("substitutions", {})
        if not isinstance(substitutions, dict) or not all(
            isinstance(key, str) and key in variables and isinstance(value, str)
            for key, value in substitutions.items()
        ):
            raise ValueError(
                "symbolic_compute substitutions must map declared variables to strings"
            )
        allowed_names = set(variables) | _SYMBOLIC_FUNCTIONS | _SYMBOLIC_CONSTANTS
        for expression in [*expressions, *substitutions.values()]:
            if len(expression) > self.max_expression_chars:
                raise ValueError("symbolic_compute expression is too long")
            error = _validate_expression(expression, allowed_names, symbolic=True)
            if error:
                raise ValueError(f"symbolic_compute rejected expression: {error}")
        return _run_json_action(
            runner=_SYMBOLIC_RUNNER,
            payload={
                "operation": operation,
                "expressions": expressions,
                "variables": variables,
                "substitutions": substitutions,
                "domain": domain,
            },
            timeout_s=self.timeout_s,
            memory_limit_mb=self.memory_limit_mb,
            max_output_chars=self.max_output_chars,
        )


_FINITE_FUNCTIONS = {"abs", "comb", "digit_sum", "gcd", "isqrt", "lcm", "perm"}


@dataclass(frozen=True)
class FiniteSearchTool:
    """Bounded integer enumeration with a deliberately small expression language."""

    timeout_s: float = 3.0
    max_output_chars: int = 8000
    memory_limit_mb: int = 1024
    max_variables: int = 4
    max_assignments: int = 200_000
    max_results_limit: int = 100
    max_expression_chars: int = 2000
    name: str = "finite_search"
    description: str = (
        "Enumerate a bounded integer or combinatorial search space using a safe expression "
        "condition. Ranges are inclusive. Allowed helpers: abs, gcd, lcm, isqrt, comb, perm, "
        "digit_sum."
    )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "variables": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "min": {"type": "integer"},
                            "max": {"type": "integer"},
                            "step": {"type": "integer", "minimum": 1},
                        },
                        "required": ["name", "min", "max"],
                        "additionalProperties": False,
                    },
                    "minItems": 1,
                    "maxItems": self.max_variables,
                },
                "condition": {"type": "string"},
                "return_expressions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                    "maxItems": 8,
                },
                "max_results": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": self.max_results_limit,
                },
            },
            "required": ["variables", "condition", "return_expressions"],
            "additionalProperties": False,
        }

    def execute(self, arguments: dict[str, Any]) -> str:
        raw_variables = arguments.get("variables")
        if not isinstance(raw_variables, list) or not 1 <= len(raw_variables) <= self.max_variables:
            raise ValueError(f"finite_search requires 1..{self.max_variables} variables")
        variables: list[dict[str, int | str]] = []
        sizes: list[int] = []
        names: set[str] = set()
        for raw in raw_variables:
            if not isinstance(raw, dict) or not _valid_identifier(raw.get("name")):
                raise ValueError("finite_search variable names must be simple identifiers")
            name = str(raw["name"])
            if name in names or name in _FINITE_FUNCTIONS:
                raise ValueError("finite_search variable names must be unique and not reserved")
            lower = _strict_int(raw.get("min"), "min")
            upper = _strict_int(raw.get("max"), "max")
            step = _strict_int(raw.get("step", 1), "step")
            if step <= 0 or lower > upper:
                raise ValueError("finite_search requires step > 0 and min <= max")
            size = ((upper - lower) // step) + 1
            names.add(name)
            sizes.append(size)
            variables.append({"name": name, "min": lower, "max": upper, "step": step})
        assignments = reduce(mul, sizes, 1)
        if assignments > self.max_assignments:
            raise ValueError(
                f"finite_search space {assignments} exceeds limit {self.max_assignments}"
            )
        condition = arguments.get("condition")
        returns = arguments.get("return_expressions")
        if not isinstance(condition, str) or not condition.strip():
            raise ValueError("finite_search condition must be a non-empty string")
        if (
            not isinstance(returns, list)
            or not 1 <= len(returns) <= 8
            or not all(isinstance(value, str) and value.strip() for value in returns)
        ):
            raise ValueError("finite_search requires 1..8 return expressions")
        allowed_names = names | _FINITE_FUNCTIONS
        for expression in [condition, *returns]:
            if len(expression) > self.max_expression_chars:
                raise ValueError("finite_search expression is too long")
            error = _validate_expression(expression, allowed_names, symbolic=False)
            if error:
                raise ValueError(f"finite_search rejected expression: {error}")
        max_results = _strict_int(arguments.get("max_results", 50), "max_results")
        if not 1 <= max_results <= self.max_results_limit:
            raise ValueError(f"finite_search max_results must be in 1..{self.max_results_limit}")
        return _run_json_action(
            runner=_FINITE_SEARCH_RUNNER,
            payload={
                "variables": variables,
                "condition": condition,
                "return_expressions": returns,
                "max_results": max_results,
            },
            timeout_s=self.timeout_s,
            memory_limit_mb=self.memory_limit_mb,
            max_output_chars=self.max_output_chars,
        )


def _validate_python(
    code: str, allowed_modules: frozenset[str] | set[str] = frozenset(_ALLOWED_PYTHON_MODULES)
) -> str | None:
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as exc:
        return f"SyntaxError: {exc.msg} (line {exc.lineno})"
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = (
                [alias.name for alias in node.names]
                if isinstance(node, ast.Import)
                else [str(node.module or "")]
            )
            for name in names:
                root = name.split(".", 1)[0]
                if root not in allowed_modules:
                    return f"ImportError: module {root!r} is not allowed"
        if isinstance(node, ast.Name) and node.id in _FORBIDDEN_PYTHON_NAMES:
            return f"SecurityError: name {node.id!r} is not allowed"
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            return f"SecurityError: attribute {node.attr!r} is not allowed"
    return None


def _valid_identifier(value: Any) -> bool:
    return (
        isinstance(value, str)
        and value.isidentifier()
        and not value.startswith("_")
        and len(value) <= 32
    )


def _strict_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"finite_search {field} must be an integer")
    return value


def _validate_expression(expression: str, allowed_names: set[str], *, symbolic: bool) -> str | None:
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        return f"SyntaxError: {exc.msg}"
    if sum(1 for _ in ast.walk(tree)) > 200:
        return "expression contains too many syntax nodes"
    common: tuple[type[ast.AST], ...] = (
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.Call,
        ast.Name,
        ast.Load,
        ast.Constant,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.FloorDiv,
        ast.Mod,
        ast.Pow,
        ast.UAdd,
        ast.USub,
    )
    finite_only: tuple[type[ast.AST], ...] = (
        ast.BoolOp,
        ast.Compare,
        ast.And,
        ast.Or,
        ast.Not,
        ast.Eq,
        ast.NotEq,
        ast.Lt,
        ast.LtE,
        ast.Gt,
        ast.GtE,
    )
    allowed_nodes = common if symbolic else (*common, *finite_only)
    for node in ast.walk(tree):
        if not isinstance(node, allowed_nodes):
            return f"syntax {type(node).__name__} is not allowed"
        if isinstance(node, ast.Name) and node.id not in allowed_names:
            return f"name {node.id!r} is not allowed"
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in allowed_names:
                return "only whitelisted direct function calls are allowed"
            if node.keywords:
                return "keyword arguments are not allowed"
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool):
                if symbolic:
                    return "boolean literals are not allowed"
            elif not isinstance(node.value, (int, float)):
                return "only numeric literals are allowed"
    return None


def _limit_resources(timeout_s: float, memory_limit_mb: int) -> None:
    import resource

    cpu_seconds = max(1, int(timeout_s) + 1)
    memory_bytes = memory_limit_mb * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
    resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
    resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
    resource.setrlimit(resource.RLIMIT_NOFILE, (16, 16))


def _run_json_action(
    *,
    runner: str,
    payload: dict[str, Any],
    timeout_s: float,
    memory_limit_mb: int,
    max_output_chars: int,
) -> str:
    started = time.monotonic()
    try:
        with tempfile.TemporaryDirectory(prefix="aime-action-") as workdir:
            completed = subprocess.run(
                [sys.executable, "-I", "-c", runner],
                cwd=workdir,
                env={"PYTHONIOENCODING": "utf-8"},
                input=json.dumps(payload, ensure_ascii=False),
                capture_output=True,
                text=True,
                timeout=timeout_s,
                check=False,
                preexec_fn=lambda: _limit_resources(timeout_s, memory_limit_mb),
            )
    except subprocess.TimeoutExpired:
        return json.dumps(
            {
                "status": "timeout",
                "error": f"execution exceeded {timeout_s:g} seconds",
                "elapsed_ms": round((time.monotonic() - started) * 1000),
                "truncated": False,
            },
            ensure_ascii=False,
        )
    stdout = completed.stdout.strip()
    if completed.returncode != 0:
        error = completed.stderr.strip() or "action process exited without a result"
        return json.dumps(
            {
                "status": "error",
                "error": error[:max_output_chars],
                "elapsed_ms": round((time.monotonic() - started) * 1000),
                "truncated": len(error) > max_output_chars,
            },
            ensure_ascii=False,
        )
    try:
        result = json.loads(stdout)
    except json.JSONDecodeError:
        result = {"status": "error", "error": "action returned invalid JSON"}
    if not isinstance(result, dict):
        result = {"status": "error", "error": "action returned a non-object result"}
    result["elapsed_ms"] = round((time.monotonic() - started) * 1000)
    result.setdefault("truncated", False)
    encoded = json.dumps(result, ensure_ascii=False)
    if len(encoded) > max_output_chars:
        return json.dumps(
            {
                "status": "error",
                "error": "action output exceeded the configured limit",
                "elapsed_ms": result["elapsed_ms"],
                "truncated": True,
            },
            ensure_ascii=False,
        )
    return encoded


_SYMBOLIC_RUNNER = r"""
import json
import sys

import sympy as sp
from sympy.parsing.sympy_parser import parse_expr

payload = json.loads(sys.stdin.read())
domain = payload["domain"]
assumptions = {
    "integer": {"integer": True},
    "rational": {"rational": True},
    "real": {"real": True},
    "complex": {},
}[domain]
symbols = {name: sp.Symbol(name, **assumptions) for name in payload["variables"]}
functions = {
    "Abs": sp.Abs,
    "Mod": sp.Mod,
    "binomial": sp.binomial,
    "ceiling": sp.ceiling,
    "cos": sp.cos,
    "factorial": sp.factorial,
    "floor": sp.floor,
    "gcd": sp.gcd,
    "lcm": sp.lcm,
    "log": sp.log,
    "sin": sp.sin,
    "sqrt": sp.sqrt,
    "tan": sp.tan,
    "E": sp.E,
    "I": sp.I,
    "pi": sp.pi,
}
local = {**symbols, **functions}
global_values = {
    "__builtins__": {},
    "Integer": sp.Integer,
    "Float": sp.Float,
    "Rational": sp.Rational,
}

def parse(value):
    return parse_expr(value, local_dict=local, global_dict=global_values, evaluate=True)

try:
    expressions = [parse(value) for value in payload["expressions"]]
    substitutions = {
        symbols[name]: parse(value) for name, value in payload["substitutions"].items()
    }
    expressions = [value.subs(substitutions) for value in expressions]
    operation = payload["operation"]
    if operation == "solve":
        solutions = sp.solve(expressions, list(symbols.values()), dict=True)
        result = [
            {str(key): sp.sstr(value) for key, value in solution.items()}
            for solution in solutions
        ]
    else:
        function = {
            "evaluate": sp.simplify,
            "simplify": sp.simplify,
            "expand": sp.expand,
            "factor": sp.factor,
        }[operation]
        result = [sp.sstr(function(value)) for value in expressions]
    print(json.dumps({"status": "ok", "operation": operation, "result": result}))
except Exception as exc:
    print(json.dumps({"status": "error", "error": f"{type(exc).__name__}: {exc}"}))
"""


_FINITE_SEARCH_RUNNER = r"""
import itertools
import json
import math
import sys

payload = json.loads(sys.stdin.read())

def digit_sum(value):
    return sum(int(character) for character in str(abs(int(value))))

functions = {
    "abs": abs,
    "comb": math.comb,
    "digit_sum": digit_sum,
    "gcd": math.gcd,
    "isqrt": math.isqrt,
    "lcm": math.lcm,
    "perm": math.perm,
}
try:
    condition = compile(payload["condition"], "<finite_search_condition>", "eval")
    returns = [
        compile(value, "<finite_search_return>", "eval")
        for value in payload["return_expressions"]
    ]
    ranges = [
        range(value["min"], value["max"] + 1, value["step"])
        for value in payload["variables"]
    ]
    names = [value["name"] for value in payload["variables"]]
    matches = []
    total_matches = 0
    checked = 0
    for values in itertools.product(*ranges):
        local = {**functions, **dict(zip(names, values))}
        checked += 1
        if bool(eval(condition, {"__builtins__": {}}, local)):
            total_matches += 1
            if len(matches) < payload["max_results"]:
                matches.append([
                    eval(expression, {"__builtins__": {}}, local)
                    for expression in returns
                ])
    print(json.dumps({
        "status": "ok",
        "result": {
            "return_expressions": payload["return_expressions"],
            "matches": matches,
            "match_count": total_matches,
            "checked": checked,
        },
        "truncated": total_matches > len(matches),
    }))
except Exception as exc:
    print(json.dumps({"status": "error", "error": f"{type(exc).__name__}: {exc}"}))
"""


def _stream_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value


@dataclass(frozen=True)
class SearchServiceTool:
    """Small client for the Apache-2.0 SESA/Search-R1 `/retrieve` contract."""

    service_url: str
    top_k: int = 3
    timeout_s: float = 120.0
    max_queries: int = 1
    passage_chars: int = 0
    name: str = "search"
    description: str = (
        "Search the local evidence corpus with exactly one query string. "
        "Use another Action call with a revised query when the evidence is insufficient."
    )
    rollout_deadline: RolloutDeadline | None = None

    def set_deadline_context(self, deadline: RolloutDeadline | None) -> None:
        object.__setattr__(self, "rollout_deadline", deadline)

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
            },
            "required": ["query"],
            "additionalProperties": False,
        }

    def execute(self, arguments: dict[str, Any]) -> str:
        return self.execute_with_deadline(arguments, deadline=self.rollout_deadline)

    def execute_with_deadline(
        self, arguments: dict[str, Any], *, deadline: RolloutDeadline | None,
    ) -> str:
        """Explicit trajectory context avoids mutating a shared search client."""
        if deadline is not None:
            deadline.check("retrieval_request_start")
        queries = arguments.get("queries", arguments.get("query_list", arguments.get("query")))
        if isinstance(queries, str):
            queries = [queries]
        if (
            not isinstance(queries, list)
            or not queries
            or not all(isinstance(query, str) and query.strip() for query in queries)
        ):
            raise ValueError("search requires a non-empty query or query list")
        if len(queries) > self.max_queries:
            noun = "query" if self.max_queries == 1 else "queries"
            raise ValueError(f"search accepts at most {self.max_queries} {noun} per Action")
        payload = json.dumps(
            {"queries": queries, "topk": self.top_k, "return_scores": True}
        ).encode("utf-8")
        request = Request(
            self.service_url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        timeout_s = self.timeout_s
        if deadline is not None:
            timeout_s = min(
                timeout_s,
                deadline.request_budget_s("retrieval_request"),
            )
        try:
            with urlopen(request, timeout=timeout_s) as response:
                body = json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"local retrieval service failed: {exc}") from exc
        if deadline is not None:
            deadline.check("retrieval_request_complete")
        if not isinstance(body, dict):
            raise RuntimeError("local retrieval service returned an invalid response")
        results = body.get("result")
        if not isinstance(results, list):
            raise RuntimeError("local retrieval service returned no result list")
        if len(results) != len(queries):
            raise RuntimeError("local retrieval service returned mismatched query/result counts")
        from .public_evidence import public_search_results

        projected = public_search_results([_document_hits(group) for group in results])
        # Search-R1 text: `Doc i(Title: "T") text` per hit.
        return "\n".join(
            format_search_r1(group, self.passage_chars) for group in projected
        )


def _document_hits(group: object) -> object:
    """Wrap the retrieval service's flat `{"id","title","text","score"}` hits as `{"document", "score"}`."""

    if not isinstance(group, list):
        return group
    return [
        {"document": {key: hit[key] for key in ("id", "title", "text") if key in hit},
         **{key: hit[key] for key in ("score",) if key in hit}}
        if isinstance(hit, dict) and "document" not in hit and "text" in hit else hit
        for hit in group
    ]


def format_search_r1(hits: list[dict[str, Any]], passage_chars: int = 0) -> str:
    """Search-R1 `_passages2string` formatting of the hits."""

    lines = []
    for index, hit in enumerate(hits, 1):
        document = hit["document"]
        text = str(document.get("text", document.get("contents", "")))
        if passage_chars and len(text) > passage_chars:
            text = text[:passage_chars].rstrip() + " ..."
        lines.append(f'Doc {index}(Title: "{document.get("title", "")}") {text}\n')
    return "".join(lines)
