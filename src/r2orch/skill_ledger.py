"""Arm-coupled scaffold ledger (R2Orch ``[r2orch.arms]``).

Design.  Every task runs ``hidden + visible`` independent Director siblings.  One
candidate card k is chosen per task among the cards that retrieval shows for it (the "on"
context); the ``hidden`` siblings see that context without k (and only without k), the
``visible`` siblings see all of it.  A task with at least one completed (outcome-trusted)
sibling in each arm adds one record to card k::

    z = ybar_v - ybar_h        (raw 0/1 correctness, never the cost-adjusted Director reward)

Posterior of the population effect Delta_k of showing k (pooled over every scope that holds the
same card, i.e. the same id, version and content): the Bayesian bootstrap (Rubin 1981),
Dirichlet(1, ..., 1) weights w over the card's task records and Delta = sum_i w_i z_i.  Records
with equal z are aggregated (a sum of Dirichlet components is Dirichlet with the summed
parameters), so one draw costs one Gamma variate per distinct z value, not per record.  The
summary reports the exact posterior mean (the record mean), E[Delta^+], the 5 % / 95 % quantiles
and n.

* ``decide``: ``retire`` once ``n >= min_tasks`` and ``E[Delta^+] <= epsilon`` (the card no longer
  helps by more than epsilon); else ``admit`` once ``n >= min_tasks`` and the ``admit_quantile``
  quantile is > 0; else ``undecided``.
* ``thompson_pick``: one posterior draw per candidate, the smallest wins, so evidence concentrates
  on the most retirable cards.  The Thompson posterior adds one pseudo-record drawn from the wide
  prior N(0, prior_sd^2) to the bootstrap (Dirichlet weight 1, like a Dirichlet-process posterior
  with concentration 1): a card without data gets exactly a prior draw, and a card seen once is
  not a point mass at its single contrast.
* Randomness is a ``random.Random`` seeded from sha256(base_seed, cycle, purpose, key): the
  plan, the decisions and every summary are reproducible across resumes.

Delta_k is the effect of k on the CURRENT Director, so a retirement must be revisable:

* ``window_cycles`` W: every posterior used for a decision or a Thompson draw reads only the
  records of the last W cycles of evidence (cycle > latest - W, ``decision_contrasts``).
* ``audit_share`` a: a task whose scope holds ledger-retired cards shows, with probability a,
  one of them (uniform) to its visible arm instead of the Thompson pick, so a retired card keeps
  being measured under the current policy.  Records of such tasks carry ``audit: true``.
* ``reinstate_quantile`` q: a retired card whose (windowed) posterior would admit it at
  quantile q (n >= min_tasks, E[Delta^+] > epsilon, Q_q > 0) is reinstated.

The ledger is one JSON file under the run's state directory (``<run_root>/state``).  The records
of a cycle are written once by the cycle's post-collection step (an interrupted step rewrites the
same cycle, never duplicates it); readers filter by cycle, so a plan, a Frontier prior or a
decision only ever sees cycles that were complete when it was first computed.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import random
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

LEDGER_SCHEMA = "r2orch_skill_ledger@1"
RECORD_SCHEMA = "r2orch_skill_ledger_record@1"
SUMMARY_SCHEMA = "r2orch_skill_ledger_summary@1"
ARM_PLAN_SCHEMA = "r2orch_skill_arm_plan@1"
ARM_REPORT_SCHEMA = "r2orch_skill_arms@1"
LEDGER_FILENAME = "skill_ledger.json"
ARMS = ("hidden", "visible")
# Key of the PATS snapshot view that lists each scope's ledger-retired cards (audit candidates).
RETIRED_VIEW_KEY = "ledger_retired"
DEFAULT_PRIOR_SD = 0.25
DEFAULT_SCOPE_PRIOR = 0.5
# Contrasts are means of a handful of 0/1 outcomes; rounding merges 2/3 and 1 - 1/3.
_DIGITS = 12


# --------------------------------------------------------------------------- identity


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()


def card_key(skill_id: str, version: int, content_sha256: str) -> str:
    """Ledger identity of one card: id, version and content (two scopes share a seed card)."""
    return f"{skill_id}@{int(version)}#{str(content_sha256)[:12]}"


def record_card_key(record: Mapping[str, Any]) -> str:
    """Key of a PATS / SkillStore card record ``{"card": {...}, "version": v, ...}``."""
    card = record["card"]
    return card_key(str(card["skill_id"]), int(record.get("version", 1)), canonical_sha256(card))


def derived_seed(*parts: Any) -> int:
    digest = hashlib.sha256("\0".join(str(part) for part in parts).encode()).digest()
    return int.from_bytes(digest[:8], "big")


def ledger_rng(base_seed: int, cycle: int, *salt: Any) -> random.Random:
    """Deterministic generator for (base_seed, cycle) and a purpose/key salt."""
    return random.Random(derived_seed("skill_ledger", int(base_seed), int(cycle), *salt))


def skill_ledger_path(r2orch: Any) -> Path:
    """``<run_root>/state/skill_ledger.json``."""
    return Path(r2orch.paths.run_root) / "state" / LEDGER_FILENAME


def arm_assignment(cycle: int, task_id: str, hidden: int, *, siblings: int) -> tuple[str, ...]:
    """Seeded permutation of the sibling slots; the first ``hidden`` slots are the hidden arm."""
    if not 0 <= int(hidden) <= int(siblings):
        raise ValueError("hidden siblings must be within [0, siblings]")
    digest = hashlib.sha256(f"{int(cycle)}\0{task_id}\0skill_arms".encode()).digest()
    order = list(range(int(siblings)))
    random.Random(int.from_bytes(digest[:8], "big")).shuffle(order)
    chosen = set(order[: int(hidden)])
    return tuple("hidden" if index in chosen else "visible" for index in range(int(siblings)))


# ------------------------------------------------------------------- posterior math


def _grouped(contrasts: Iterable[float]) -> list[tuple[float, int]]:
    counts = Counter(round(float(value), _DIGITS) for value in contrasts)
    return sorted(counts.items())


def _mix(groups: Sequence[tuple[float, int]], rng: random.Random, prior_atom: float | None) -> float:
    """One draw of sum_j W_j z_j with (W_prior, W_1..W_J) ~ Dirichlet(1, m_1, ..., m_J)."""
    total = weighted = 0.0
    if prior_atom is not None:
        gamma = rng.gammavariate(1.0, 1.0)
        total += gamma
        weighted += gamma * prior_atom
    for value, count in groups:
        gamma = rng.gammavariate(float(count), 1.0)
        total += gamma
        weighted += gamma * value
    return weighted / total


def bootstrap_draws(contrasts: Sequence[float], *, draws: int, rng: random.Random) -> list[float]:
    """Bayesian-bootstrap draws of the mean contrast (empty without records)."""
    groups = _grouped(contrasts)
    if not groups:
        return []
    if len(groups) == 1:
        return [groups[0][0]] * int(draws)
    return [_mix(groups, rng, None) for _ in range(int(draws))]


def quantile(sorted_values: Sequence[float], q: float) -> float:
    """Linear interpolation between order statistics (numpy's default)."""
    if not sorted_values:
        raise ValueError("quantile of an empty sample")
    position = (len(sorted_values) - 1) * float(q)
    lower = math.floor(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = position - lower
    return float(sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * fraction)


@dataclass(frozen=True)
class EffectPosterior:
    n: int
    mean: float | None
    E_delta_plus: float | None
    q05: float | None
    q95: float | None
    q_admit: float | None
    admit_quantile: float
    draws: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def effect_posterior(
    contrasts: Sequence[float],
    *,
    draws: int = 2000,
    rng: random.Random,
    admit_quantile: float = 0.05,
) -> EffectPosterior:
    values = [float(value) for value in contrasts]
    if not values:
        return EffectPosterior(0, None, None, None, None, None, float(admit_quantile), 0)
    sample = sorted(bootstrap_draws(values, draws=draws, rng=rng))
    return EffectPosterior(
        n=len(values),
        mean=math.fsum(values) / len(values),
        E_delta_plus=math.fsum(max(value, 0.0) for value in sample) / len(sample),
        q05=quantile(sample, 0.05),
        q95=quantile(sample, 0.95),
        q_admit=quantile(sample, float(admit_quantile)),
        admit_quantile=float(admit_quantile),
        draws=len(sample),
    )


def decide(posterior: EffectPosterior, *, min_tasks: int, epsilon: float) -> str:
    """``retire`` / ``admit`` / ``undecided`` (retire is checked first)."""
    if posterior.n < int(min_tasks) or posterior.n == 0:
        return "undecided"
    if float(posterior.E_delta_plus) <= float(epsilon):
        return "retire"
    if float(posterior.q_admit) > 0.0:
        return "admit"
    return "undecided"


def thompson_draw(
    contrasts: Sequence[float], rng: random.Random, *, prior_sd: float = DEFAULT_PRIOR_SD
) -> float:
    """One Thompson draw of Delta: the bootstrap plus one pseudo-record from N(0, prior_sd^2)."""
    prior_atom = rng.gauss(0.0, float(prior_sd))
    groups = _grouped(contrasts)
    if not groups:
        return prior_atom
    return _mix(groups, rng, prior_atom)


def thompson_pick(
    candidates: Mapping[str, Sequence[float]],
    rng: random.Random,
    *,
    prior_sd: float = DEFAULT_PRIOR_SD,
) -> tuple[str | None, dict[str, float]]:
    """Draw once per candidate (sorted key order) and return (argmin key, draws)."""
    draws = {
        key: thompson_draw(candidates[key], rng, prior_sd=prior_sd) for key in sorted(candidates)
    }
    if not draws:
        return None, {}
    return min(draws, key=lambda key: (draws[key], key)), draws


def audit_choice(
    retired: Mapping[str, Any], share: float, rng: random.Random
) -> str | None:
    """``audit_share``: with probability ``share`` one of the ``retired`` card keys (uniform).

    One uniform variate decides the audit, a second (only on an audit) picks the card among the
    sorted keys.  No variate is drawn without a retired card.
    """
    if not retired:
        return None
    if rng.random() >= float(share):
        return None
    keys = sorted(retired)
    return keys[rng.randrange(len(keys))]


def choose_arm_card(
    selected: Sequence[Mapping[str, Any]], draws: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any] | None:
    """Card k of a task among the retrieved cards: the smallest planned Thompson draw.

    ``selected`` is the retrieval manifest (``[{"id", "version"}]``), ``draws`` the plan's
    ``{card_id: {"version", "key", "draw"}}``.  Retrieval is deterministic per snapshot and
    query, so every sibling of the task resolves the same k.
    """
    candidates = []
    for item in selected:
        planned = draws.get(str(item.get("id")))
        if planned is None or int(planned.get("version", -1)) != int(item.get("version", -2)):
            continue
        candidates.append(
            {
                "id": str(item["id"]),
                "version": int(item["version"]),
                "key": str(planned["key"]),
                "draw": float(planned["draw"]),
            }
        )
    if not candidates:
        return None
    return min(candidates, key=lambda item: (item["draw"], item["key"]))


# ------------------------------------------------------------------ Proposer reward


def bell(p: float) -> float:
    """B(p) = 4 p (1 - p): 1 at p = 1/2, 0 at the endpoints."""
    return 4.0 * float(p) * (1.0 - float(p))


def shrunk_rate(successes: float, n: int, *, prior: float, strength: float) -> float:
    """Beta shrinkage (s + m * prior) / (n + m) of a success rate toward the scope prior."""
    denominator = int(n) + float(strength)
    if denominator <= 0.0:
        return float(prior)
    return (float(successes) + float(strength) * float(prior)) / denominator


def scope_hidden_prior(
    records: Iterable[Mapping[str, Any]],
    scope: str,
    *,
    decay: float,
    default: float = DEFAULT_SCOPE_PRIOR,
) -> tuple[float, int]:
    """EMA over cycles of the scope's pooled hidden-arm success; (value, cycles observed).

    The first observed cycle initialises the EMA; ``default`` (1/2, where B is largest)
    stands in until the scope has a record.
    """
    by_cycle: dict[int, list[float]] = defaultdict(lambda: [0.0, 0.0])
    for record in records:
        if record.get("scope") != scope:
            continue
        bucket = by_cycle[int(record["cycle"])]
        bucket[0] += float(record["s_h"])
        bucket[1] += float(record["n_h"])
    value, observed = None, 0
    for cycle in sorted(by_cycle):
        successes, total = by_cycle[cycle]
        if total <= 0:
            continue
        rate = successes / total
        value = rate if value is None else float(decay) * value + (1.0 - float(decay)) * rate
        observed += 1
    return (float(default) if value is None else float(value)), observed


def task_duration(durations: Iterable[Any]) -> float | None:
    """T_q: mean ``duration_s`` of a task's completed siblings (finite positive values only)."""
    values = []
    for value in durations:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number) and number > 0.0:
            values.append(number)
    return math.fsum(values) / len(values) if values else None


def compute_factor(task_seconds: float | None, dataset_seconds: float | None, exponent: float) -> float:
    """(Tbar_d / T_q) ** exponent; 1 when disabled or a duration is unknown."""
    if float(exponent) <= 0.0 or not task_seconds or not dataset_seconds:
        return 1.0
    return (float(dataset_seconds) / float(task_seconds)) ** float(exponent)


def duration_ema(
    cycles: Mapping[str, Mapping[str, Any]], dataset: str, *, decay: float, through_cycle: int
) -> tuple[float | None, int]:
    """EMA over cycles <= ``through_cycle`` of the dataset's mean T_q; first cycle initialises it."""
    value, observed = None, 0
    for cycle in sorted(int(key) for key in cycles):
        if cycle > int(through_cycle):
            break
        entry = ((cycles[str(cycle)].get("durations") or {}).get(dataset)) or {}
        count = int(entry.get("count", 0))
        if count <= 0:
            continue
        mean = float(entry["sum"]) / count
        value = mean if value is None else float(decay) * value + (1.0 - float(decay)) * mean
        observed += 1
    return value, observed


# ------------------------------------------------------------------ task records


def task_arm_record(
    *,
    cycle: int,
    task_id: str,
    scope: str,
    siblings: Sequence[tuple[float, Mapping[str, Any]]],
) -> tuple[dict[str, Any] | None, str]:
    """One ledger record from a task's outcome-trusted siblings ``[(reward, r2orch), ...]``.

    Returns ``(record, "recorded")`` or ``(None, reason)``.
    """
    armed = [
        (float(reward), meta)
        for reward, meta in siblings
        if meta.get("skill_arm") in ARMS
    ]
    if not armed:
        return None, "no_arm"
    keys = {str((meta.get("arm_card") or {}).get("key", "")) for _reward, meta in armed}
    if len(keys) != 1 or "" in keys:
        return None, "arm_card_mismatch"
    hidden = [reward for reward, meta in armed if meta["skill_arm"] == "hidden"]
    visible = [reward for reward, meta in armed if meta["skill_arm"] == "visible"]
    if not hidden or not visible:
        return None, "missing_arm"
    card = dict(armed[0][1]["arm_card"])
    ybar_h = math.fsum(hidden) / len(hidden)
    ybar_v = math.fsum(visible) / len(visible)
    record = {
        "schema": RECORD_SCHEMA,
        "cycle": int(cycle),
        "task_id": str(task_id),
        "scope": str(scope),
        "card": str(card["key"]),
        "card_id": str(card["id"]),
        "card_version": int(card["version"]),
        "n_h": len(hidden),
        "n_v": len(visible),
        "s_h": math.fsum(hidden),
        "s_v": math.fsum(visible),
        "ybar_h": ybar_h,
        "ybar_v": ybar_v,
        "z": ybar_v - ybar_h,
        "hidden_conditions": sorted(
            {str(meta.get("scaffold_condition", "")) for _r, meta in armed if meta["skill_arm"] == "hidden"}
        ),
    }
    if card.get("audit"):
        # ``audit_share``: k was a ledger-retired card shown to the visible arm only.
        record["audit"] = True
    return record, "recorded"


# ----------------------------------------------------------------------- the store


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=1, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


class SkillLedger:
    """JSON-persisted per-card task contrasts with the arms decision rules."""

    def __init__(
        self,
        path: Path,
        *,
        base_seed: int = 0,
        epsilon: float = 0.02,
        admit_quantile: float = 0.05,
        min_tasks: int = 12,
        bootstrap_draws: int = 2000,
        scope_prior_decay: float = 0.9,
        thompson_prior_sd: float = DEFAULT_PRIOR_SD,
        window_cycles: int = 30,
        audit_share: float = 0.3,
        reinstate_quantile: float = 0.05,
    ) -> None:
        self.path = Path(path)
        self.base_seed = int(base_seed)
        self.epsilon = float(epsilon)
        self.admit_quantile = float(admit_quantile)
        self.min_tasks = int(min_tasks)
        self.bootstrap_draws = int(bootstrap_draws)
        self.scope_prior_decay = float(scope_prior_decay)
        self.thompson_prior_sd = float(thompson_prior_sd)
        self.window_cycles = int(window_cycles)
        self.audit_share = float(audit_share)
        self.reinstate_quantile = float(reinstate_quantile)

    @classmethod
    def from_arms(cls, path: Path, arms: Any, *, base_seed: int = 0) -> SkillLedger:
        """Build from ``config.r2orch.arms`` (``ArmsConfig``)."""
        return cls(
            path,
            base_seed=base_seed,
            epsilon=float(arms.epsilon),
            admit_quantile=float(arms.admit_quantile),
            min_tasks=int(arms.min_tasks),
            bootstrap_draws=int(arms.bootstrap_draws),
            scope_prior_decay=float(arms.scope_prior_decay),
            thompson_prior_sd=float(arms.thompson_prior_sd),
            window_cycles=int(arms.window_cycles),
            audit_share=float(arms.audit_share),
            reinstate_quantile=float(arms.reinstate_quantile),
        )

    def window_start(self, latest_cycle: int) -> int:
        """First cycle of the evidence window ending at ``latest_cycle``."""
        return int(latest_cycle) - self.window_cycles + 1

    # -- persistence ---------------------------------------------------------------

    @contextmanager
    def _locked(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.with_suffix(".lock").open("a") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            yield

    def _read(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"schema": LEDGER_SCHEMA, "records": [], "cycles": {}}
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if payload.get("schema") != LEDGER_SCHEMA:
            raise ValueError(f"unknown skill ledger schema in {self.path}")
        return payload

    def append_cycle(
        self,
        cycle: int,
        records: Sequence[Mapping[str, Any]],
        *,
        report: Mapping[str, Any] | None = None,
        durations: Mapping[str, Sequence[float]] | None = None,
    ) -> None:
        """Store the records of one cycle, replacing any earlier write of that cycle.

        ``durations`` maps a dataset to the T_q of the cycle's Proposer-chosen tasks; the cycle
        keeps their sum and count for the dataset duration EMA.
        """
        cycle = int(cycle)
        if any(int(record["cycle"]) != cycle for record in records):
            raise ValueError("every ledger record must belong to the appended cycle")
        with self._locked():
            payload = self._read()
            kept = [row for row in payload["records"] if int(row["cycle"]) != cycle]
            kept.extend(json.loads(json.dumps(list(records))))
            kept.sort(key=lambda row: (int(row["cycle"]), str(row["task_id"])))
            payload["records"] = kept
            payload["cycles"][str(cycle)] = {
                "records": len(records),
                **({"report": json.loads(json.dumps(dict(report)))} if report else {}),
                **(
                    {
                        "durations": {
                            str(dataset): {
                                "sum": math.fsum(float(value) for value in values),
                                "count": len(values),
                            }
                            for dataset, values in sorted(durations.items())
                            if values
                        }
                    }
                    if durations
                    else {}
                ),
            }
            _atomic_json(self.path, payload)

    def dataset_durations(self, *, through_cycle: int) -> dict[str, dict[str, Any]]:
        """``{dataset: {"value": Tbar_d, "cycles": n}}`` over the cycles <= ``through_cycle``."""
        cycles = self._read().get("cycles") or {}
        datasets = sorted(
            {name for entry in cycles.values() for name in (entry.get("durations") or {})}
        )
        result = {}
        for dataset in datasets:
            value, observed = duration_ema(
                cycles, dataset, decay=self.scope_prior_decay, through_cycle=through_cycle
            )
            if value is not None:
                result[dataset] = {"value": value, "cycles": observed}
        return result

    def records(
        self,
        *,
        before_cycle: int | None = None,
        through_cycle: int | None = None,
        since_cycle: int | None = None,
    ) -> list[dict[str, Any]]:
        rows = self._read()["records"]
        if before_cycle is not None:
            rows = [row for row in rows if int(row["cycle"]) < int(before_cycle)]
        if through_cycle is not None:
            rows = [row for row in rows if int(row["cycle"]) <= int(through_cycle)]
        if since_cycle is not None:
            rows = [row for row in rows if int(row["cycle"]) >= int(since_cycle)]
        return rows

    def contrasts_by_card(
        self,
        *,
        before_cycle: int | None = None,
        through_cycle: int | None = None,
        since_cycle: int | None = None,
    ) -> dict[str, list[float]]:
        grouped: dict[str, list[float]] = defaultdict(list)
        for row in self.records(
            before_cycle=before_cycle, through_cycle=through_cycle, since_cycle=since_cycle
        ):
            grouped[str(row["card"])].append(float(row["z"]))
        return dict(grouped)

    def decision_contrasts(self, *, through_cycle: int) -> dict[str, list[float]]:
        """Contrasts per card that a decision or a Thompson draw reads after ``through_cycle``.

        The records of the last ``window_cycles`` W cycles up to ``through_cycle`` (cycle >
        through_cycle - W).
        """
        return self.contrasts_by_card(
            through_cycle=through_cycle, since_cycle=self.window_start(through_cycle)
        )

    def audit_records(self, *, cycle: int) -> list[dict[str, Any]]:
        """The records of ``cycle`` whose card k was a retired card under audit."""
        return [
            row
            for row in self.records(through_cycle=cycle, since_cycle=cycle)
            if row.get("audit")
        ]

    # -- decisions -----------------------------------------------------------------

    def posterior(
        self,
        card: str,
        *,
        cycle: int,
        contrasts: Sequence[float] | None = None,
        admit_quantile: float | None = None,
    ) -> EffectPosterior:
        """Posterior of ``card`` from the records through ``cycle`` (seeded by base seed, cycle).

        Without ``contrasts`` it reads ``decision_contrasts`` (the window).
        ``admit_quantile`` overrides the quantile reported as ``q_admit``; the draws are the same,
        so one card's posteriors at one cycle agree on every other field.
        """
        values = (
            list(contrasts)
            if contrasts is not None
            else self.decision_contrasts(through_cycle=cycle).get(card, [])
        )
        return effect_posterior(
            values,
            draws=self.bootstrap_draws,
            rng=ledger_rng(self.base_seed, cycle, "posterior", card),
            admit_quantile=self.admit_quantile if admit_quantile is None else admit_quantile,
        )

    def decide(self, posterior: EffectPosterior) -> str:
        return decide(posterior, min_tasks=self.min_tasks, epsilon=self.epsilon)

    def scope_hidden_priors(self, *, before_cycle: int) -> dict[str, dict[str, float]]:
        rows = self.records(before_cycle=before_cycle)
        result = {}
        for scope in sorted({str(row["scope"]) for row in rows}):
            value, observed = scope_hidden_prior(rows, scope, decay=self.scope_prior_decay)
            result[scope] = {"value": value, "cycles": observed}
        return result


__all__ = [
    "ARMS",
    "ARM_PLAN_SCHEMA",
    "ARM_REPORT_SCHEMA",
    "LEDGER_SCHEMA",
    "RETIRED_VIEW_KEY",
    "SUMMARY_SCHEMA",
    "EffectPosterior",
    "SkillLedger",
    "arm_assignment",
    "audit_choice",
    "bell",
    "bootstrap_draws",
    "card_key",
    "choose_arm_card",
    "compute_factor",
    "decide",
    "duration_ema",
    "effect_posterior",
    "ledger_rng",
    "quantile",
    "record_card_key",
    "scope_hidden_prior",
    "shrunk_rate",
    "skill_ledger_path",
    "task_arm_record",
    "task_duration",
    "thompson_draw",
    "thompson_pick",
]
