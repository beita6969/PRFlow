from __future__ import annotations

import copy
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from .contracts import AgentNode, Relation, RelationType
from .unified_contract import ACTION_PROTOCOL, PROTOCOL


class GraphValidationError(ValueError):
    """Raised when a canvas mutation violates graph execution semantics."""


@dataclass
class MutationResult:
    message: str
    dirty_agents: set[str] = field(default_factory=set)
    changed: bool = True
    change_reason: str = "graph_changed"


class MultiAgentGraph:
    """Layered mixed graph of Agents.

    Directed edges must go from a lower layer to a higher layer. Bidirectional
    edges connect agents in the same layer and form one bounded synchronous block.
    """

    def __init__(self, *, max_agents: int = 8, runtime_routes: tuple[str, ...] = ()) -> None:
        self.max_agents = int(max_agents)
        self.runtime_routes = tuple(runtime_routes)
        self.nodes: dict[str, AgentNode] = {}
        self.directed_edges: set[tuple[str, str]] = set()
        self.bidirectional_edges: set[tuple[str, str]] = set()
        self.output_agent: str | None = None
        self.version = 0
        self._next_agent_index = 1

    def clone(self) -> MultiAgentGraph:
        return copy.deepcopy(self)

    def add_agent(self, agent_id: str | None = None) -> MutationResult:
        if len(self.nodes) >= self.max_agents:
            raise GraphValidationError(f"agent budget exceeded ({self.max_agents})")
        if agent_id is None:
            agent_id = self._allocate_agent_id()
        agent_id = str(agent_id).strip()
        if not agent_id:
            raise GraphValidationError("agent_id cannot be empty")
        if agent_id in self.nodes:
            raise GraphValidationError(f"agent already exists: {agent_id}")
        self.nodes[agent_id] = AgentNode(agent_id=agent_id)
        if self.runtime_routes:
            self.nodes[agent_id].metadata["model_selection_required"] = True
        self._advance_index_from_id(agent_id)
        self._touch()
        return MutationResult(
            f"added agent {agent_id}",
            {agent_id},
            change_reason="agent_added",
        )

    def set_prompt(
        self,
        agent_id: str,
        prompt: str,
        *,
        runtime_route: str | None = None,
        metadata_updates: dict[str, Any] | None = None,
    ) -> MutationResult:
        node = self.require_node(agent_id)
        prompt = str(prompt).strip()
        if not prompt:
            raise GraphValidationError("agent prompt cannot be empty")
        route: str | None = None
        if runtime_route is not None:
            route = str(runtime_route).strip()
            if not route:
                raise GraphValidationError("runtime_route cannot be empty")
        next_metadata = copy.deepcopy(node.metadata)
        if route is not None:
            next_metadata["runtime_route"] = route
        if metadata_updates:
            next_metadata.update(copy.deepcopy(metadata_updates))
        if node.prompt == prompt and node.metadata == next_metadata:
            return MutationResult(
                f"prompt for {agent_id} already has the same effective Worker input",
                changed=False,
                change_reason="no_effective_change",
            )
        node.prompt = prompt
        node.metadata = next_metadata
        self._touch()
        dirty = self.dirty_closure({agent_id})
        return MutationResult(
            f"set prompt for {agent_id}",
            dirty,
            change_reason="worker_input_changed",
        )

    def set_model(self, agent_id: str, runtime_route: str) -> MutationResult:
        node = self.require_node(agent_id)
        route = str(runtime_route).strip()
        if not node.prompt_configured:
            raise GraphValidationError("SET_MODEL requires a configured responsibility prompt")
        if not route or (self.runtime_routes and route not in self.runtime_routes):
            raise GraphValidationError(f"unknown Worker runtime_route: {route or '<missing>'}")
        if node.metadata.get("runtime_route") == route:
            return MutationResult(
                f"model for {agent_id} unchanged",
                changed=False,
                change_reason="no_effective_change",
            )
        node.metadata["runtime_route"] = route
        node.metadata["model_selection_source"] = "director_set_model"
        self._touch()
        return MutationResult(
            f"set model for {agent_id} to {route}",
            self.dirty_closure({agent_id}),
            change_reason="model_changed",
        )

    def configure_action_environment(
        self,
        agent_id: str,
        *,
        adapter_id: str,
        action_names: tuple[str, ...],
        initial_action_budget: int,
        revision_action_budget: int,
        total_action_budget: int,
        capability_policy: dict[str, object] | None = None,
    ) -> MutationResult:
        node = self.require_node(agent_id)
        adapter_id = str(adapter_id).strip()
        initial_action_budget = int(initial_action_budget)
        revision_action_budget = int(revision_action_budget)
        total_action_budget = int(total_action_budget)
        if not adapter_id:
            raise GraphValidationError("action adapter_id cannot be empty")
        if min(initial_action_budget, revision_action_budget, total_action_budget) < 0:
            raise GraphValidationError("action budgets must be non-negative")
        if initial_action_budget + revision_action_budget > total_action_budget:
            raise GraphValidationError("phase action budgets exceed total action budget")
        if action_names and total_action_budget <= 0:
            raise GraphValidationError("visible actions require a positive total action budget")
        if not action_names and any(
            (initial_action_budget, revision_action_budget, total_action_budget)
        ):
            raise GraphValidationError("action budgets must be zero when no actions are visible")
        next_allowed_tools = tuple(dict.fromkeys(action_names))
        next_metadata = copy.deepcopy(node.metadata)
        next_metadata["action_adapter"] = adapter_id
        if capability_policy is not None:
            next_metadata["dataset_capability_policy"] = copy.deepcopy(capability_policy)
        if (
            node.allowed_tools == next_allowed_tools
            and node.initial_tool_budget == initial_action_budget
            and node.revision_tool_budget == revision_action_budget
            and node.total_tool_budget == total_action_budget
            and node.operation_policy_configured
            and node.metadata == next_metadata
        ):
            return MutationResult(
                f"{adapter_id} action environment for {agent_id} is already configured",
                changed=False,
                change_reason="no_effective_change",
            )
        node.allowed_tools = next_allowed_tools
        node.initial_tool_budget = initial_action_budget
        node.revision_tool_budget = revision_action_budget
        node.total_tool_budget = total_action_budget
        node.operation_policy_configured = True
        node.metadata = next_metadata
        self._touch()
        return MutationResult(
            f"configured {adapter_id} action environment for {agent_id}",
            self.dirty_closure({agent_id}),
            change_reason="action_environment_changed",
        )

    def set_layer(self, agent_id: str, layer: int) -> MutationResult:
        node = self.require_node(agent_id)
        layer = int(layer)
        if layer < 0:
            raise GraphValidationError("layer must be >= 0")
        if node.layer == layer:
            return MutationResult(
                f"agent {agent_id} is already in layer {layer}",
                changed=False,
                change_reason="no_effective_change",
            )
        old_component = self.bidirectional_component(agent_id)
        old_layer = node.layer
        node.layer = layer
        try:
            self.validate_relations()
        except GraphValidationError:
            node.layer = old_layer
            raise
        self._touch()
        dirty = self.dirty_closure(old_component | self.bidirectional_component(agent_id))
        return MutationResult(
            f"set layer of {agent_id} to {layer}",
            dirty,
            change_reason="layer_changed",
        )

    def set_relation(
        self,
        source: str,
        target: str,
        relation_type: RelationType | str,
    ) -> MutationResult:
        self.require_node(source)
        self.require_node(target)
        if source == target:
            raise GraphValidationError("self relations are not allowed")
        relation_type = RelationType(relation_type)
        if relation_type is RelationType.DIRECTED:
            if self.nodes[source].layer >= self.nodes[target].layer:
                raise GraphValidationError(
                    "directed relation requires layer(source) < layer(target)"
                )
            if self._pair(source, target) in self.bidirectional_edges:
                raise GraphValidationError("agent pair already has a bidirectional relation")
            edge = (source, target)
            if edge in self.directed_edges:
                raise GraphValidationError(
                    f"directed relation already exists: {source} -> {target}"
                )
            self.directed_edges.add(edge)
            seeds = self.bidirectional_component(target)
        else:
            if self.nodes[source].layer != self.nodes[target].layer:
                raise GraphValidationError(
                    "bidirectional relation requires both agents in the same layer"
                )
            if (source, target) in self.directed_edges or (target, source) in self.directed_edges:
                raise GraphValidationError("agent pair already has a directed relation")
            edge = self._pair(source, target)
            if edge in self.bidirectional_edges:
                raise GraphValidationError(
                    f"bidirectional relation already exists: {edge[0]} <-> {edge[1]}"
                )
            self.bidirectional_edges.add(edge)
            seeds = self.bidirectional_component(source)
        self._touch()
        return MutationResult(
            f"set {relation_type.value} relation between {source} and {target}",
            self.dirty_closure(seeds),
            change_reason="relation_added",
        )

    def remove_relation(
        self,
        source: str,
        target: str,
        relation_type: RelationType | str,
    ) -> MutationResult:
        self.require_node(source)
        self.require_node(target)
        relation_type = RelationType(relation_type)
        old_component = self.bidirectional_component(source) | self.bidirectional_component(target)
        if relation_type is RelationType.DIRECTED:
            edge = (source, target)
            if edge not in self.directed_edges:
                raise GraphValidationError(
                    f"directed relation does not exist: {source} -> {target}"
                )
            self.directed_edges.remove(edge)
            seeds = self.bidirectional_component(target)
        else:
            edge = self._pair(source, target)
            if edge not in self.bidirectional_edges:
                raise GraphValidationError(
                    f"bidirectional relation does not exist: {edge[0]} <-> {edge[1]}"
                )
            self.bidirectional_edges.remove(edge)
            seeds = old_component
        self._touch()
        return MutationResult(
            f"removed {relation_type.value} relation between {source} and {target}",
            self.dirty_closure(seeds),
            change_reason="relation_removed",
        )

    def delete_agent(self, agent_id: str) -> MutationResult:
        self.require_node(agent_id)
        downstream = self.directed_successors(agent_id)
        old_component = self.bidirectional_component(agent_id)
        self.directed_edges = {edge for edge in self.directed_edges if agent_id not in edge}
        self.bidirectional_edges = {
            edge for edge in self.bidirectional_edges if agent_id not in edge
        }
        del self.nodes[agent_id]
        if self.output_agent == agent_id:
            self.output_agent = None
        self._touch()
        remaining_seeds = (downstream | old_component) - {agent_id}
        return MutationResult(
            f"deleted agent {agent_id}",
            self.dirty_closure(remaining_seeds) if remaining_seeds else set(),
            change_reason="agent_deleted",
        )

    def set_output(self, agent_id: str) -> MutationResult:
        self.require_node(agent_id)
        old = self.output_agent
        if old == agent_id:
            return MutationResult(
                f"agent {agent_id} is already the output Agent",
                changed=False,
                change_reason="no_effective_change",
            )
        self.output_agent = agent_id
        self._touch()
        dirty = {agent_id}
        if old:
            dirty.add(old)
        return MutationResult(
            f"set output agent to {agent_id}",
            self.dirty_closure(dirty),
            change_reason="output_changed",
        )

    def require_node(self, agent_id: str) -> AgentNode:
        try:
            return self.nodes[agent_id]
        except KeyError as exc:
            raise GraphValidationError(f"unknown agent: {agent_id}") from exc

    def validate_relations(self) -> None:
        for source, target in self.directed_edges:
            self.require_node(source)
            self.require_node(target)
            if self.nodes[source].layer >= self.nodes[target].layer:
                raise GraphValidationError(
                    f"invalid directed relation {source} -> {target}: layers must increase"
                )
        for source, target in self.bidirectional_edges:
            self.require_node(source)
            self.require_node(target)
            if self.nodes[source].layer != self.nodes[target].layer:
                raise GraphValidationError(
                    f"invalid bidirectional relation {source} <-> {target}: layers must match"
                )

    def validate(self, *, final: bool = False) -> list[str]:
        errors: list[str] = []
        if len(self.nodes) > self.max_agents:
            errors.append(f"agent budget exceeded ({len(self.nodes)}/{self.max_agents})")
        try:
            self.validate_relations()
        except GraphValidationError as exc:
            errors.append(str(exc))
        for node in self.nodes.values():
            route = str(node.metadata.get("runtime_route", ""))
            if self.runtime_routes and route and route not in self.runtime_routes:
                errors.append(f"unknown Worker runtime_route: {route}")
        if final:
            if not self.nodes:
                errors.append("graph is empty")
            unconfigured = sorted(key for key, node in self.nodes.items() if not node.configured)
            if unconfigured:
                errors.append(f"agents missing prompts or models: {', '.join(unconfigured)}")
            if self.output_agent is None:
                errors.append("output agent is not set")
            elif self.output_agent in self.nodes:
                unreachable = sorted(
                    agent_id
                    for agent_id in self.nodes
                    if agent_id != self.output_agent
                    and self.output_agent not in self.reachable_from(agent_id)
                )
                if unreachable:
                    errors.append("agents cannot influence output: " + ", ".join(unreachable))
        return errors

    def assert_valid(self, *, final: bool = False) -> None:
        errors = self.validate(final=final)
        if errors:
            raise GraphValidationError("; ".join(errors))

    def directed_predecessors(self, agent_id: str) -> set[str]:
        return {source for source, target in self.directed_edges if target == agent_id}

    def directed_successors(self, agent_id: str) -> set[str]:
        return {target for source, target in self.directed_edges if source == agent_id}

    def bidirectional_neighbors(self, agent_id: str) -> set[str]:
        neighbors: set[str] = set()
        for source, target in self.bidirectional_edges:
            if source == agent_id:
                neighbors.add(target)
            elif target == agent_id:
                neighbors.add(source)
        return neighbors

    def bidirectional_component(self, agent_id: str) -> set[str]:
        if agent_id not in self.nodes:
            return set()
        component = {agent_id}
        queue = deque([agent_id])
        while queue:
            current = queue.popleft()
            for neighbor in self.bidirectional_neighbors(current):
                if neighbor not in component:
                    component.add(neighbor)
                    queue.append(neighbor)
        return component

    def components_in_layer(self, layer: int) -> list[list[str]]:
        remaining = {key for key, node in self.nodes.items() if node.layer == layer}
        components: list[list[str]] = []
        while remaining:
            seed = min(remaining)
            component = self.bidirectional_component(seed) & remaining
            components.append(sorted(component))
            remaining -= component
        return sorted(components, key=lambda item: item[0])

    def reachable_from(self, agent_id: str) -> set[str]:
        seen: set[str] = set()
        queue = deque([agent_id])
        while queue:
            current = queue.popleft()
            neighbors = self.directed_successors(current) | self.bidirectional_neighbors(current)
            for neighbor in neighbors:
                if neighbor not in seen and neighbor != agent_id:
                    seen.add(neighbor)
                    queue.append(neighbor)
        return seen

    def dirty_closure(self, seeds: Iterable[str]) -> set[str]:
        expanded: set[str] = set()
        queue = deque(agent_id for agent_id in seeds if agent_id in self.nodes)
        while queue:
            current = queue.popleft()
            for peer in self.bidirectional_component(current):
                if peer not in expanded:
                    expanded.add(peer)
                    queue.append(peer)
            for child in self.directed_successors(current):
                if child not in expanded:
                    expanded.add(child)
                    queue.append(child)
        return expanded

    def layers(self) -> list[int]:
        return sorted({node.layer for node in self.nodes.values()})

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "max_agents": self.max_agents,
            "runtime_routes": list(self.runtime_routes),
            "action_protocol": ACTION_PROTOCOL,
            "submission_protocol": PROTOCOL,
            "nodes": [self.nodes[key].to_dict() for key in sorted(self.nodes)],
            "relations": [
                Relation(source, target, RelationType.DIRECTED).to_dict()
                for source, target in sorted(self.directed_edges)
            ]
            + [
                Relation(source, target, RelationType.BIDIRECTIONAL).to_dict()
                for source, target in sorted(self.bidirectional_edges)
            ],
            "output_agent": self.output_agent,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> MultiAgentGraph:
        graph = cls(
            max_agents=int(payload.get("max_agents", 8)),
            runtime_routes=tuple(payload.get("runtime_routes", ())),
        )
        for raw_node in payload.get("nodes", []):
            agent_id = str(raw_node["agent_id"])
            graph.add_agent(agent_id)
            prompt = str(raw_node.get("prompt", "")).strip()
            if prompt:
                graph.set_prompt(agent_id, prompt)
            graph.set_layer(agent_id, int(raw_node.get("layer", 0)))
            node = graph.nodes[agent_id]
            node.allowed_tools = tuple(raw_node.get("allowed_tools", []))
            node.operation_policy_configured = bool(
                raw_node.get("operation_policy_configured", False)
            )
            node.initial_tool_budget = int(raw_node.get("initial_tool_budget", 0))
            node.revision_tool_budget = int(raw_node.get("revision_tool_budget", 0))
            node.total_tool_budget = int(raw_node.get("total_tool_budget", 0))
            graph.nodes[agent_id].metadata = dict(raw_node.get("metadata", {}))
            if graph.runtime_routes:
                graph.nodes[agent_id].metadata["model_selection_required"] = True
        for raw_relation in payload.get("relations", []):
            graph.set_relation(
                str(raw_relation["source"]),
                str(raw_relation["target"]),
                str(raw_relation["relation"]),
            )
        output = payload.get("output_agent")
        if output is not None:
            graph.set_output(str(output))
        graph.assert_valid(final=False)
        return graph

    def describe(self) -> str:
        if not self.nodes:
            return "Graph is empty."
        lines: list[str] = []
        for layer in self.layers():
            agents = sorted(key for key, node in self.nodes.items() if node.layer == layer)
            described = [
                f"{agent_id}["
                f"{'configured' if self.nodes[agent_id].configured else 'awaiting_model' if self.nodes[agent_id].prompt_configured else 'awaiting_prompt'};"
                f"route={self.nodes[agent_id].metadata.get('runtime_route', 'unassigned')}]"
                for agent_id in agents
            ]
            lines.append(f"Layer {layer}: {', '.join(described)}")
            for component in self.components_in_layer(layer):
                if len(component) > 1:
                    lines.append("  bidirectional: " + " <-> ".join(component))
        for source, target in sorted(self.directed_edges):
            lines.append(f"  {source} -> {target}")
        lines.append(f"Submission target: {self.output_agent or 'chosen only by FINISH(target)'}")
        return "\n".join(lines)

    def _allocate_agent_id(self) -> str:
        while f"agent_{self._next_agent_index}" in self.nodes:
            self._next_agent_index += 1
        return f"agent_{self._next_agent_index}"

    def _advance_index_from_id(self, agent_id: str) -> None:
        prefix = "agent_"
        if agent_id.startswith(prefix) and agent_id[len(prefix) :].isdigit():
            self._next_agent_index = max(
                self._next_agent_index,
                int(agent_id[len(prefix) :]) + 1,
            )

    def _touch(self) -> None:
        self.version += 1

    @staticmethod
    def _pair(source: str, target: str) -> tuple[str, str]:
        return tuple(sorted((source, target)))  # type: ignore[return-value]


