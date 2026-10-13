from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .graph import MultiAgentGraph

STRUCTURE_ONLY_SCHEMA = "structure_only"
SEMANTIC_FEATURE_VERSION = "structure_plus_semantics_layered_equal_weight"
DELEGATION_TEMPLATE_VERSION = "delegation_query"


class DelegationEmbedder(Protocol):
    encoder_id: str

    def encode(self, texts: list[str]) -> list[tuple[float, ...]]: ...


@dataclass(frozen=True)
class GraphFeatureSchema:
    version: str
    encoder_id: str
    encoder_hash: str
    delegation_template_version: str
    max_agents: int
    structure_scaler_version: str
    pooling: str
    block_weights: tuple[float, ...]
    output_encoding_mode: str
    feature_dimension: int
    runtime_routes: tuple[str, ...] = ()

    @property
    def schema_id(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "encoder_id": self.encoder_id,
            "encoder_hash": self.encoder_hash,
            "delegation_template_version": self.delegation_template_version,
            "max_agents": self.max_agents,
            "structure_scaler_version": self.structure_scaler_version,
            "pooling": self.pooling,
            "block_weights": list(self.block_weights),
            "output_encoding_mode": self.output_encoding_mode,
            "feature_dimension": self.feature_dimension,
            "runtime_routes": list(self.runtime_routes),
        }


@dataclass(frozen=True)
class GraphFeatures:
    values: tuple[float, ...]
    names: tuple[str, ...]
    schema_id: str = STRUCTURE_ONLY_SCHEMA
    schema: GraphFeatureSchema | None = None

    def to_dict(self) -> dict[str, float]:
        return dict(zip(self.names, self.values, strict=True))


class E5DelegationEncoder:
    """Frozen CPU E5 encoder isolated from SkillBank cache semantics."""

    def __init__(self, model_path: str | Path) -> None:
        from .skills import E5SkillEmbedder

        shared = E5SkillEmbedder(model_path)
        self.torch = shared.torch
        self.tokenizer = shared.tokenizer
        self.model = shared.model.eval().cpu()
        self.encoder_id = str(model_path)

    def encode(self, texts: list[str]) -> list[tuple[float, ...]]:
        vectors: list[tuple[float, ...]] = []
        with self.torch.no_grad():
            for offset in range(0, len(texts), 32):
                batch = ["query: " + text for text in texts[offset : offset + 32]]
                inputs = self.tokenizer(
                    batch, padding=True, truncation=True, max_length=512, return_tensors="pt"
                )
                output = self.model(**inputs)
                mask = inputs["attention_mask"].unsqueeze(-1).float()
                pooled = (output.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
                pooled = self.torch.nn.functional.normalize(pooled, p=2, dim=1)
                vectors.extend(tuple(float(value) for value in row) for row in pooled)
        return vectors


class SemanticGraphFeatureExtractor:
    """Frozen four-block graph representation."""

    def __init__(self, embedder: DelegationEmbedder) -> None:
        self.embedder = embedder
        self._cache: dict[tuple[str, str, str], tuple[float, ...]] = {}
        self.encoder_hash = hashlib.sha256(embedder.encoder_id.encode("utf-8")).hexdigest()
        self._embedding_dimension: int | None = None

    def extract_many(self, graphs: Sequence[MultiAgentGraph]) -> list[GraphFeatures]:
        graphs = _shared_route_feature_graphs(graphs)
        texts = {_delegation_text(node) for graph in graphs for node in graph.nodes.values()}
        missing = [text for text in sorted(texts) if self._cache_key(text) not in self._cache]
        if missing:
            vectors = self.embedder.encode(missing)
            if len(vectors) != len(missing):
                raise ValueError("delegation encoder returned the wrong number of embeddings")
            for text, vector in zip(missing, vectors, strict=True):
                if not vector or not all(math.isfinite(value) for value in vector):
                    raise ValueError("delegation embedding must be finite and non-empty")
                if self._embedding_dimension not in (None, len(vector)):
                    raise ValueError("delegation encoder changed embedding dimension")
                self._embedding_dimension = len(vector)
                self._cache[self._cache_key(text)] = tuple(float(value) for value in vector)
        if graphs and self._embedding_dimension is None:
            # Empty policy graphs are valid terminal failures. Probe only the
            # encoder shape; this vector is never used as a fabricated node.
            probe = self.embedder.encode([""])
            if len(probe) != 1 or not probe[0] or not all(math.isfinite(v) for v in probe[0]):
                raise ValueError("invalid empty-graph encoder shape")
            self._embedding_dimension = len(probe[0])
        return [self._extract(graph) for graph in graphs]

    def _cache_key(self, text: str) -> tuple[str, str, str]:
        return (
            self.encoder_hash,
            DELEGATION_TEMPLATE_VERSION,
            hashlib.sha256(text.encode("utf-8")).hexdigest(),
        )

    def _extract(self, graph: MultiAgentGraph) -> GraphFeatures:
        embeddings = {
            agent_id: self._cache[self._cache_key(_delegation_text(node))]
            for agent_id, node in graph.nodes.items()
        }
        routes = tuple(sorted(graph.runtime_routes))
        if routes:
            # Tensor-product binding preserves which responsibility uses which route,
            # unlike concatenating independent mean prompt/model histograms.
            embeddings = {
                agent_id: tuple(vector)
                + tuple(
                    value if graph.nodes[agent_id].metadata.get("runtime_route") == route else 0.0
                    for route in routes
                    for value in (1.0, *vector)
                )
                for agent_id, vector in embeddings.items()
            }
        dimensions = {len(vector) for vector in embeddings.values()}
        if len(dimensions) > 1:
            raise ValueError("delegation embeddings have inconsistent dimensions")
        base_dimension = int(self._embedding_dimension or 0)
        dimension = next(iter(dimensions), base_dimension + len(routes) * (base_dimension + 1))
        max_agents = int(graph.max_agents)
        ranked_layers = {
            layer: rank
            for rank, layer in enumerate(sorted({node.layer for node in graph.nodes.values()}))
        }
        structure = _normalized_structure_values(graph)
        node_block: list[float] = []
        for rank in range(max_agents):
            members = [
                agent_id
                for agent_id, node in graph.nodes.items()
                if ranked_layers[node.layer] == rank
            ]
            pooled = _vector_mean([embeddings[agent_id] for agent_id in members], dimension)
            output_rate = (
                sum(agent_id == graph.output_agent for agent_id in members) / len(members)
                if members
                else 0.0
            )
            node_block.extend((*pooled, output_rate))
        directed_block = _vector_mean(
            [(*embeddings[source], *embeddings[target]) for source, target in graph.directed_edges],
            dimension * 2,
        )
        bidirectional_vectors = []
        for source, target in graph.bidirectional_edges:
            left, right = embeddings[source], embeddings[target]
            bidirectional_vectors.append(
                tuple(a + b for a, b in zip(left, right, strict=True))
                + tuple(abs(a - b) for a, b in zip(left, right, strict=True))
            )
        bidirectional_block = _vector_mean(bidirectional_vectors, dimension * 2)
        blocks = [structure, tuple(node_block), directed_block, bidirectional_block]
        weighted = [value * 0.5 for block in blocks for value in _l2_normalize(block)]
        values = _l2_normalize(weighted)
        schema = GraphFeatureSchema(
            version=(
                "structure_semantics_model_binding" if routes else SEMANTIC_FEATURE_VERSION
            ),
            encoder_id=self.embedder.encoder_id,
            encoder_hash=self.encoder_hash,
            delegation_template_version=DELEGATION_TEMPLATE_VERSION,
            max_agents=max_agents,
            structure_scaler_version="hard_budget_scaler",
            pooling="layer_mean_directed_endpoint_mean_bidirectional_symmetric_mean",
            block_weights=(0.25, 0.25, 0.25, 0.25),
            output_encoding_mode="node_block_is_output",
            feature_dimension=len(values),
            runtime_routes=routes,
        )
        names = tuple(f"semantic_feature_{index}" for index in range(len(values)))
        return GraphFeatures(values, names, schema.schema_id, schema)


def _shared_route_feature_graphs(graphs: Sequence[MultiAgentGraph]) -> list[MultiAgentGraph]:
    """Align feature coordinates after route availability changes, never policy graphs.

    Actual node bindings and graph budgets remain unchanged. Only the encoder's
    route vocabulary is unioned; unavailable routes get zero feature entries.
    """
    routes = sorted({route for graph in graphs for route in graph.runtime_routes})
    if all(sorted(graph.runtime_routes) == routes for graph in graphs):
        return list(graphs)
    return [
        MultiAgentGraph.from_dict({**graph.to_dict(), "runtime_routes": routes}) for graph in graphs
    ]


def execution_policy_features_many(graphs: Sequence[MultiAgentGraph]) -> list[GraphFeatures]:
    return [execution_policy_features(graph) for graph in _shared_route_feature_graphs(graphs)]


def execution_policy_features(graph: MultiAgentGraph) -> GraphFeatures:
    """Versioned fixed-pool model/responsibility representation when E5 is not configured."""
    base = structural_features(graph)
    routes = tuple(sorted(graph.runtime_routes))
    if not routes:
        return base
    values = list(base.values)
    # Stable signed role/model hash, pooled by layer. Never Python's randomized hash().
    binding = [0.0] * (graph.max_agents * 128)
    ranked = {layer: index for index, layer in enumerate(graph.layers())}
    for node in graph.nodes.values():
        route = str(node.metadata.get("runtime_route", ""))
        if route not in routes:
            # A partially recorded terminal graph can reach the fallback
            # extractor before the Canvas has a selected model for every
            # responsibility.  Keep feature extraction total here: the
            # structure and other valid route bindings remain useful, while a
            # missing/retired route contributes no fabricated binding.  The
            # execution layer still rejects such a graph when it is actually
            # submitted to a Worker.
            continue
        digest = hashlib.sha256((route + "\\0" + " ".join(node.prompt.split())).encode()).digest()
        index = ranked[node.layer] * 128 + int.from_bytes(digest[:4], "big") % 128
        binding[index] += 1.0 if digest[4] % 2 else -1.0
    values.extend(_l2_normalize(binding))
    schema_id = (
        "structure_role_model_hash:"
        + hashlib.sha256(json.dumps([routes, graph.max_agents]).encode()).hexdigest()
    )
    return GraphFeatures(
        tuple(values),
        base.names + tuple(f"role_model_binding_{i}" for i in range(len(binding))),
        schema_id,
    )


def structural_features(graph: MultiAgentGraph) -> GraphFeatures:
    """Existing structure-only ablation, retained under an explicit schema."""

    names = (
        "agent_count",
        "layer_count",
        "max_layer_width",
        "mean_layer_width",
        "directed_edge_count",
        "bidirectional_edge_count",
        "mean_in_degree",
        "mean_out_degree",
        "max_in_degree",
        "max_out_degree",
        "source_count",
        "sink_count",
        "bidirectional_component_count",
        "mean_bidirectional_component_size",
        "graph_depth",
    )
    return GraphFeatures(_raw_structure_values(graph), names)


def _delegation_text(node: Any) -> str:
    delegation = node.metadata.get("director_delegation", {})
    if not isinstance(delegation, dict):
        delegation = {}
    fields = {
        "role": str(delegation.get("role", "")),
        "objective": str(delegation.get("objective", "")),
        "scope": str(delegation.get("scope", "")),
        "expected_output": str(delegation.get("expected_output", "")),
    }
    if not any(value.strip() for value in fields.values()):
        fields["objective"] = str(node.prompt)
    return "\n".join(f"{key}: {' '.join(value.split())}" for key, value in fields.items())


def _raw_structure_values(graph: MultiAgentGraph) -> tuple[float, ...]:
    layers = graph.layers()
    widths = Counter(node.layer for node in graph.nodes.values())
    in_degrees = [len(graph.directed_predecessors(key)) for key in graph.nodes]
    out_degrees = [len(graph.directed_successors(key)) for key in graph.nodes]
    components = [
        component
        for layer in layers
        for component in graph.components_in_layer(layer)
        if len(component) > 1
    ]
    return (
        float(len(graph.nodes)),
        float(len(layers)),
        float(max(widths.values(), default=0)),
        _mean(list(widths.values())),
        float(len(graph.directed_edges)),
        float(len(graph.bidirectional_edges)),
        _mean(in_degrees),
        _mean(out_degrees),
        float(max(in_degrees, default=0)),
        float(max(out_degrees, default=0)),
        float(sum(degree == 0 for degree in in_degrees)),
        float(sum(degree == 0 for degree in out_degrees)),
        float(len(components)),
        _mean([len(component) for component in components]),
        float(len(layers)),
    )


def _normalized_structure_values(graph: MultiAgentGraph) -> tuple[float, ...]:
    raw = _raw_structure_values(graph)
    n = max(1, int(graph.max_agents))
    edge_max = max(1, n * (n - 1) // 2)
    degree_max = max(1, n - 1)
    scales = (
        n,
        n,
        n,
        n,
        edge_max,
        edge_max,
        degree_max,
        degree_max,
        degree_max,
        degree_max,
        n,
        n,
        n,
        n,
        n,
    )
    return tuple(value / scale for value, scale in zip(raw, scales, strict=True))


def _vector_mean(vectors: Sequence[Sequence[float]], dimension: int) -> tuple[float, ...]:
    if not vectors:
        return (0.0,) * dimension
    if any(len(vector) != dimension for vector in vectors):
        raise ValueError("vector pooling dimension mismatch")
    return tuple(
        sum(vector[index] for vector in vectors) / len(vectors) for index in range(dimension)
    )


def _l2_normalize(values: Sequence[float], *, epsilon: float = 1e-12) -> tuple[float, ...]:
    norm = math.sqrt(sum(float(value) ** 2 for value in values))
    if norm <= epsilon:
        return tuple(0.0 for _ in values)
    return tuple(float(value) / norm for value in values)


def _mean(values: list[int]) -> float:
    return float(sum(values) / len(values)) if values else 0.0
