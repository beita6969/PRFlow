from __future__ import annotations

import itertools
import json

from .graph import MultiAgentGraph


def canonical_graph_key(graph: MultiAgentGraph, *, include_prompts: bool = False) -> str:
    """Exact permutation-invariant key for the small (<=8 node) Canvas graphs."""

    graph.assert_valid(final=False)
    groups: list[list[str]] = []
    for layer in graph.layers():
        layer_nodes = [key for key, node in graph.nodes.items() if node.layer == layer]
        if include_prompts:
            by_prompt: dict[tuple[object, ...], list[str]] = {}
            for key in layer_nodes:
                node = graph.nodes[key]
                signature = (
                    " ".join(node.prompt.split()),
                    str(node.metadata.get("runtime_route", "")),
                )
                by_prompt.setdefault(signature, []).append(key)
            groups.extend(by_prompt[key] for key in sorted(by_prompt))
        else:
            groups.append(layer_nodes)
    candidates: list[str] = []
    for permutations in itertools.product(*(itertools.permutations(group) for group in groups)):
        order = [node for group in permutations for node in group]
        index = {node: position for position, node in enumerate(order)}
        nodes = []
        for node_id in order:
            node = graph.nodes[node_id]
            item: list[object] = [node.layer, node_id == graph.output_agent]
            if include_prompts:
                item.append(" ".join(node.prompt.split()))
                item.append(str(node.metadata.get("runtime_route", "")))
            nodes.append(item)
        directed = sorted((index[source], index[target]) for source, target in graph.directed_edges)
        bidirectional = sorted(
            tuple(sorted((index[source], index[target])))
            for source, target in graph.bidirectional_edges
        )
        candidates.append(
            json.dumps([nodes, directed, bidirectional], ensure_ascii=False, separators=(",", ":"))
        )
    return min(candidates, default="[[],[],[]]")
