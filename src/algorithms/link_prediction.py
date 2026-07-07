"""Candidate-limited classical link prediction algorithms."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from numbers import Real
from typing import Any, Hashable

import networkx as nx


METHOD_TO_NETWORKX_FUNCTION = {
    "jaccard": "jaccard_coefficient",
    "adamic_adar": "adamic_adar_index",
    "resource_allocation": "resource_allocation_index",
    "preferential_attachment": "preferential_attachment",
}


@dataclass(frozen=True)
class LinkPredictionScore:
    """One scored candidate edge."""

    source: Hashable
    target: Hashable
    score: float
    method: str


def score_candidate_pairs(
    graph: nx.Graph,
    candidate_pairs: Iterable[tuple[Hashable, Hashable]],
    method: str,
) -> list[LinkPredictionScore]:
    """Score candidate pairs and return them sorted by descending score."""

    scores = score_candidate_pairs_in_order(graph, candidate_pairs, method)
    return sorted(scores, key=lambda item: (-item.score, repr(item.source), repr(item.target)))


def score_candidate_pairs_in_order(
    graph: nx.Graph,
    candidate_pairs: Iterable[tuple[Hashable, Hashable]],
    method: str,
) -> list[LinkPredictionScore]:
    """Score candidate pairs without enumerating graph-wide non-edges.

    The result preserves the input order, which is required for citation2 MRR
    matrices. Public ranking callers should use :func:`score_candidate_pairs`.
    """

    normalized_method = normalize_method(method)
    pairs = _normalize_candidate_pairs(candidate_pairs)
    if not pairs:
        return []

    topology_graph = to_simple_undirected_for_topology(graph, include_isolated_nodes=False)
    topology_graph.add_nodes_from(_candidate_nodes(pairs))

    valid_pairs: list[tuple[Hashable, Hashable]] = []
    valid_positions: list[int] = []
    ordered_scores: list[LinkPredictionScore | None] = [None] * len(pairs)

    for index, pair in enumerate(pairs):
        source, target = pair
        if source == target:
            ordered_scores[index] = LinkPredictionScore(source, target, 0.0, normalized_method)
            continue
        valid_positions.append(index)
        valid_pairs.append(pair)

    if valid_pairs:
        scorer = getattr(nx, METHOD_TO_NETWORKX_FUNCTION[normalized_method])
        for position, (source, target, score) in zip(
            valid_positions,
            scorer(topology_graph, ebunch=valid_pairs),
            strict=True,
        ):
            ordered_scores[position] = LinkPredictionScore(
                source,
                target,
                float(score),
                normalized_method,
            )

    return [
        score
        if score is not None
        else LinkPredictionScore(source, target, 0.0, normalized_method)
        for score, (source, target) in zip(ordered_scores, pairs, strict=True)
    ]


def to_simple_undirected_for_topology(
    graph: nx.Graph,
    *,
    include_isolated_nodes: bool = True,
) -> nx.Graph:
    """Project a NetworkX graph to a simple undirected topology graph.

    NetworkX classical link prediction APIs do not accept directed graphs or
    multigraphs. This projection keeps nodes, merges parallel/reversed edges by
    summing numeric weights, and removes self-loops before scoring.
    """

    topology_graph = nx.Graph()
    topology_graph.graph.update(graph.graph)
    topology_graph.graph.update(
        {
            "original_directed": graph.is_directed(),
            "original_multigraph": graph.is_multigraph(),
            "topology_projection": "simple_undirected",
            "self_loops_removed": nx.number_of_selfloops(graph),
        }
    )
    if graph.is_directed():
        topology_graph.graph["projection_note"] = (
            "Directed graph projected to an undirected simple graph for "
            "NetworkX classical topology scores."
        )
    if topology_graph.graph.get("dataset_name") == "ogbl_citation2":
        topology_graph.graph["citation2_topology_note"] = (
            "ogbl_citation2 is a directed task; these scores are an "
            "undirected-projection classical baseline, not a directed model."
        )

    if include_isolated_nodes:
        topology_graph.add_nodes_from(graph.nodes(data=True))
    for source, target, attrs in graph.edges(data=True):
        if source == target:
            continue
        if not include_isolated_nodes:
            _copy_node_attrs(topology_graph, graph, source)
            _copy_node_attrs(topology_graph, graph, target)
        _merge_simple_edge(topology_graph, source, target, attrs)

    topology_graph.remove_edges_from(nx.selfloop_edges(topology_graph))
    return topology_graph


def normalize_method(method: str) -> str:
    normalized = method.strip().lower().replace("-", "_")
    if normalized in {"aa", "adamic_adar_index"}:
        normalized = "adamic_adar"
    elif normalized in {"ra", "resource_allocation_index"}:
        normalized = "resource_allocation"
    elif normalized in {"pa", "preferential_attachment_index"}:
        normalized = "preferential_attachment"
    elif normalized in {"jaccard_coefficient", "jaccard_index"}:
        normalized = "jaccard"

    if normalized not in METHOD_TO_NETWORKX_FUNCTION:
        supported = ", ".join(sorted(METHOD_TO_NETWORKX_FUNCTION))
        raise ValueError(f"Unsupported link prediction method '{method}'. Supported: {supported}")
    return normalized


def _merge_simple_edge(
    graph: nx.Graph,
    source: Hashable,
    target: Hashable,
    attrs: dict[str, Any],
) -> None:
    weight = _numeric_weight(attrs.get("weight", 1.0))
    if graph.has_edge(source, target):
        graph[source][target]["weight"] = graph[source][target].get("weight", 1.0) + weight
        graph[source][target]["merged_edge_count"] = (
            graph[source][target].get("merged_edge_count", 1) + 1
        )
        return

    merged_attrs = dict(attrs)
    merged_attrs["weight"] = weight
    merged_attrs["merged_edge_count"] = 1
    graph.add_edge(source, target, **merged_attrs)


def _copy_node_attrs(graph: nx.Graph, source_graph: nx.Graph, node: Hashable) -> None:
    if node not in graph:
        graph.add_node(node, **source_graph.nodes[node])


def _numeric_weight(value: Any) -> float:
    if isinstance(value, Real):
        return float(value)
    return 1.0


def _normalize_candidate_pairs(
    candidate_pairs: Iterable[tuple[Hashable, Hashable]],
) -> list[tuple[Hashable, Hashable]]:
    pairs: list[tuple[Hashable, Hashable]] = []
    for pair in candidate_pairs:
        if len(pair) != 2:
            raise ValueError(f"Candidate pair must contain exactly two nodes: {pair!r}")
        source, target = pair
        pairs.append((source, target))
    return pairs


def _candidate_nodes(
    candidate_pairs: Iterable[tuple[Hashable, Hashable]],
) -> Iterable[Hashable]:
    for source, target in candidate_pairs:
        yield source
        yield target
