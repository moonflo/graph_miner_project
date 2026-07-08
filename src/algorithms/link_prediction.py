"""Candidate-limited classical link prediction algorithms."""

from __future__ import annotations

import math
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

CUSTOM_METHODS = (
    "common_neighbors",
    "weighted_common_neighbors",
    "weighted_resource_allocation",
    "weighted_adamic_adar",
    "time_decay_common_neighbors",
    "time_decay_resource_allocation",
)

SUPPORTED_LINK_PREDICTION_METHODS = tuple(METHOD_TO_NETWORKX_FUNCTION) + CUSTOM_METHODS


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
    *,
    decay: float = 0.9,
) -> list[LinkPredictionScore]:
    """Score candidate pairs and return them sorted by descending score."""

    scores = score_candidate_pairs_in_order(graph, candidate_pairs, method, decay=decay)
    return sorted(scores, key=lambda item: (-item.score, repr(item.source), repr(item.target)))


def score_candidate_pairs_in_order(
    graph: nx.Graph,
    candidate_pairs: Iterable[tuple[Hashable, Hashable]],
    method: str,
    *,
    decay: float = 0.9,
    topology_graph: nx.Graph | None = None,
) -> list[LinkPredictionScore]:
    """Score candidate pairs without enumerating graph-wide non-edges.

    The result preserves the input order, which is required for citation2 MRR
    matrices. Public ranking callers should use :func:`score_candidate_pairs`.
    """

    normalized_method = normalize_method(method)
    pairs = _normalize_candidate_pairs(candidate_pairs)
    if not pairs:
        return []

    scoring_graph = topology_graph
    if scoring_graph is None:
        scoring_graph = to_simple_undirected_for_topology(graph, include_isolated_nodes=False)
    scoring_graph.add_nodes_from(_candidate_nodes(pairs))

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
        if normalized_method in METHOD_TO_NETWORKX_FUNCTION:
            scorer = getattr(nx, METHOD_TO_NETWORKX_FUNCTION[normalized_method])
            score_iter = scorer(scoring_graph, ebunch=valid_pairs)
        else:
            score_iter = _score_custom_method(scoring_graph, valid_pairs, normalized_method, decay)

        for position, (source, target, score) in zip(valid_positions, score_iter, strict=True):
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
    if normalized in {"cn", "common_neighbor", "common_neighbours", "common_neighbors"}:
        normalized = "common_neighbors"
    elif normalized in {"aa", "adamic_adar_index"}:
        normalized = "adamic_adar"
    elif normalized in {"ra", "resource_allocation_index"}:
        normalized = "resource_allocation"
    elif normalized in {"pa", "preferential_attachment_index"}:
        normalized = "preferential_attachment"
    elif normalized in {"jaccard_coefficient", "jaccard_index"}:
        normalized = "jaccard"

    if normalized not in SUPPORTED_LINK_PREDICTION_METHODS:
        supported = ", ".join(sorted(SUPPORTED_LINK_PREDICTION_METHODS))
        raise ValueError(f"Unsupported link prediction method '{method}'. Supported: {supported}")
    return normalized


def _score_custom_method(
    graph: nx.Graph,
    candidate_pairs: Iterable[tuple[Hashable, Hashable]],
    method: str,
    decay: float,
) -> Iterable[tuple[Hashable, Hashable, float]]:
    """Score explicit pairs with project-local weighted/time-aware heuristics."""

    if method.startswith("time_decay_"):
        _validate_decay(decay)

    for source, target in candidate_pairs:
        if source not in graph or target not in graph:
            yield source, target, 0.0
            continue

        common = sorted(nx.common_neighbors(graph, source, target), key=repr)
        if method == "common_neighbors":
            score = float(len(common))
        elif method == "weighted_common_neighbors":
            score = _weighted_common_neighbors_score(graph, source, target, common)
        elif method == "weighted_resource_allocation":
            score = _weighted_resource_allocation_score(graph, source, target, common)
        elif method == "weighted_adamic_adar":
            score = _weighted_adamic_adar_score(graph, source, target, common)
        elif method == "time_decay_common_neighbors":
            score = _time_decay_common_neighbors_score(graph, source, target, common, decay)
        elif method == "time_decay_resource_allocation":
            score = _time_decay_resource_allocation_score(graph, source, target, common, decay)
        else:
            raise ValueError(f"Unsupported custom link prediction method: {method}")
        yield source, target, score


def _weighted_common_neighbors_score(
    graph: nx.Graph,
    source: Hashable,
    target: Hashable,
    common_neighbors: list[Hashable],
) -> float:
    """Sum the average incident edge weight for each common neighbor."""

    return sum(
        _average_edge_weight(graph, source, neighbor, target, neighbor)
        for neighbor in common_neighbors
    )


def _weighted_resource_allocation_score(
    graph: nx.Graph,
    source: Hashable,
    target: Hashable,
    common_neighbors: list[Hashable],
) -> float:
    total = 0.0
    for neighbor in common_neighbors:
        degree_weight = _weighted_degree(graph, neighbor)
        if degree_weight <= 0:
            continue
        total += _average_edge_weight(graph, source, neighbor, target, neighbor) / degree_weight
    return total


def _weighted_adamic_adar_score(
    graph: nx.Graph,
    source: Hashable,
    target: Hashable,
    common_neighbors: list[Hashable],
) -> float:
    total = 0.0
    for neighbor in common_neighbors:
        denominator = math.log1p(_weighted_degree(graph, neighbor))
        if denominator <= 0:
            continue
        total += _average_edge_weight(graph, source, neighbor, target, neighbor) / denominator
    return total


def _time_decay_common_neighbors_score(
    graph: nx.Graph,
    source: Hashable,
    target: Hashable,
    common_neighbors: list[Hashable],
    decay: float,
) -> float:
    """Sum average incident weights after optional recency decay.

    When ``max_train_year`` or edge ``max_year`` is missing, this intentionally
    falls back to the weighted common-neighbor score.
    """

    if graph.graph.get("max_train_year") is None:
        return _weighted_common_neighbors_score(graph, source, target, common_neighbors)

    return sum(
        _average_time_decay_weight(graph, source, neighbor, target, neighbor, decay)
        for neighbor in common_neighbors
    )


def _time_decay_resource_allocation_score(
    graph: nx.Graph,
    source: Hashable,
    target: Hashable,
    common_neighbors: list[Hashable],
    decay: float,
) -> float:
    if graph.graph.get("max_train_year") is None:
        return _weighted_resource_allocation_score(graph, source, target, common_neighbors)

    total = 0.0
    for neighbor in common_neighbors:
        degree_weight = _time_decay_weighted_degree(graph, neighbor, decay)
        if degree_weight <= 0:
            continue
        total += _average_time_decay_weight(
            graph,
            source,
            neighbor,
            target,
            neighbor,
            decay,
        ) / degree_weight
    return total


def _merge_simple_edge(
    graph: nx.Graph,
    source: Hashable,
    target: Hashable,
    attrs: dict[str, Any],
) -> None:
    weight = _numeric_weight(attrs.get("weight", 1.0))
    if graph.has_edge(source, target):
        edge_attrs = graph[source][target]
        existing_edge_count = int(
            edge_attrs.get("edge_count", edge_attrs.get("merged_edge_count", 1))
        )
        edge_attrs["weight"] = edge_attrs.get("weight", 1.0) + weight
        edge_attrs["merged_edge_count"] = edge_attrs.get("merged_edge_count", 1) + 1
        edge_attrs["edge_count"] = existing_edge_count + int(attrs.get("edge_count", 1))
        edge_attrs["min_year"] = _merge_min_year(
            edge_attrs.get("min_year"),
            attrs.get("min_year"),
        )
        edge_attrs["max_year"] = _merge_max_year(
            edge_attrs.get("max_year"),
            attrs.get("max_year"),
        )
        return

    merged_attrs = dict(attrs)
    merged_attrs["weight"] = weight
    merged_attrs["merged_edge_count"] = 1
    merged_attrs["edge_count"] = int(attrs.get("edge_count", 1))
    graph.add_edge(source, target, **merged_attrs)


def _copy_node_attrs(graph: nx.Graph, source_graph: nx.Graph, node: Hashable) -> None:
    if node not in graph:
        graph.add_node(node, **source_graph.nodes[node])


def _numeric_weight(value: Any) -> float:
    if isinstance(value, Real):
        return float(value)
    return 1.0


def _edge_weight(graph: nx.Graph, source: Hashable, target: Hashable) -> float:
    if not graph.has_edge(source, target):
        return 1.0
    return _numeric_weight(graph[source][target].get("weight", 1.0))


def _average_edge_weight(
    graph: nx.Graph,
    source_left: Hashable,
    neighbor_left: Hashable,
    source_right: Hashable,
    neighbor_right: Hashable,
) -> float:
    return (
        _edge_weight(graph, source_left, neighbor_left)
        + _edge_weight(graph, source_right, neighbor_right)
    ) / 2.0


def _weighted_degree(graph: nx.Graph, node: Hashable) -> float:
    return sum(
        _numeric_weight(attrs.get("weight", 1.0))
        for _, _, attrs in graph.edges(node, data=True)
    )


def _time_decay_edge_weight(
    graph: nx.Graph,
    source: Hashable,
    target: Hashable,
    decay: float,
) -> float:
    weight = _edge_weight(graph, source, target)
    max_train_year = graph.graph.get("max_train_year")
    if max_train_year is None or not graph.has_edge(source, target):
        return weight

    edge_year = graph[source][target].get("max_year")
    if edge_year is None:
        return weight

    try:
        age = max(0, int(max_train_year) - int(edge_year))
    except (TypeError, ValueError):
        return weight
    return weight * (decay**age)


def _average_time_decay_weight(
    graph: nx.Graph,
    source_left: Hashable,
    neighbor_left: Hashable,
    source_right: Hashable,
    neighbor_right: Hashable,
    decay: float,
) -> float:
    return (
        _time_decay_edge_weight(graph, source_left, neighbor_left, decay)
        + _time_decay_edge_weight(graph, source_right, neighbor_right, decay)
    ) / 2.0


def _time_decay_weighted_degree(graph: nx.Graph, node: Hashable, decay: float) -> float:
    return sum(
        _time_decay_edge_weight(graph, node, neighbor, decay)
        for neighbor in graph.neighbors(node)
    )


def _validate_decay(decay: float) -> None:
    if not 0 < decay <= 1:
        raise ValueError("decay must be in the range (0, 1]")


def _merge_min_year(existing: Any, new_year: Any) -> int | None:
    existing_year = _optional_year(existing)
    incoming_year = _optional_year(new_year)
    if existing_year is None:
        return incoming_year
    if incoming_year is None:
        return existing_year
    return min(existing_year, incoming_year)


def _merge_max_year(existing: Any, new_year: Any) -> int | None:
    existing_year = _optional_year(existing)
    incoming_year = _optional_year(new_year)
    if existing_year is None:
        return incoming_year
    if incoming_year is None:
        return existing_year
    return max(existing_year, incoming_year)


def _optional_year(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


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
