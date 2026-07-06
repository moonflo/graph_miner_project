"""Graph algorithms for latent relation mining."""

from __future__ import annotations

from typing import Iterable

import networkx as nx


def adamic_adar_predictions(graph: nx.Graph, top_n: int = 10) -> list[tuple[str, str, float]]:
    """Rank non-edge pairs using the Adamic-Adar index."""

    scores = nx.adamic_adar_index(graph)
    ranked = sorted(scores, key=lambda item: item[2], reverse=True)
    return [(str(u), str(v), float(score)) for u, v, score in ranked[:top_n]]


def jaccard_predictions(graph: nx.Graph, top_n: int = 10) -> list[tuple[str, str, float]]:
    """Rank non-edge pairs using the Jaccard coefficient."""

    scores = nx.jaccard_coefficient(graph)
    ranked = sorted(scores, key=lambda item: item[2], reverse=True)
    return [(str(u), str(v), float(score)) for u, v, score in ranked[:top_n]]


def resource_allocation_predictions(graph: nx.Graph, top_n: int = 10) -> list[tuple[str, str, float]]:
    """Rank non-edge pairs using the resource allocation index."""

    scores = nx.resource_allocation_index(graph)
    ranked = sorted(scores, key=lambda item: item[2], reverse=True)
    return [(str(u), str(v), float(score)) for u, v, score in ranked[:top_n]]


def detect_louvain_communities(graph: nx.Graph) -> list[set[str]]:
    """Detect communities with Louvain, falling back to greedy modularity if needed."""

    if graph.number_of_edges() == 0:
        return [{str(node)} for node in graph.nodes]

    community_module = nx.algorithms.community
    if hasattr(community_module, "louvain_communities"):
        communities = community_module.louvain_communities(graph, weight="weight", seed=42)
    else:
        communities = community_module.greedy_modularity_communities(graph, weight="weight")

    return [set(map(str, community)) for community in communities]


def shortest_path_chain(graph: nx.Graph, source: str, target: str) -> list[str] | None:
    """Return the shortest relationship chain between two nodes, if one exists."""

    try:
        return list(map(str, nx.shortest_path(graph, source=source, target=target)))
    except nx.NetworkXNoPath:
        return None


def edges_from_predictions(predictions: Iterable[tuple[str, str, float]]) -> list[tuple[str, str]]:
    """Drop scores from prediction tuples for evaluation helpers."""

    return [(source, target) for source, target, _ in predictions]
