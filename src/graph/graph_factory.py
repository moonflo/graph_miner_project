"""NetworkX graph factory functions for processed data and OGB splits."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np

from .dataset_registry import get_dataset_config
from .ogb_split_loader import load_ogb_split
from .processed_loader import load_graph_edges, load_graph_nodes, load_stats
from .schemas import GraphEdge, GraphNode, NodeIdMode, OGBSplitData


def infer_directed(dataset_name: str) -> bool:
    """Return the registry-directed default for a dataset."""

    return get_dataset_config(dataset_name).directed


def build_networkx_graph_from_processed(
    dataset_name: str,
    processed_root: str | Path = "data/processed",
    *,
    directed: bool | None = None,
    limit_edges: int | None = None,
    limit_nodes: int | None = None,
    node_id_mode: NodeIdMode = "node_idx",
) -> nx.Graph:
    """Build a NetworkX graph from processed JSONL files.

    Use limit_edges for smoke tests. Building a full processed graph can be very
    memory heavy for OGB-sized data and should be done intentionally.
    """

    config = get_dataset_config(dataset_name)
    graph = _new_graph(config.canonical_name, directed if directed is not None else config.directed)
    graph.graph.update(
        {
            "source": "processed",
            "edge_source": "processed/graph_edges.jsonl",
            "node_id_mode": node_id_mode,
            "processed_graph_is_aggregated": config.processed_graph_is_aggregated,
        }
    )
    if config.processed_graph_is_aggregated:
        graph.graph["warning"] = (
            "Processed graph_edges are aggregated; use official train split for metrics."
        )

    try:
        stats = load_stats(config.canonical_name, processed_root)
        graph.graph["num_nodes_expected"] = stats.get("num_graph_nodes")
        graph.graph["num_edges_expected"] = stats.get("num_graph_edges")
    except FileNotFoundError:
        graph.graph["num_nodes_expected"] = None

    node_map: dict[str, Any] = {}
    for node in load_graph_nodes(config.canonical_name, processed_root, limit=limit_nodes):
        graph_node_id = _node_id_from_node(node, node_id_mode)
        node_map[node.id] = graph_node_id
        graph.add_node(
            graph_node_id,
            processed_id=node.id,
            label=node.label,
            type=node.type,
            weight=node.weight,
            metadata=node.metadata,
        )

    for edge in load_graph_edges(config.canonical_name, processed_root, limit=limit_edges):
        source = _node_id_from_processed_id(edge.source, node_id_mode, node_map)
        target = _node_id_from_processed_id(edge.target, node_id_mode, node_map)
        graph.add_edge(
            source,
            target,
            relation=edge.relation,
            weight=edge.weight,
            metadata=edge.metadata,
            evidence_doc_ids=edge.evidence_doc_ids,
        )

    return graph


def build_networkx_graph_from_train_split(
    dataset_name: str,
    raw_root: str | Path = "data/raw",
    *,
    directed: bool | None = None,
    limit_edges: int | None = None,
    include_isolated_nodes: bool = False,
    split_data: OGBSplitData | None = None,
) -> nx.Graph:
    """Build the visible graph from official train_edges only."""

    config = get_dataset_config(dataset_name)
    split = split_data if split_data is not None else load_ogb_split(config.canonical_name, raw_root)
    graph = _new_graph(config.canonical_name, directed if directed is not None else config.directed)
    graph.graph.update(
        {
            "source": "ogb_train",
            "edge_source": "official train_edges",
            "num_nodes_expected": split.num_nodes,
            "split_source": split.source,
            "include_isolated_nodes": include_isolated_nodes,
            "has_edge_weight": split.train_edge_weight is not None,
            "has_edge_year": split.train_edge_year is not None,
            "max_train_year": _max_year(split.train_edge_year, limit_edges),
        }
    )
    if include_isolated_nodes:
        graph.add_nodes_from(range(split.num_nodes))

    for index, source, target in _limited_edges_with_index(split.train_edges, limit_edges):
        weight = _edge_weight_at(split.train_edge_weight, index)
        year = _edge_year_at(split.train_edge_year, index)
        _add_or_merge_train_edge(
            graph,
            int(source),
            int(target),
            relation=config.edge_relation_name,
            weight=weight,
            year=year,
        )
    return graph


def build_visible_graph(
    dataset_name: str,
    *,
    source: str = "ogb_train",
    processed_root: str | Path = "data/processed",
    raw_root: str | Path = "data/raw",
    directed: bool | None = None,
    limit_edges: int | None = None,
    limit_nodes: int | None = None,
    include_isolated_nodes: bool = False,
    node_id_mode: NodeIdMode = "node_idx",
) -> nx.Graph:
    """Build a graph from the requested visible source."""

    if source == "ogb_train":
        return build_networkx_graph_from_train_split(
            dataset_name,
            raw_root=raw_root,
            directed=directed,
            limit_edges=limit_edges,
            include_isolated_nodes=include_isolated_nodes,
        )
    if source == "processed":
        return build_networkx_graph_from_processed(
            dataset_name,
            processed_root=processed_root,
            directed=directed,
            limit_edges=limit_edges,
            limit_nodes=limit_nodes,
            node_id_mode=node_id_mode,
        )
    raise ValueError("source must be 'ogb_train' or 'processed'")


def _new_graph(dataset_name: str, directed: bool) -> nx.Graph:
    graph: nx.Graph = nx.DiGraph() if directed else nx.Graph()
    graph.graph["dataset_name"] = dataset_name
    graph.graph["directed"] = directed
    return graph


def _limited_edges(edges: np.ndarray, limit: int | None) -> Iterable[tuple[int, int]]:
    count = len(edges) if limit is None else min(limit, len(edges))
    for index in range(count):
        yield int(edges[index, 0]), int(edges[index, 1])


def _limited_edges_with_index(
    edges: np.ndarray,
    limit: int | None,
) -> Iterable[tuple[int, int, int]]:
    count = len(edges) if limit is None else min(limit, len(edges))
    for index in range(count):
        yield index, int(edges[index, 0]), int(edges[index, 1])


def _add_or_merge_train_edge(
    graph: nx.Graph,
    source: int,
    target: int,
    *,
    relation: str,
    weight: float,
    year: int | None,
) -> None:
    if graph.has_edge(source, target):
        attrs = graph[source][target]
        attrs["weight"] = float(attrs.get("weight", 0.0)) + weight
        attrs["edge_count"] = int(attrs.get("edge_count", 1)) + 1
        attrs["min_year"] = _merge_min_year(attrs.get("min_year"), year)
        attrs["max_year"] = _merge_max_year(attrs.get("max_year"), year)
        attrs["relation"] = relation
        return

    graph.add_edge(
        source,
        target,
        relation=relation,
        weight=weight,
        edge_count=1,
        min_year=year,
        max_year=year,
    )


def _edge_weight_at(values: np.ndarray | None, index: int) -> float:
    if values is None or index >= len(values):
        return 1.0
    try:
        return float(values[index])
    except (TypeError, ValueError):
        return 1.0


def _edge_year_at(values: np.ndarray | None, index: int) -> int | None:
    if values is None or index >= len(values):
        return None
    try:
        return int(values[index])
    except (TypeError, ValueError):
        return None


def _max_year(values: np.ndarray | None, limit: int | None) -> int | None:
    if values is None:
        return None
    bounded = values if limit is None else values[:limit]
    if len(bounded) == 0:
        return None
    return int(np.max(bounded))


def _merge_min_year(existing: Any, new_year: int | None) -> int | None:
    if existing is None:
        return new_year
    if new_year is None:
        return int(existing)
    return min(int(existing), new_year)


def _merge_max_year(existing: Any, new_year: int | None) -> int | None:
    if existing is None:
        return new_year
    if new_year is None:
        return int(existing)
    return max(int(existing), new_year)


def _node_id_from_node(node: GraphNode, node_id_mode: NodeIdMode) -> int | str:
    if node_id_mode == "processed_id":
        return node.id
    node_idx = node.node_idx
    if node_idx is None:
        raise ValueError(f"Node {node.id!r} is missing integer metadata.node_idx")
    return node_idx


def _node_id_from_processed_id(
    processed_id: str,
    node_id_mode: NodeIdMode,
    node_map: dict[str, Any],
) -> int | str:
    if node_id_mode == "processed_id":
        return processed_id
    if processed_id in node_map:
        return node_map[processed_id]
    try:
        return int(processed_id.rsplit(":node:", 1)[1])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"Cannot derive node_idx from processed id: {processed_id!r}") from exc
