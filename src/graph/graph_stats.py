"""Small graph and split summaries for smoke tests."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import networkx as nx

from .schemas import GraphStats, OGBSplitData


def compute_basic_graph_stats(
    graph: nx.Graph,
    *,
    compute_components: bool = True,
    max_component_nodes: int = 100_000,
    compute_density: bool = True,
) -> GraphStats:
    """Compute lightweight graph statistics without running heavy algorithms."""

    num_nodes = graph.number_of_nodes()
    num_edges = graph.number_of_edges()
    degrees = [degree for _, degree in graph.degree()]
    average_degree = sum(degrees) / num_nodes if num_nodes else 0.0
    max_degree = max(degrees) if degrees else 0
    density = nx.density(graph) if compute_density and num_nodes > 1 else None
    self_loops = nx.number_of_selfloops(graph)

    num_components: int | None = None
    component_type: str | None = None
    components_skipped = False
    if compute_components:
        if num_nodes > max_component_nodes:
            components_skipped = True
        elif graph.is_directed():
            num_components = nx.number_weakly_connected_components(graph)
            component_type = "weakly_connected_components"
        else:
            num_components = nx.number_connected_components(graph)
            component_type = "connected_components"

    return GraphStats(
        num_nodes=num_nodes,
        num_edges=num_edges,
        is_directed=graph.is_directed(),
        density=density,
        average_degree=average_degree,
        max_degree=max_degree,
        self_loops=self_loops,
        num_components=num_components,
        component_type=component_type,
        components_skipped=components_skipped,
    )


def summarize_split(split_data: OGBSplitData) -> dict[str, Any]:
    """Return split array shapes and source metadata."""

    return {
        "dataset_name": split_data.dataset_name,
        "source": split_data.source,
        "num_nodes": split_data.num_nodes,
        "train_edges": _shape(split_data.train_edges),
        "valid_edges": _shape(split_data.valid_edges),
        "test_edges": _shape(split_data.test_edges),
        "valid_edge_neg": _shape(split_data.valid_edge_neg),
        "test_edge_neg": _shape(split_data.test_edge_neg),
        "valid_official_negatives_available": _official_negatives_available(
            split_data.valid_edge_neg
        ),
        "test_official_negatives_available": _official_negatives_available(
            split_data.test_edge_neg
        ),
        "valid_official_negative_layout": _negative_layout(split_data.valid_edge_neg),
        "test_official_negative_layout": _negative_layout(split_data.test_edge_neg),
        "valid_neg_edges": _shape(split_data.valid_neg_edges),
        "test_neg_edges": _shape(split_data.test_neg_edges),
        "valid_target_node_neg": _shape(split_data.valid_target_node_neg),
        "test_target_node_neg": _shape(split_data.test_target_node_neg),
        "notes": split_data.notes,
    }


def print_graph_summary(name: str, stats: GraphStats | Mapping[str, Any]) -> str:
    """Format and print a compact graph summary."""

    if isinstance(stats, GraphStats):
        payload = stats.__dict__
    else:
        payload = dict(stats)

    parts = [
        f"{name}:",
        f"nodes={payload.get('num_nodes')}",
        f"edges={payload.get('num_edges')}",
        f"directed={payload.get('is_directed')}",
        f"avg_degree={payload.get('average_degree'):.4f}",
        f"max_degree={payload.get('max_degree')}",
        f"self_loops={payload.get('self_loops')}",
    ]
    if payload.get("num_components") is not None:
        parts.append(f"{payload.get('component_type')}={payload.get('num_components')}")
    elif payload.get("components_skipped"):
        parts.append("components=skipped")

    line = " ".join(parts)
    print(line)
    return line


def _shape(value: Any) -> tuple[int, ...] | None:
    if value is None:
        return None
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    return tuple(int(item) for item in shape)


def _official_negatives_available(value: Any) -> bool:
    return value is not None and getattr(value, "ndim", 0) in {2, 3}


def _negative_layout(value: Any) -> str:
    if value is None:
        return "none"
    ndim = getattr(value, "ndim", None)
    if ndim == 3:
        return "per_positive"
    if ndim == 2:
        return "shared_pool"
    return f"unsupported_{ndim}d"
