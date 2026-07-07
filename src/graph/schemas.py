"""Lightweight data structures for graph construction and split handling."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np


NodeIdMode = Literal["node_idx", "processed_id"]
GraphSource = Literal["processed", "ogb_train"]
EvalSplit = Literal["valid", "test"]


@dataclass(frozen=True)
class DatasetConfig:
    """Manual registry entry for one supported graph dataset."""

    canonical_name: str
    ogb_name: str
    processed_dir_name: str
    raw_dir_name: str
    task_type: str
    directed: bool
    edge_relation_name: str
    processed_graph_is_aggregated: bool
    use_official_split_for_metrics: bool
    notes: str = ""


@dataclass(frozen=True)
class GraphNode:
    """One record from processed graph_nodes.jsonl."""

    id: str
    label: str
    type: str
    weight: float
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def node_idx(self) -> int | None:
        value = self.metadata.get("node_idx")
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None


@dataclass(frozen=True)
class GraphEdge:
    """One record from processed graph_edges.jsonl."""

    source: str
    target: str
    relation: str
    weight: float
    metadata: dict[str, Any] = field(default_factory=dict)
    evidence_doc_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class OGBSplitData:
    """Official OGB split arrays normalized for graph construction."""

    dataset_name: str
    ogb_name: str
    num_nodes: int
    train_edges: np.ndarray
    valid_edges: np.ndarray
    test_edges: np.ndarray
    valid_neg_edges: np.ndarray | None = None
    test_neg_edges: np.ndarray | None = None
    valid_source_nodes: np.ndarray | None = None
    valid_target_nodes: np.ndarray | None = None
    test_source_nodes: np.ndarray | None = None
    test_target_nodes: np.ndarray | None = None
    valid_target_node_neg: np.ndarray | None = None
    test_target_node_neg: np.ndarray | None = None
    source: str = ""
    notes: str = ""


@dataclass(frozen=True)
class CandidateEdges:
    """Candidate-limited evaluation inputs for one OGB split."""

    dataset_name: str
    split: EvalSplit
    positive_edges: np.ndarray
    negative_edges: np.ndarray | None = None
    source_nodes: np.ndarray | None = None
    target_nodes: np.ndarray | None = None
    target_node_neg: np.ndarray | None = None
    notes: str = ""


@dataclass(frozen=True)
class GraphBuildConfig:
    """Options for constructing a NetworkX graph."""

    dataset_name: str
    source: GraphSource = "ogb_train"
    directed: bool | None = None
    node_id_mode: NodeIdMode = "node_idx"
    limit_edges: int | None = None
    limit_nodes: int | None = None
    include_isolated_nodes: bool = False


@dataclass(frozen=True)
class GraphStats:
    """Small graph summary suitable for smoke tests and logs."""

    num_nodes: int
    num_edges: int
    is_directed: bool
    density: float | None
    average_degree: float
    max_degree: int
    self_loops: int
    num_components: int | None = None
    component_type: str | None = None
    components_skipped: bool = False
