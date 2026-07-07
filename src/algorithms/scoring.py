"""OGB-aware candidate scoring for classical link prediction methods."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import networkx as nx
import numpy as np

from src.graph.candidates import candidates_from_split, edges_to_node_pairs
from src.graph.dataset_registry import get_dataset_config, require_supported_dataset
from src.graph.graph_factory import build_networkx_graph_from_train_split
from src.graph.ogb_split_loader import load_ogb_split
from src.graph.schemas import EvalSplit

from .link_prediction import (
    LinkPredictionScore,
    normalize_method,
    score_candidate_pairs_in_order,
    to_simple_undirected_for_topology,
)


@dataclass(frozen=True)
class DatasetCandidateScores:
    """Scored OGB candidates for one dataset, split, and method."""

    dataset_name: str
    method: str
    split: EvalSplit
    pos_scores: np.ndarray
    neg_scores: np.ndarray | None = None
    neg_scores_matrix: np.ndarray | None = None
    pos_predictions: list[LinkPredictionScore] = field(default_factory=list)
    neg_predictions: list[LinkPredictionScore] = field(default_factory=list)
    metric_name: str = ""
    graph_metadata: dict = field(default_factory=dict)
    notes: str = ""


def score_candidates_for_dataset(
    dataset_name: str,
    method: str,
    split: Literal["valid", "test"],
    raw_root: str = "data/raw",
    limit_pos: int | None = None,
    limit_neg_per_pos: int | None = None,
    limit_train_edges: int | None = None,
) -> DatasetCandidateScores:
    """Score official OGB candidates with a classical topology method.

    The visible graph is built only from official train edges. Valid/test
    positives are never added to the graph. Candidate scoring is always limited
    to the supplied official candidates and never calls graph-wide non-edge
    enumeration.
    """

    canonical_name = require_supported_dataset(dataset_name)
    normalized_method = normalize_method(method)
    split_data = load_ogb_split(canonical_name, raw_root)
    candidates = candidates_from_split(split_data, split)
    graph = build_networkx_graph_from_train_split(
        canonical_name,
        raw_root=raw_root,
        limit_edges=limit_train_edges,
        include_isolated_nodes=True,
        split_data=split_data,
    )

    if canonical_name == "ogbl_citation2":
        return _score_citation2_candidates(
            graph,
            canonical_name,
            normalized_method,
            split,
            candidates,
            limit_pos,
            limit_neg_per_pos,
            limit_train_edges,
        )

    return _score_edge_candidates(
        graph,
        canonical_name,
        normalized_method,
        split,
        candidates.positive_edges,
        candidates.negative_edges,
        limit_pos,
        limit_neg_per_pos,
        limit_train_edges,
    )


def _score_edge_candidates(
    graph: nx.Graph,
    dataset_name: str,
    method: str,
    split: EvalSplit,
    positive_edges: np.ndarray,
    negative_edges: np.ndarray | None,
    limit_pos: int | None,
    limit_neg_per_pos: int | None,
    limit_train_edges: int | None,
) -> DatasetCandidateScores:
    if negative_edges is None:
        raise ValueError(f"{dataset_name} {split} split is missing negative edge candidates")

    positive_limit = _bounded_count(len(positive_edges), limit_pos)
    positive_pairs = edges_to_node_pairs(positive_edges, limit=positive_limit)
    negative_limit = _negative_edge_limit(len(negative_edges), positive_limit, limit_neg_per_pos)
    negative_pairs = edges_to_node_pairs(negative_edges, limit=negative_limit)

    all_predictions = score_candidate_pairs_in_order(
        graph,
        [*positive_pairs, *negative_pairs],
        method,
    )
    pos_predictions = all_predictions[: len(positive_pairs)]
    neg_predictions = all_predictions[len(positive_pairs) :]

    return DatasetCandidateScores(
        dataset_name=dataset_name,
        method=method,
        split=split,
        pos_scores=_scores_array(pos_predictions),
        neg_scores=_scores_array(neg_predictions),
        pos_predictions=pos_predictions,
        neg_predictions=neg_predictions,
        metric_name=_metric_name(dataset_name),
        graph_metadata=_topology_metadata(graph),
        notes=_scoring_notes(dataset_name, limit_train_edges),
    )


def _score_citation2_candidates(
    graph: nx.Graph,
    dataset_name: str,
    method: str,
    split: EvalSplit,
    candidates,
    limit_pos: int | None,
    limit_neg_per_pos: int | None,
    limit_train_edges: int | None,
) -> DatasetCandidateScores:
    source_nodes = candidates.source_nodes
    target_nodes = candidates.target_nodes
    target_node_neg = candidates.target_node_neg

    if source_nodes is None or target_nodes is None:
        positive_edges = candidates.positive_edges
        source_nodes = positive_edges[:, 0]
        target_nodes = positive_edges[:, 1]
    if target_node_neg is None:
        raise ValueError(
            "ogbl_citation2 requires target_node_neg matrix for MRR-style scoring"
        )

    row_count = _bounded_count(len(source_nodes), limit_pos)
    source_nodes = source_nodes[:row_count]
    target_nodes = target_nodes[:row_count]
    target_node_neg = target_node_neg[:row_count]
    if limit_neg_per_pos is not None:
        target_node_neg = target_node_neg[:, :limit_neg_per_pos]

    positive_pairs = [(int(source), int(target)) for source, target in zip(source_nodes, target_nodes)]
    negative_pairs = [
        (int(source), int(target))
        for source, row in zip(source_nodes, target_node_neg, strict=True)
        for target in row
    ]

    all_predictions = score_candidate_pairs_in_order(
        graph,
        [*positive_pairs, *negative_pairs],
        method,
    )
    pos_predictions = all_predictions[: len(positive_pairs)]
    neg_predictions = all_predictions[len(positive_pairs) :]
    neg_scores_matrix = _scores_array(neg_predictions).reshape(target_node_neg.shape)

    return DatasetCandidateScores(
        dataset_name=dataset_name,
        method=method,
        split=split,
        pos_scores=_scores_array(pos_predictions),
        neg_scores_matrix=neg_scores_matrix,
        pos_predictions=pos_predictions,
        neg_predictions=neg_predictions,
        metric_name="MRR",
        graph_metadata=_topology_metadata(graph),
        notes=(
            _scoring_notes(dataset_name, limit_train_edges)
            + " target_node_neg is preserved as a row-wise matrix for MRR."
        ),
    )


def _bounded_count(total: int, limit: int | None) -> int:
    if limit is None:
        return total
    if limit < 0:
        raise ValueError("limit values must be non-negative")
    return min(total, limit)


def _negative_edge_limit(
    total_negative_edges: int,
    positive_count: int,
    limit_neg_per_pos: int | None,
) -> int:
    if limit_neg_per_pos is None:
        return total_negative_edges
    if limit_neg_per_pos < 0:
        raise ValueError("limit values must be non-negative")
    return min(total_negative_edges, positive_count * limit_neg_per_pos)


def _scores_array(predictions: list[LinkPredictionScore]) -> np.ndarray:
    return np.asarray([prediction.score for prediction in predictions], dtype=float)


def _metric_name(dataset_name: str) -> str:
    canonical_name = get_dataset_config(dataset_name).canonical_name
    if canonical_name == "ogbl_collab":
        return "Hits@50"
    if canonical_name == "ogbl_ppa":
        return "Hits@100"
    if canonical_name == "ogbl_citation2":
        return "MRR"
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def _topology_metadata(graph: nx.Graph) -> dict:
    topology = to_simple_undirected_for_topology(graph, include_isolated_nodes=False)
    return {
        "dataset_name": graph.graph.get("dataset_name"),
        "graph_source": graph.graph.get("source"),
        "num_nodes": graph.number_of_nodes(),
        "num_edges": graph.number_of_edges(),
        "include_isolated_nodes": graph.graph.get("include_isolated_nodes"),
        "topology_num_nodes": topology.number_of_nodes(),
        "topology_num_edges": topology.number_of_edges(),
        "topology_projection": topology.graph.get("topology_projection"),
        "projection_note": topology.graph.get("projection_note", ""),
        "citation2_topology_note": topology.graph.get("citation2_topology_note", ""),
    }


def _scoring_notes(dataset_name: str, limit_train_edges: int | None) -> str:
    notes = (
        "Visible graph uses official train_edges only with include_isolated_nodes=True. "
        "Valid/test scoring is candidate-limited."
    )
    if dataset_name == "ogbl_citation2":
        notes += (
            " Citation2 directed edges are projected to an undirected simple graph "
            "for this classical baseline."
        )
    if limit_train_edges is not None:
        notes += f" Smoke graph limited to first {limit_train_edges} train edges."
    return notes
