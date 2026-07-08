"""OGB-aware candidate scoring for classical link prediction methods."""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

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


DEFAULT_OFFICIAL_TRAIN_EDGE_LIMIT = 50_000


@dataclass(frozen=True)
class OfficialOGBResult:
    """OGB Evaluator-backed result for one official-style collab run."""

    dataset: str
    split: EvalSplit
    method: str
    decay: float
    pos_used: int
    neg_per_pos_used: int | None
    total_neg_used: int
    official_mode: bool
    hits_at_50: float | None
    graph_metadata: dict = field(default_factory=dict)
    runtime_seconds: float = 0.0
    y_pred_pos_shape: tuple[int, ...] = ()
    y_pred_neg_shape: tuple[int, ...] = ()
    edge_neg_shape: tuple[int, ...] | None = None
    official_negatives_available: bool = False
    negative_layout: str = ""
    evaluator_metric: str = "hits@50"
    notes: str = ""
    error: str = ""


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
    requested_negative_count: int | None = None
    available_negative_count: int | None = None
    negative_truncated: bool | None = None
    positive_split_full: bool = False
    notes: str = ""
    error: str = ""

    @property
    def positive_count(self) -> int:
        """Number of positive candidate scores actually produced."""

        return int(self.pos_scores.reshape(-1).shape[0])

    @property
    def negative_count(self) -> int:
        """Number of negative candidate scores actually produced."""

        if self.neg_scores is not None:
            return int(self.neg_scores.reshape(-1).shape[0])
        if self.neg_scores_matrix is not None:
            return int(self.neg_scores_matrix.size)
        return 0


def score_candidates_for_dataset(
    dataset_name: str,
    method: str,
    split: Literal["valid", "test"],
    raw_root: str = "data/raw",
    limit_pos: int | None = None,
    limit_neg_per_pos: int | None = None,
    limit_train_edges: int | None = None,
    full_positive_split: bool = False,
    decay: float = 0.9,
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
            full_positive_split,
            decay,
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
        full_positive_split,
        decay,
    )


def score_multiple_methods_for_dataset(
    dataset_name: str,
    methods: list[str] | tuple[str, ...],
    split: Literal["valid", "test"],
    raw_root: str = "data/raw",
    limit_pos: int | None = None,
    limit_neg_per_pos: int | None = None,
    limit_train_edges: int | None = None,
    full_positive_split: bool = False,
    decay: float = 0.9,
    continue_on_error: bool = False,
) -> list[DatasetCandidateScores]:
    """Score several methods while reusing split, candidates, graph, and topology."""

    canonical_name = require_supported_dataset(dataset_name)
    normalized_methods = [normalize_method(method) for method in methods]
    split_data = load_ogb_split(canonical_name, raw_root)
    candidates = candidates_from_split(split_data, split)
    graph = build_networkx_graph_from_train_split(
        canonical_name,
        raw_root=raw_root,
        limit_edges=limit_train_edges,
        include_isolated_nodes=True,
        split_data=split_data,
    )
    topology_graph = to_simple_undirected_for_topology(graph, include_isolated_nodes=False)

    results: list[DatasetCandidateScores] = []
    for method in normalized_methods:
        try:
            if canonical_name == "ogbl_citation2":
                results.append(
                    _score_citation2_candidates(
                        graph,
                        canonical_name,
                        method,
                        split,
                        candidates,
                        limit_pos,
                        limit_neg_per_pos,
                        limit_train_edges,
                        full_positive_split,
                        decay,
                        topology_graph=topology_graph,
                    )
                )
                continue

            results.append(
                _score_edge_candidates(
                    graph,
                    canonical_name,
                    method,
                    split,
                    candidates.positive_edges,
                    candidates.negative_edges,
                    limit_pos,
                    limit_neg_per_pos,
                    limit_train_edges,
                    full_positive_split,
                    decay,
                    topology_graph=topology_graph,
                )
            )
        except Exception as exc:  # noqa: BLE001 - caller opted into row-level errors.
            if not continue_on_error:
                raise
            results.append(
                _error_result(
                    graph,
                    topology_graph,
                    canonical_name,
                    method,
                    split,
                    limit_train_edges,
                    full_positive_split,
                    decay,
                    exc,
                )
            )
    return results


def score_ogb_official_for_dataset(
    dataset_name: str,
    method: str,
    split_name: Literal["valid", "test"],
    raw_root: str | Path = "data/raw",
    full_train_graph: bool = True,
    limit_pos: int | None = None,
    limit_neg_per_pos: int | None = None,
    decay: float = 0.8,
    batch_size: int = 1000,
    require_per_positive_negatives: bool = False,
) -> OfficialOGBResult:
    """Score one method with OGB's official ``ogbl-collab`` Evaluator."""

    return score_ogb_official_multiple_methods(
        dataset_name=dataset_name,
        methods=[method],
        split_name=split_name,
        raw_root=raw_root,
        full_train_graph=full_train_graph,
        limit_pos=limit_pos,
        limit_neg_per_pos=limit_neg_per_pos,
        decay=decay,
        batch_size=batch_size,
        require_per_positive_negatives=require_per_positive_negatives,
    )[0]


def score_ogb_official_multiple_methods(
    dataset_name: str,
    methods: Sequence[str],
    split_name: Literal["valid", "test"],
    raw_root: str | Path = "data/raw",
    full_train_graph: bool = True,
    limit_pos: int | None = None,
    limit_neg_per_pos: int | None = None,
    decay: float = 0.8,
    batch_size: int = 1000,
    require_per_positive_negatives: bool = False,
    continue_on_error: bool = False,
) -> list[OfficialOGBResult]:
    """Run OGB official-style Hits@50 evaluation for ``ogbl-collab``.

    The split and train graph are loaded once, and every method reuses the same
    graph/topology projection. A 3D ``edge_neg`` is scored row-wise as
    ``[num_pos, num_neg_per_pos]``. A 2D ``edge_neg`` is treated as the shared
    negative pool accepted by the installed OGB ``ogbl-collab`` Hits evaluator;
    this is reported explicitly instead of being presented as per-positive data.
    """

    canonical_name = require_supported_dataset(dataset_name)
    if canonical_name != "ogbl_collab":
        raise ValueError("OGB official mode currently supports only ogbl_collab")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    normalized_methods = [normalize_method(method) for method in methods]
    split_data = load_ogb_split(canonical_name, raw_root)
    train_edge_limit = None if full_train_graph else DEFAULT_OFFICIAL_TRAIN_EDGE_LIMIT
    graph = build_networkx_graph_from_train_split(
        canonical_name,
        raw_root=raw_root,
        limit_edges=train_edge_limit,
        include_isolated_nodes=True,
        split_data=split_data,
    )
    topology_graph = to_simple_undirected_for_topology(graph, include_isolated_nodes=False)
    evaluator = _make_ogb_evaluator(split_data.ogb_name)

    positive_edges, negative_edges, negative_layout, notes = _official_edges_for_split(
        split_data,
        split_name,
        limit_pos=limit_pos,
        limit_neg_per_pos=limit_neg_per_pos,
        require_per_positive_negatives=require_per_positive_negatives,
    )

    results: list[OfficialOGBResult] = []
    for method in normalized_methods:
        started = time.perf_counter()
        try:
            y_pred_pos, y_pred_neg = _score_official_prediction_arrays(
                graph,
                positive_edges,
                negative_edges,
                negative_layout,
                method,
                decay,
                batch_size,
                topology_graph,
            )
            eval_result = evaluator.eval(
                {"y_pred_pos": y_pred_pos, "y_pred_neg": y_pred_neg}
            )
            hits_at_50 = _extract_hits_at_50(eval_result)
            runtime = time.perf_counter() - started
            results.append(
                OfficialOGBResult(
                    dataset=canonical_name,
                    split=split_name,
                    method=method,
                    decay=decay,
                    pos_used=int(y_pred_pos.shape[0]),
                    neg_per_pos_used=_negative_per_positive_count(
                        y_pred_neg,
                        negative_layout,
                    ),
                    total_neg_used=int(y_pred_neg.size),
                    official_mode=True,
                    hits_at_50=hits_at_50,
                    graph_metadata=_topology_metadata(graph, topology_graph, decay),
                    runtime_seconds=runtime,
                    y_pred_pos_shape=_shape_tuple(y_pred_pos),
                    y_pred_neg_shape=_shape_tuple(y_pred_neg),
                    edge_neg_shape=_shape_tuple(negative_edges),
                    official_negatives_available=True,
                    negative_layout=negative_layout,
                    notes=_official_notes(notes, train_edge_limit),
                )
            )
        except Exception as exc:  # noqa: BLE001 - caller may want table rows.
            if not continue_on_error:
                raise
            runtime = time.perf_counter() - started
            results.append(
                _official_error_result(
                    graph,
                    topology_graph,
                    canonical_name,
                    split_name,
                    method,
                    decay,
                    negative_edges,
                    negative_layout,
                    notes,
                    train_edge_limit,
                    runtime,
                    exc,
                )
            )
    return results


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
    full_positive_split: bool,
    decay: float,
    *,
    topology_graph: nx.Graph | None = None,
) -> DatasetCandidateScores:
    if negative_edges is None:
        raise ValueError(f"{dataset_name} {split} split is missing negative edge candidates")

    positive_limit = _bounded_count(
        len(positive_edges),
        _effective_positive_limit(limit_pos, full_positive_split),
    )
    positive_pairs = edges_to_node_pairs(positive_edges, limit=positive_limit)
    requested_neg, available_neg, negative_limit, negative_truncated = _negative_count_metadata(
        len(negative_edges),
        positive_limit,
        limit_neg_per_pos,
    )
    negative_pairs = edges_to_node_pairs(negative_edges, limit=negative_limit)

    all_predictions = score_candidate_pairs_in_order(
        graph,
        [*positive_pairs, *negative_pairs],
        method,
        decay=decay,
        topology_graph=topology_graph,
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
        graph_metadata=_topology_metadata(graph, topology_graph, decay),
        requested_negative_count=requested_neg,
        available_negative_count=available_neg,
        negative_truncated=negative_truncated,
        positive_split_full=full_positive_split,
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
    full_positive_split: bool,
    decay: float,
    *,
    topology_graph: nx.Graph | None = None,
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

    row_count = _bounded_count(
        len(source_nodes),
        _effective_positive_limit(limit_pos, full_positive_split),
    )
    source_nodes = source_nodes[:row_count]
    target_nodes = target_nodes[:row_count]
    target_node_neg = target_node_neg[:row_count]
    requested_neg, available_neg, negative_limit, negative_truncated = (
        _citation2_negative_count_metadata(target_node_neg, row_count, limit_neg_per_pos)
    )
    if target_node_neg.shape[1] > negative_limit:
        target_node_neg = target_node_neg[:, :negative_limit]

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
        decay=decay,
        topology_graph=topology_graph,
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
        graph_metadata=_topology_metadata(graph, topology_graph, decay),
        requested_negative_count=requested_neg,
        available_negative_count=available_neg,
        negative_truncated=negative_truncated,
        positive_split_full=full_positive_split,
        notes=(
            _scoring_notes(dataset_name, limit_train_edges)
            + " target_node_neg is preserved as a row-wise matrix for MRR."
        ),
    )


def _official_edges_for_split(
    split_data,
    split: EvalSplit,
    *,
    limit_pos: int | None,
    limit_neg_per_pos: int | None,
    require_per_positive_negatives: bool,
) -> tuple[np.ndarray, np.ndarray, str, str]:
    if split == "valid":
        positive_edges = split_data.valid_edges
        edge_neg = split_data.valid_edge_neg
        flat_edge_neg = split_data.valid_neg_edges
    elif split == "test":
        positive_edges = split_data.test_edges
        edge_neg = split_data.test_edge_neg
        flat_edge_neg = split_data.test_neg_edges
    else:
        raise ValueError("split_name must be 'valid' or 'test'")

    if edge_neg is None:
        edge_neg = flat_edge_neg
    if edge_neg is None:
        raise ValueError(f"{split_data.dataset_name} {split} split is missing edge_neg")

    layout = _official_negative_layout(edge_neg)
    if layout == "per_positive" and edge_neg.shape[0] != len(positive_edges):
        raise ValueError(
            f"{split_data.dataset_name} {split} edge_neg has {edge_neg.shape[0]} "
            f"positive rows but positive edge split has {len(positive_edges)} rows"
        )
    if layout == "shared_pool" and require_per_positive_negatives:
        raise ValueError(
            "Strict per-positive official evaluation requires edge_neg shape "
            f"[num_pos, num_neg, 2], got {edge_neg.shape}. This split exposes a "
            "2D shared negative pool; use the shared-pool OGB Evaluator fallback "
            "only when that is the intended protocol."
        )

    pos_count = _bounded_count(len(positive_edges), limit_pos)
    positive_edges = positive_edges[:pos_count]

    if layout == "per_positive":
        edge_neg = edge_neg[:pos_count]
        neg_per_pos = _bounded_count(edge_neg.shape[1], limit_neg_per_pos)
        edge_neg = edge_neg[:, :neg_per_pos, :]
        notes = (
            "edge_neg is 3D per-positive data; y_pred_neg is scored as "
            "[num_pos, num_neg_per_pos]."
        )
    else:
        neg_count = _bounded_count(edge_neg.shape[0], limit_neg_per_pos)
        edge_neg = edge_neg[:neg_count]
        notes = (
            "edge_neg is a 2D shared negative pool. The installed OGB "
            "ogbl-collab Hits evaluator accepts y_pred_neg as a 1D shared "
            "pool, so this official fallback is explicit and not row-wise."
        )

    return positive_edges, edge_neg, layout, notes


def _official_negative_layout(edge_neg: np.ndarray) -> str:
    if edge_neg.ndim == 3 and edge_neg.shape[-1] == 2:
        return "per_positive"
    if edge_neg.ndim == 2 and edge_neg.shape[-1] == 2:
        return "shared_pool"
    raise ValueError(
        "edge_neg must have shape [num_neg, 2] or [num_pos, num_neg, 2], "
        f"got {edge_neg.shape}"
    )


def _score_official_prediction_arrays(
    graph: nx.Graph,
    positive_edges: np.ndarray,
    negative_edges: np.ndarray,
    negative_layout: str,
    method: str,
    decay: float,
    batch_size: int,
    topology_graph: nx.Graph,
) -> tuple[np.ndarray, np.ndarray]:
    if negative_layout == "per_positive":
        return _score_per_positive_official_arrays(
            graph,
            positive_edges,
            negative_edges,
            method,
            decay,
            batch_size,
            topology_graph,
        )
    if negative_layout == "shared_pool":
        y_pred_pos = _score_edge_array_batches(
            graph,
            positive_edges,
            method,
            decay,
            batch_size,
            topology_graph,
        )
        y_pred_neg = _score_edge_array_batches(
            graph,
            negative_edges,
            method,
            decay,
            batch_size,
            topology_graph,
        )
        return y_pred_pos, y_pred_neg
    raise ValueError(f"Unsupported official negative layout: {negative_layout}")


def _score_per_positive_official_arrays(
    graph: nx.Graph,
    positive_edges: np.ndarray,
    negative_edges: np.ndarray,
    method: str,
    decay: float,
    batch_size: int,
    topology_graph: nx.Graph,
) -> tuple[np.ndarray, np.ndarray]:
    pos_batches: list[np.ndarray] = []
    neg_batches: list[np.ndarray] = []
    neg_per_pos = int(negative_edges.shape[1])

    for batch_start in range(0, len(positive_edges), batch_size):
        batch_end = min(batch_start + batch_size, len(positive_edges))
        batch_pos_edges = positive_edges[batch_start:batch_end]
        batch_neg_edges = negative_edges[batch_start:batch_end]
        positive_pairs = edges_to_node_pairs(batch_pos_edges)
        negative_pairs = edges_to_node_pairs(batch_neg_edges.reshape(-1, 2))
        predictions = score_candidate_pairs_in_order(
            graph,
            [*positive_pairs, *negative_pairs],
            method,
            decay=decay,
            topology_graph=topology_graph,
        )
        pos_predictions = predictions[: len(positive_pairs)]
        neg_predictions = predictions[len(positive_pairs) :]
        pos_batches.append(_scores_array(pos_predictions))
        neg_batches.append(_scores_array(neg_predictions).reshape(-1, neg_per_pos))

    return _concat_1d(pos_batches), _concat_2d(neg_batches, neg_per_pos)


def _score_edge_array_batches(
    graph: nx.Graph,
    edges: np.ndarray,
    method: str,
    decay: float,
    batch_size: int,
    topology_graph: nx.Graph,
) -> np.ndarray:
    batches: list[np.ndarray] = []
    for batch_start in range(0, len(edges), batch_size):
        batch_end = min(batch_start + batch_size, len(edges))
        pairs = edges_to_node_pairs(edges[batch_start:batch_end])
        predictions = score_candidate_pairs_in_order(
            graph,
            pairs,
            method,
            decay=decay,
            topology_graph=topology_graph,
        )
        batches.append(_scores_array(predictions))
    return _concat_1d(batches)


def _concat_1d(batches: list[np.ndarray]) -> np.ndarray:
    if not batches:
        return np.asarray([], dtype=float)
    return np.concatenate(batches).astype(float, copy=False)


def _concat_2d(batches: list[np.ndarray], width: int) -> np.ndarray:
    if not batches:
        return np.empty((0, width), dtype=float)
    return np.concatenate(batches, axis=0).astype(float, copy=False)


def _make_ogb_evaluator(ogb_name: str):
    try:
        from ogb.linkproppred import Evaluator
    except ImportError as exc:
        raise ImportError("Install the 'ogb' package to run official evaluation.") from exc

    return Evaluator(name=ogb_name)


def _extract_hits_at_50(result: dict[str, Any]) -> float:
    for key in ("hits@50", "Hits@50"):
        if key in result:
            return float(result[key])
    raise ValueError(f"OGB Evaluator result did not contain hits@50: {result}")


def _negative_per_positive_count(
    y_pred_neg: np.ndarray,
    negative_layout: str,
) -> int | None:
    if negative_layout == "per_positive":
        return int(y_pred_neg.shape[1])
    return None


def _shape_tuple(value: Any) -> tuple[int, ...] | None:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    return tuple(int(item) for item in shape)


def _official_notes(notes: str, train_edge_limit: int | None) -> str:
    suffix = " Visible graph uses official train_edges."
    if train_edge_limit is not None:
        suffix += f" Debug graph limited to first {train_edge_limit} train edges."
    return notes + suffix


def _official_error_result(
    graph: nx.Graph,
    topology_graph: nx.Graph,
    dataset_name: str,
    split: EvalSplit,
    method: str,
    decay: float,
    negative_edges: np.ndarray,
    negative_layout: str,
    notes: str,
    train_edge_limit: int | None,
    runtime_seconds: float,
    exc: Exception,
) -> OfficialOGBResult:
    return OfficialOGBResult(
        dataset=dataset_name,
        split=split,
        method=method,
        decay=decay,
        pos_used=0,
        neg_per_pos_used=None,
        total_neg_used=0,
        official_mode=True,
        hits_at_50=None,
        graph_metadata=_topology_metadata(graph, topology_graph, decay),
        runtime_seconds=runtime_seconds,
        edge_neg_shape=_shape_tuple(negative_edges),
        official_negatives_available=True,
        negative_layout=negative_layout,
        notes=_official_notes(notes, train_edge_limit),
        error=f"{type(exc).__name__}: {exc}",
    )


def _effective_positive_limit(limit_pos: int | None, full_positive_split: bool) -> int | None:
    if full_positive_split:
        return None
    return limit_pos


def _bounded_count(total: int, limit: int | None) -> int:
    if limit is None:
        return total
    if limit < 0:
        raise ValueError("limit values must be non-negative")
    return min(total, limit)


def _negative_count_metadata(
    total_negative_edges: int,
    positive_count: int,
    limit_neg_per_pos: int | None,
) -> tuple[int, int, int, bool]:
    available_negative_count = int(total_negative_edges)
    if limit_neg_per_pos is None:
        return (
            available_negative_count,
            available_negative_count,
            available_negative_count,
            False,
        )
    if limit_neg_per_pos < 0:
        raise ValueError("limit values must be non-negative")
    requested_negative_count = int(positive_count * limit_neg_per_pos)
    used_negative_count = min(available_negative_count, requested_negative_count)
    return (
        requested_negative_count,
        available_negative_count,
        used_negative_count,
        used_negative_count < requested_negative_count,
    )


def _citation2_negative_count_metadata(
    target_node_neg: np.ndarray,
    positive_count: int,
    limit_neg_per_pos: int | None,
) -> tuple[int, int, int, bool]:
    available_per_positive = int(target_node_neg.shape[1])
    available_negative_count = int(positive_count * available_per_positive)
    if limit_neg_per_pos is None:
        return (
            available_negative_count,
            available_negative_count,
            available_per_positive,
            False,
        )
    if limit_neg_per_pos < 0:
        raise ValueError("limit values must be non-negative")
    requested_negative_count = int(positive_count * limit_neg_per_pos)
    used_per_positive = min(available_per_positive, limit_neg_per_pos)
    used_negative_count = int(positive_count * used_per_positive)
    return (
        requested_negative_count,
        available_negative_count,
        used_per_positive,
        used_negative_count < requested_negative_count,
    )


def _scores_array(predictions: list[LinkPredictionScore]) -> np.ndarray:
    return np.asarray([prediction.score for prediction in predictions], dtype=float)


def _error_result(
    graph: nx.Graph,
    topology_graph: nx.Graph | None,
    dataset_name: str,
    method: str,
    split: EvalSplit,
    limit_train_edges: int | None,
    full_positive_split: bool,
    decay: float,
    exc: Exception,
) -> DatasetCandidateScores:
    return DatasetCandidateScores(
        dataset_name=dataset_name,
        method=method,
        split=split,
        pos_scores=np.asarray([], dtype=float),
        neg_scores=np.asarray([], dtype=float),
        metric_name=_metric_name(dataset_name),
        graph_metadata=_topology_metadata(graph, topology_graph, decay),
        positive_split_full=full_positive_split,
        notes=_scoring_notes(dataset_name, limit_train_edges),
        error=f"{type(exc).__name__}: {exc}",
    )


def _metric_name(dataset_name: str) -> str:
    canonical_name = get_dataset_config(dataset_name).canonical_name
    if canonical_name == "ogbl_collab":
        return "Hits@50"
    if canonical_name == "ogbl_ppa":
        return "Hits@100"
    if canonical_name == "ogbl_citation2":
        return "MRR"
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def _topology_metadata(
    graph: nx.Graph,
    topology_graph: nx.Graph | None = None,
    decay: float = 0.9,
) -> dict:
    topology = topology_graph
    if topology is None:
        topology = to_simple_undirected_for_topology(graph, include_isolated_nodes=False)
    return {
        "dataset_name": graph.graph.get("dataset_name"),
        "graph_source": graph.graph.get("source"),
        "num_nodes": graph.number_of_nodes(),
        "num_edges": graph.number_of_edges(),
        "include_isolated_nodes": graph.graph.get("include_isolated_nodes"),
        "has_edge_weight": bool(graph.graph.get("has_edge_weight")),
        "has_edge_year": bool(graph.graph.get("has_edge_year")),
        "max_train_year": graph.graph.get("max_train_year"),
        "decay": decay,
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
