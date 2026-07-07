"""Manual OGB-style metrics for classical link prediction scores."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from src.graph.dataset_registry import require_supported_dataset


def hits_at_k_from_scores(
    pos_scores: Any,
    neg_scores: Any,
    k: int,
) -> float:
    """Compute Hits@K from positive scores and candidate negative scores.

    A 2D negative score array is interpreted row-wise: each positive score is
    compared only with the negatives in the same row. A 1D negative score array
    is interpreted as a shared global negative pool for every positive score.
    Ties are handled conservatively with ``negative >= positive``.
    """

    if k <= 0:
        raise ValueError("k must be positive")

    pos = np.asarray(pos_scores, dtype=float).reshape(-1)
    neg = np.asarray(neg_scores, dtype=float)
    if pos.size == 0:
        return 0.0
    if neg.size == 0:
        return 1.0

    if neg.ndim == 1:
        ranks = 1 + np.sum(neg.reshape(1, -1) >= pos.reshape(-1, 1), axis=1)
    elif neg.ndim == 2:
        if neg.shape[0] != pos.shape[0]:
            raise ValueError(
                "2D neg_scores must have the same number of rows as pos_scores"
            )
        ranks = 1 + np.sum(neg >= pos.reshape(-1, 1), axis=1)
    else:
        raise ValueError(f"neg_scores must be 1D or 2D, got shape {neg.shape}")
    return float(np.mean(ranks <= k))


def mrr_from_citation2_scores(pos_scores: Any, neg_scores_matrix: Any) -> float:
    """Compute citation2-style MRR from positive scores and negative matrix."""

    pos = np.asarray(pos_scores, dtype=float).reshape(-1)
    neg = np.asarray(neg_scores_matrix, dtype=float)
    if pos.size == 0:
        return 0.0
    if neg.ndim != 2:
        raise ValueError(f"neg_scores_matrix must be 2D, got shape {neg.shape}")
    if neg.shape[0] != pos.shape[0]:
        raise ValueError("neg_scores_matrix rows must match pos_scores length")

    ranks = 1 + np.sum(neg >= pos.reshape(-1, 1), axis=1)
    return float(np.mean(1.0 / ranks))


def evaluate_ogb_style(dataset_name: str, predictions: Any) -> dict[str, float]:
    """Evaluate predictions with this project's OGB-style manual metrics."""

    canonical_name = require_supported_dataset(dataset_name)
    pos_scores = _prediction_value(predictions, "pos_scores")

    if canonical_name == "ogbl_collab":
        neg_scores = _prediction_value(predictions, "neg_scores")
        return {"Hits@50": hits_at_k_from_scores(pos_scores, neg_scores, 50)}
    if canonical_name == "ogbl_ppa":
        neg_scores = _prediction_value(predictions, "neg_scores")
        return {"Hits@100": hits_at_k_from_scores(pos_scores, neg_scores, 100)}
    if canonical_name == "ogbl_citation2":
        neg_scores_matrix = _prediction_value(predictions, "neg_scores_matrix")
        return {"MRR": mrr_from_citation2_scores(pos_scores, neg_scores_matrix)}

    raise ValueError(f"Unsupported dataset for evaluation: {dataset_name}")


def _prediction_value(predictions: Any, key: str) -> Any:
    if isinstance(predictions, Mapping):
        return predictions[key]
    return getattr(predictions, key)
