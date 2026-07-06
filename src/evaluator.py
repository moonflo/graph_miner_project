"""Evaluation helpers for link prediction outputs."""

from __future__ import annotations

from typing import Iterable


def hits_at_k(
    predictions: Iterable[tuple[str, str, float]],
    true_edges: Iterable[tuple[str, str]],
    k: int = 10,
) -> float:
    """Compute Hits@K for predicted links against ground truth edges."""

    normalized_truth = {_normalize_edge(edge) for edge in true_edges}
    top_edges = [_normalize_edge((u, v)) for u, v, _ in list(predictions)[:k]]
    if not top_edges:
        return 0.0

    hits = sum(edge in normalized_truth for edge in top_edges)
    return hits / len(top_edges)


def evaluate_hits(
    predictions: Iterable[tuple[str, str, float]],
    true_edges: Iterable[tuple[str, str]],
    k_values: tuple[int, ...] = (5, 10),
) -> dict[str, float]:
    """Evaluate link predictions with multiple Hits@K values."""

    prediction_list = list(predictions)
    truth_list = list(true_edges)
    return {f"hits@{k}": hits_at_k(prediction_list, truth_list, k) for k in k_values}


def _normalize_edge(edge: tuple[str, str]) -> tuple[str, str]:
    source, target = map(str, edge)
    return tuple(sorted((source, target)))
