"""Candidate-limited edge helpers for split-based link prediction evaluation."""

from __future__ import annotations

import random
from typing import Any, Hashable

import networkx as nx
import numpy as np

from .dataset_registry import get_dataset_config
from .ogb_split_loader import load_ogb_split, normalize_edge_array as _normalize_edges
from .schemas import CandidateEdges, EvalSplit, OGBSplitData


def get_eval_candidates(
    dataset_name: str,
    split: EvalSplit = "valid",
    raw_root: str = "data/raw",
) -> CandidateEdges:
    """Load candidate positive/negative edges for one official OGB split."""

    split_data = load_ogb_split(dataset_name, raw_root)
    return candidates_from_split(split_data, split)


def candidates_from_split(split_data: OGBSplitData, split: EvalSplit = "valid") -> CandidateEdges:
    """Create candidate edges from an already loaded OGB split object."""

    get_dataset_config(split_data.dataset_name)
    if split not in {"valid", "test"}:
        raise ValueError("split must be 'valid' or 'test'")

    if split == "valid":
        positive = split_data.valid_edges
        negative = split_data.valid_neg_edges
        source_nodes = split_data.valid_source_nodes
        target_nodes = split_data.valid_target_nodes
        target_node_neg = split_data.valid_target_node_neg
    else:
        positive = split_data.test_edges
        negative = split_data.test_neg_edges
        source_nodes = split_data.test_source_nodes
        target_nodes = split_data.test_target_nodes
        target_node_neg = split_data.test_target_node_neg

    notes = ""
    if target_node_neg is not None:
        notes = (
            "Target-negative matrix is preserved for citation-style ranking; "
            "it is not flattened into full non-edge enumeration."
        )

    return CandidateEdges(
        dataset_name=split_data.dataset_name,
        split=split,
        positive_edges=positive,
        negative_edges=negative,
        source_nodes=source_nodes,
        target_nodes=target_nodes,
        target_node_neg=target_node_neg,
        notes=notes,
    )


def sample_non_edges(
    graph: nx.Graph,
    num_samples: int,
    seed: int = 42,
    *,
    max_attempts_factor: int = 50,
) -> list[tuple[Hashable, Hashable]]:
    """Randomly sample candidate non-edges without enumerating all non-edges."""

    if num_samples < 0:
        raise ValueError("num_samples must be non-negative")
    if num_samples == 0:
        return []

    nodes = list(graph.nodes)
    if len(nodes) < 2:
        raise ValueError("graph must contain at least two nodes")

    rng = random.Random(seed)
    directed = graph.is_directed()
    samples: set[tuple[Hashable, Hashable]] = set()
    attempts = 0
    max_attempts = max(num_samples * max_attempts_factor, 100)

    while len(samples) < num_samples and attempts < max_attempts:
        attempts += 1
        source = rng.choice(nodes)
        target = rng.choice(nodes)
        if source == target or graph.has_edge(source, target):
            continue
        key = (source, target) if directed else _undirected_key(source, target)
        if key in samples:
            continue
        samples.add(key)

    if len(samples) < num_samples:
        raise RuntimeError(
            f"Sampled only {len(samples)} non-edges out of {num_samples}. "
            "Use fewer samples or a sparser graph."
        )
    return list(samples)


def edges_to_node_pairs(edges: Any, limit: int | None = None) -> list[tuple[int, int]]:
    """Convert an edge array-like object to Python node-pair tuples."""

    array = normalize_edge_array(edges)
    if limit is not None:
        array = array[:limit]
    return [(int(source), int(target)) for source, target in array]


def normalize_edge_array(edges: Any) -> np.ndarray:
    """Normalize edge arrays to [N, 2] int64 arrays."""

    return _normalize_edges(edges)


def _undirected_key(source: Hashable, target: Hashable) -> tuple[Hashable, Hashable]:
    left, right = sorted((source, target), key=lambda item: repr(item))
    return left, right
