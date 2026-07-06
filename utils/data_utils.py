"""Dataset utilities for OGB link prediction and graph preprocessing.

OGB datasets are used here for evaluation-only workflows. This module keeps
all returned graph data in numpy arrays and intentionally avoids PyTorch,
torch_geometric, and GNN training pipeline dependencies.
"""

from __future__ import annotations

import builtins
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np


SUPPORTED_OGB_LINK_DATASETS = {"ogbl-collab", "ogbl-ppa", "ogbl-citation2"}


def load_dataset(
    name: str,
    root: str | Path = "data/raw",
    *,
    mask_ratio: float = 0.0,
    num_negative_samples: int | None = None,
    seed: int = 42,
    confirm_download: bool = True,
) -> dict[str, Any]:
    """Load an OGB link prediction dataset in a standard numpy format.

    Parameters
    ----------
    name:
        OGB link prediction dataset name. Supported values are
        ``ogbl-collab``, ``ogbl-ppa``, and ``ogbl-citation2``.
    root:
        Directory used by OGB for automatic dataset download/cache.
    mask_ratio:
        Optional ratio of training edges to hide for latent-relation
        simulation. Hidden edges are returned as ``masked_edges``.
    num_negative_samples:
        Optional number of graph-wide negative samples to generate. Validation
        and test negative edges supplied by OGB are returned separately when
        available.
    seed:
        Random seed used for edge masking and generated negative sampling.
    confirm_download:
        Automatically answer yes to OGB's large-download confirmation prompt.

    Returns
    -------
    dict
        A dictionary containing ``edge_index`` with shape ``[2, E]``,
        ``node_features`` when available, ``num_nodes``, and standard
        train/valid/test edge splits with shape ``[N, 2]``.
    """

    graph, split_edge = load_ogb_link_prediction_dataset(
        name,
        root=root,
        confirm_download=confirm_download,
    )

    edge_index = _as_edge_index(graph.get("edge_index"))
    node_features = _as_numpy(graph.get("node_feat"))
    num_nodes = _infer_num_nodes(graph, edge_index, node_features)

    train_edges = _positive_edges_from_split(split_edge["train"])
    valid_edges = _positive_edges_from_split(split_edge["valid"])
    test_edges = _positive_edges_from_split(split_edge["test"])

    visible_train_edges = train_edges
    masked_edges = np.empty((0, 2), dtype=np.int64)
    if mask_ratio > 0:
        visible_train_edges, masked_edges = edge_masking(
            train_edges,
            mask_ratio=mask_ratio,
            seed=seed,
        )

    data: dict[str, Any] = {
        "name": name,
        "edge_index": edge_index,
        "node_features": node_features,
        "num_nodes": num_nodes,
        "train_edges": visible_train_edges,
        "valid_edges": valid_edges,
        "test_edges": test_edges,
        "masked_edges": masked_edges,
        "valid_negative_edges": _negative_edges_from_split(split_edge["valid"]),
        "test_negative_edges": _negative_edges_from_split(split_edge["test"]),
    }

    if num_negative_samples is not None and num_negative_samples > 0:
        data["negative_edges"] = negative_sampling(
            edge_index=edge_index,
            num_nodes=num_nodes,
            num_samples=num_negative_samples,
            seed=seed,
            undirected=name != "ogbl-citation2",
        )

    return data


def load_ogb_link_prediction_dataset(
    name: str,
    root: str | Path = "data/raw",
    *,
    confirm_download: bool = True,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Download and load a supported OGB link prediction dataset."""

    if name not in SUPPORTED_OGB_LINK_DATASETS:
        supported = ", ".join(sorted(SUPPORTED_OGB_LINK_DATASETS))
        raise ValueError(f"Unsupported OGB dataset '{name}'. Supported: {supported}")

    try:
        from ogb.linkproppred import LinkPropPredDataset
    except ImportError as exc:
        raise ImportError("Install the 'ogb' package to load OGB datasets.") from exc

    with _ogb_download_confirm(confirm_download), _ogb_torch_load_compat():
        dataset = LinkPropPredDataset(name=name, root=str(root))
        graph = dataset[0]
        split_edge = dataset.get_edge_split()
    return graph, split_edge


def edge_masking(
    edges: np.ndarray,
    *,
    mask_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Hide a ratio of positive edges for latent-relation simulation.

    ``edges`` may be either ``[N, 2]`` edge pairs or ``[2, E]`` edge_index.
    The returned arrays are always edge pairs with shape ``[N, 2]``.
    """

    if not 0 <= mask_ratio < 1:
        raise ValueError("mask_ratio must be in the range [0, 1)")

    edge_pairs = _as_edge_pairs(edges)
    if edge_pairs.size == 0 or mask_ratio == 0:
        return edge_pairs, np.empty((0, 2), dtype=np.int64)

    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(edge_pairs))
    mask_count = int(round(len(edge_pairs) * mask_ratio))

    masked_indices = indices[:mask_count]
    kept_indices = indices[mask_count:]
    return edge_pairs[kept_indices], edge_pairs[masked_indices]


def negative_sampling(
    edge_index: np.ndarray,
    num_nodes: int,
    num_samples: int,
    *,
    exclude_edges: np.ndarray | None = None,
    seed: int = 42,
    undirected: bool = True,
    max_attempts_factor: int = 20,
) -> np.ndarray:
    """Sample node pairs that do not exist in the positive graph."""

    if num_nodes <= 1:
        raise ValueError("num_nodes must be greater than 1")
    if num_samples < 0:
        raise ValueError("num_samples must be non-negative")
    if num_samples == 0:
        return np.empty((0, 2), dtype=np.int64)

    positives = set()
    for source, target in _as_edge_pairs(edge_index):
        _add_edge_key(positives, int(source), int(target), undirected=undirected)
    if exclude_edges is not None:
        for source, target in _as_edge_pairs(exclude_edges):
            _add_edge_key(positives, int(source), int(target), undirected=undirected)

    rng = np.random.default_rng(seed)
    samples: set[tuple[int, int]] = set()
    max_attempts = max(num_samples * max_attempts_factor, 100)
    attempts = 0

    while len(samples) < num_samples and attempts < max_attempts:
        attempts += 1
        source = int(rng.integers(0, num_nodes))
        target = int(rng.integers(0, num_nodes))
        if source == target:
            continue

        key = _edge_key(source, target, undirected=undirected)
        if key in positives or key in samples:
            continue
        samples.add(key)

    if len(samples) < num_samples:
        raise RuntimeError(
            f"Only sampled {len(samples)} negative edges out of {num_samples}. "
            "Try lowering num_samples or using a denser sampling strategy."
        )

    return np.asarray(sorted(samples), dtype=np.int64)


def _as_numpy(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _as_edge_index(value: Any) -> np.ndarray:
    array = _as_numpy(value)
    if array is None:
        raise ValueError("Graph is missing edge_index")

    array = np.asarray(array, dtype=np.int64)
    if array.ndim != 2:
        raise ValueError(f"edge_index must be 2-dimensional, got shape {array.shape}")
    if array.shape[0] == 2:
        return array
    if array.shape[1] == 2:
        return array.T
    raise ValueError(f"Cannot interpret edge_index shape {array.shape}")


def _as_edge_pairs(value: Any) -> np.ndarray:
    array = _as_numpy(value)
    if array is None:
        return np.empty((0, 2), dtype=np.int64)

    array = np.asarray(array, dtype=np.int64)
    if array.size == 0:
        return np.empty((0, 2), dtype=np.int64)
    if array.ndim != 2:
        raise ValueError(f"Edges must be 2-dimensional, got shape {array.shape}")
    if array.shape[1] == 2:
        return array
    if array.shape[0] == 2:
        return array.T
    raise ValueError(f"Cannot interpret edge shape {array.shape}")


def _positive_edges_from_split(split: dict[str, Any]) -> np.ndarray:
    if "edge" in split:
        return _as_edge_pairs(split["edge"])
    if "source_node" in split and "target_node" in split:
        source = np.asarray(_as_numpy(split["source_node"]), dtype=np.int64).reshape(-1)
        target = np.asarray(_as_numpy(split["target_node"]), dtype=np.int64).reshape(-1)
        return np.column_stack((source, target))
    raise ValueError(f"Unsupported OGB split keys: {sorted(split)}")


def _negative_edges_from_split(split: dict[str, Any]) -> np.ndarray | None:
    if "edge_neg" in split:
        return _as_edge_pairs(split["edge_neg"])
    if "source_node" not in split or "target_node_neg" not in split:
        return None

    source = np.asarray(_as_numpy(split["source_node"]), dtype=np.int64).reshape(-1)
    target_neg = np.asarray(_as_numpy(split["target_node_neg"]), dtype=np.int64)

    if target_neg.ndim == 1:
        return np.column_stack((source, target_neg))

    repeated_source = np.repeat(source, target_neg.shape[1])
    return np.column_stack((repeated_source, target_neg.reshape(-1)))


def _infer_num_nodes(
    graph: dict[str, Any],
    edge_index: np.ndarray,
    node_features: np.ndarray | None,
) -> int:
    if graph.get("num_nodes") is not None:
        return int(graph["num_nodes"])
    if node_features is not None:
        return int(node_features.shape[0])
    if edge_index.size == 0:
        return 0
    return int(edge_index.max()) + 1


def _edge_key(source: int, target: int, *, undirected: bool) -> tuple[int, int]:
    if undirected and source > target:
        return target, source
    return source, target


def _add_edge_key(
    edges: set[tuple[int, int]],
    source: int,
    target: int,
    *,
    undirected: bool,
) -> None:
    if source == target:
        return
    edges.add(_edge_key(source, target, undirected=undirected))


@contextmanager
def _ogb_torch_load_compat():
    """Handle OGB 1.3.x split files with PyTorch 2.6+ defaults.

    OGB internally uses ``torch.load`` for cached split files. Newer PyTorch
    releases default ``weights_only`` to True, which rejects older OGB numpy
    payloads. This scoped patch is limited to the OGB dataset read path.
    """

    try:
        import torch
    except ImportError:
        yield
        return

    original_load = torch.load

    def patched_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_load(*args, **kwargs)

    torch.load = patched_load
    try:
        yield
    finally:
        torch.load = original_load


@contextmanager
def _ogb_download_confirm(confirm_download: bool):
    if not confirm_download:
        yield
        return

    original_input = builtins.input

    def confirmed_input(prompt: str = "") -> str:
        if "Will you proceed?" in prompt:
            print(prompt, end="")
            print("y")
            return "y"
        return original_input(prompt)

    builtins.input = confirmed_input
    try:
        yield
    finally:
        builtins.input = original_input
