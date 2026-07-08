"""Read official OGB link prediction splits without downloading data."""

from __future__ import annotations

import gzip
from pathlib import Path
from typing import Any

import numpy as np

from .dataset_registry import get_dataset_config
from .schemas import OGBSplitData


def load_ogb_split(
    dataset_name: str,
    raw_root: str | Path = "data/raw",
) -> OGBSplitData:
    """Load the official split for a registered dataset from local cache only."""

    config = get_dataset_config(dataset_name)
    raw_dataset_dir = Path(raw_root) / config.raw_dir_name
    if not raw_dataset_dir.is_dir():
        raise FileNotFoundError(
            f"Raw directory for {config.canonical_name} not found: {raw_dataset_dir}"
        )

    try:
        return _load_manual_split(config.canonical_name, config.ogb_name, raw_dataset_dir)
    except Exception as manual_exc:  # noqa: BLE001
        processed_cache = raw_dataset_dir / "processed" / "data_processed"
        if not processed_cache.exists():
            raise

        try:
            fallback = _load_with_project_loader(config.canonical_name, config.ogb_name, raw_root)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"Manual split loading failed with {type(manual_exc).__name__}: "
                f"{manual_exc}. Project loader fallback failed with "
                f"{type(exc).__name__}: {exc}."
            ) from exc

        notes = (
            f"Manual split loading failed with {type(manual_exc).__name__}: "
            f"{manual_exc}. Fell back to utils.data_utils.load_dataset."
        )
        if fallback.notes:
            notes = f"{notes} {fallback.notes}"
        return _replace_notes(fallback, notes)


def load_train_edges(
    dataset_name: str,
    raw_root: str | Path = "data/raw",
) -> np.ndarray:
    return load_ogb_split(dataset_name, raw_root).train_edges


def load_valid_edges(
    dataset_name: str,
    raw_root: str | Path = "data/raw",
) -> np.ndarray:
    return load_ogb_split(dataset_name, raw_root).valid_edges


def load_test_edges(
    dataset_name: str,
    raw_root: str | Path = "data/raw",
) -> np.ndarray:
    return load_ogb_split(dataset_name, raw_root).test_edges


def _load_with_project_loader(
    canonical_name: str,
    ogb_name: str,
    raw_root: str | Path,
) -> OGBSplitData:
    from utils.data_utils import load_dataset

    data = load_dataset(
        ogb_name,
        root=raw_root,
        confirm_download=False,
        num_negative_samples=None,
    )
    return OGBSplitData(
        dataset_name=canonical_name,
        ogb_name=ogb_name,
        num_nodes=int(data["num_nodes"]),
        train_edges=normalize_edge_array(data["train_edges"]),
        valid_edges=normalize_edge_array(data["valid_edges"]),
        test_edges=normalize_edge_array(data["test_edges"]),
        train_edge_weight=_optional_float_vector(data.get("train_edge_weight")),
        train_edge_year=_optional_int_vector(data.get("train_edge_year")),
        valid_edge_weight=_optional_float_vector(data.get("valid_edge_weight")),
        valid_edge_year=_optional_int_vector(data.get("valid_edge_year")),
        test_edge_weight=_optional_float_vector(data.get("test_edge_weight")),
        test_edge_year=_optional_int_vector(data.get("test_edge_year")),
        valid_edge_neg=_optional_edge_neg_raw(
            data.get(
                "valid_edge_neg",
                data.get("valid_negative_edges_raw", data.get("valid_negative_edges")),
            )
        ),
        test_edge_neg=_optional_edge_neg_raw(
            data.get(
                "test_edge_neg",
                data.get("test_negative_edges_raw", data.get("test_negative_edges")),
            )
        ),
        valid_neg_edges=_optional_flat_edge_neg(
            data.get("valid_edge_neg", data.get("valid_negative_edges"))
        ),
        test_neg_edges=_optional_flat_edge_neg(
            data.get("test_edge_neg", data.get("test_negative_edges"))
        ),
        source="utils.data_utils.load_dataset",
        notes=(
            "Project loader fallback preserves split edge weight/year when the "
            "underlying OGB split exposes those attributes; otherwise they stay None."
        ),
    )


def _load_manual_split(
    canonical_name: str,
    ogb_name: str,
    raw_dataset_dir: Path,
) -> OGBSplitData:
    split_files = _find_split_files(raw_dataset_dir)
    num_nodes = _read_num_nodes(raw_dataset_dir)
    train = _load_torch_split(split_files["train"])
    valid = _load_torch_split(split_files["valid"])
    test = _load_torch_split(split_files["test"])

    valid_source_nodes = _optional_vector(valid.get("source_node"))
    valid_target_nodes = _optional_vector(valid.get("target_node"))
    test_source_nodes = _optional_vector(test.get("source_node"))
    test_target_nodes = _optional_vector(test.get("target_node"))
    valid_target_node_neg = _optional_matrix(valid.get("target_node_neg"))
    test_target_node_neg = _optional_matrix(test.get("target_node_neg"))
    valid_edge_neg = _optional_edge_neg_raw(valid.get("edge_neg"))
    test_edge_neg = _optional_edge_neg_raw(test.get("edge_neg"))

    return OGBSplitData(
        dataset_name=canonical_name,
        ogb_name=ogb_name,
        num_nodes=num_nodes,
        train_edges=_positive_edges_from_split(train),
        valid_edges=_positive_edges_from_split(valid),
        test_edges=_positive_edges_from_split(test),
        train_edge_weight=_optional_float_vector_from_keys(train, ("weight", "edge_weight")),
        train_edge_year=_optional_int_vector_from_keys(train, ("year", "edge_year")),
        valid_edge_weight=_optional_float_vector_from_keys(valid, ("weight", "edge_weight")),
        valid_edge_year=_optional_int_vector_from_keys(valid, ("year", "edge_year")),
        test_edge_weight=_optional_float_vector_from_keys(test, ("weight", "edge_weight")),
        test_edge_year=_optional_int_vector_from_keys(test, ("year", "edge_year")),
        valid_edge_neg=valid_edge_neg,
        test_edge_neg=test_edge_neg,
        valid_neg_edges=_flatten_edge_neg_array(valid_edge_neg),
        test_neg_edges=_flatten_edge_neg_array(test_edge_neg),
        valid_source_nodes=valid_source_nodes,
        valid_target_nodes=valid_target_nodes,
        test_source_nodes=test_source_nodes,
        test_target_nodes=test_target_nodes,
        valid_target_node_neg=valid_target_node_neg,
        test_target_node_neg=test_target_node_neg,
        source="cached raw files and split/*.pt",
        notes="Manual split read avoids creating OGB processed cache files or triggering downloads.",
    )


def _find_split_files(raw_dataset_dir: Path) -> dict[str, Path]:
    split_root = raw_dataset_dir / "split"
    split_files = {
        path.name.removesuffix(".pt"): path
        for path in split_root.glob("*/*.pt")
        if path.name in {"train.pt", "valid.pt", "test.pt"}
    }
    missing = {"train", "valid", "test"} - set(split_files)
    if missing:
        raise FileNotFoundError(
            f"Missing official split files under {split_root}: {sorted(missing)}"
        )
    return split_files


def _load_torch_split(path: Path) -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:
        raise ImportError("Install torch to read cached OGB split .pt files.") from exc

    payload = torch.load(path, weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"OGB split file did not contain a dict: {path}")
    return payload


def _read_num_nodes(raw_dataset_dir: Path) -> int:
    path = raw_dataset_dir / "raw" / "num-node-list.csv.gz"
    if not path.is_file():
        raise FileNotFoundError(f"Missing OGB num-node-list.csv.gz: {path}")
    with gzip.open(path, "rt", encoding="utf-8") as file_obj:
        first = file_obj.readline().strip().split(",")[0]
    try:
        return int(first)
    except ValueError as exc:
        raise ValueError(f"Could not parse num_nodes from {path}: {first!r}") from exc


def _positive_edges_from_split(split: dict[str, Any]) -> np.ndarray:
    if "edge" in split:
        return normalize_edge_array(split["edge"])
    if "source_node" in split and "target_node" in split:
        source = _as_numpy(split["source_node"]).reshape(-1)
        target = _as_numpy(split["target_node"]).reshape(-1)
        return np.column_stack((source, target)).astype(np.int64, copy=False)
    raise ValueError(f"Unsupported OGB split keys: {sorted(split)}")


def normalize_edge_array(value: Any) -> np.ndarray:
    """Normalize edge arrays to shape [N, 2] with int64 dtype."""

    array = _as_numpy(value)
    if array.size == 0:
        return np.empty((0, 2), dtype=np.int64)
    if array.ndim != 2:
        raise ValueError(f"Edges must be 2-dimensional, got shape {array.shape}")
    if array.shape[1] == 2:
        return array.astype(np.int64, copy=False)
    if array.shape[0] == 2:
        return array.T.astype(np.int64, copy=False)
    raise ValueError(f"Cannot normalize edge array with shape {array.shape}")


def _optional_edge_array(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    return normalize_edge_array(value)


def _optional_edge_neg_raw(value: Any) -> np.ndarray | None:
    """Preserve an OGB edge_neg array as 2D shared-pool or 3D row-wise data."""

    if value is None:
        return None
    array = _as_numpy(value)
    if array.ndim == 2:
        if array.size == 0:
            return np.empty((0, 2), dtype=np.int64)
        return normalize_edge_array(array)
    if array.ndim == 3 and array.shape[-1] == 2:
        return array.astype(np.int64, copy=False)
    raise ValueError(
        "edge_neg must have shape [num_neg, 2] or [num_pos, num_neg, 2], "
        f"got {array.shape}"
    )


def _optional_flat_edge_neg(value: Any) -> np.ndarray | None:
    return _flatten_edge_neg_array(_optional_edge_neg_raw(value))


def _flatten_edge_neg_array(edge_neg: np.ndarray | None) -> np.ndarray | None:
    if edge_neg is None:
        return None
    if edge_neg.ndim == 2:
        return normalize_edge_array(edge_neg)
    if edge_neg.ndim == 3 and edge_neg.shape[-1] == 2:
        return edge_neg.reshape(-1, 2).astype(np.int64, copy=False)
    raise ValueError(
        "edge_neg must have shape [num_neg, 2] or [num_pos, num_neg, 2], "
        f"got {edge_neg.shape}"
    )


def _optional_vector(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    return _as_numpy(value).reshape(-1).astype(np.int64, copy=False)


def _optional_float_vector(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    return _as_numpy(value).reshape(-1).astype(float, copy=False)


def _optional_int_vector(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    return _as_numpy(value).reshape(-1).astype(np.int64, copy=False)


def _optional_float_vector_from_keys(
    split: dict[str, Any],
    keys: tuple[str, ...],
) -> np.ndarray | None:
    for key in keys:
        if key in split and split[key] is not None:
            return _optional_float_vector(split[key])
    return None


def _optional_int_vector_from_keys(
    split: dict[str, Any],
    keys: tuple[str, ...],
) -> np.ndarray | None:
    for key in keys:
        if key in split and split[key] is not None:
            return _optional_int_vector(split[key])
    return None


def _optional_matrix(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    array = _as_numpy(value)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D negative target matrix, got {array.shape}")
    return array.astype(np.int64, copy=False)


def _as_numpy(value: Any) -> np.ndarray:
    if value is None:
        return np.empty((0, 2), dtype=np.int64)
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _replace_notes(split: OGBSplitData, notes: str) -> OGBSplitData:
    return OGBSplitData(
        dataset_name=split.dataset_name,
        ogb_name=split.ogb_name,
        num_nodes=split.num_nodes,
        train_edges=split.train_edges,
        valid_edges=split.valid_edges,
        test_edges=split.test_edges,
        train_edge_weight=split.train_edge_weight,
        train_edge_year=split.train_edge_year,
        valid_edge_weight=split.valid_edge_weight,
        valid_edge_year=split.valid_edge_year,
        test_edge_weight=split.test_edge_weight,
        test_edge_year=split.test_edge_year,
        valid_edge_neg=split.valid_edge_neg,
        test_edge_neg=split.test_edge_neg,
        valid_neg_edges=split.valid_neg_edges,
        test_neg_edges=split.test_neg_edges,
        valid_source_nodes=split.valid_source_nodes,
        valid_target_nodes=split.valid_target_nodes,
        test_source_nodes=split.test_source_nodes,
        test_target_nodes=split.test_target_nodes,
        valid_target_node_neg=split.valid_target_node_neg,
        test_target_node_neg=split.test_target_node_neg,
        source=split.source,
        notes=notes,
    )
