"""Streaming readers for processed graph JSONL files."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from .dataset_registry import get_dataset_config
from .schemas import GraphEdge, GraphNode


def resolve_processed_dataset_dir(
    dataset_name: str,
    processed_root: str | Path = "data/processed",
) -> Path:
    """Resolve a processed dataset directory through the manual registry."""

    config = get_dataset_config(dataset_name)
    path = Path(processed_root) / config.processed_dir_name
    if not path.is_dir():
        raise FileNotFoundError(
            f"Processed directory for {config.canonical_name} not found: {path}"
        )
    return path


def load_graph_nodes(
    dataset_name: str,
    processed_root: str | Path = "data/processed",
    limit: int | None = None,
) -> Iterator[GraphNode]:
    """Yield graph nodes from graph_nodes.jsonl without loading the full file."""

    dataset_dir = resolve_processed_dataset_dir(dataset_name, processed_root)
    yield from _iter_graph_nodes(dataset_dir / "graph_nodes.jsonl", limit=limit)


def load_graph_edges(
    dataset_name: str,
    processed_root: str | Path = "data/processed",
    limit: int | None = None,
) -> Iterator[GraphEdge]:
    """Yield graph edges from graph_edges.jsonl without loading the full file."""

    dataset_dir = resolve_processed_dataset_dir(dataset_name, processed_root)
    yield from _iter_graph_edges(dataset_dir / "graph_edges.jsonl", limit=limit)


def load_stats(
    dataset_name: str,
    processed_root: str | Path = "data/processed",
) -> dict[str, Any]:
    """Read stats.json for a registered processed dataset."""

    dataset_dir = resolve_processed_dataset_dir(dataset_name, processed_root)
    path = dataset_dir / "stats.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing stats.json: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"stats.json must contain a JSON object: {path}")
    return payload


def iter_jsonl(
    path: str | Path,
    *,
    limit: int | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield JSON objects from a JSONL file with line-aware errors."""

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing JSONL file: {path}")

    emitted = 0
    with path.open("r", encoding="utf-8") as file_obj:
        for line_number, line in enumerate(file_obj, start=1):
            if limit is not None and emitted >= limit:
                break
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} line {line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected JSON object in {path} line {line_number}")
            emitted += 1
            yield row


def _iter_graph_nodes(path: Path, *, limit: int | None) -> Iterator[GraphNode]:
    for row in iter_jsonl(path, limit=limit):
        missing = {"id", "label", "type", "weight", "metadata"} - set(row)
        if missing:
            raise ValueError(f"Missing graph node fields in {path}: {sorted(missing)}")
        metadata = row["metadata"]
        if not isinstance(metadata, dict):
            raise ValueError(f"graph_nodes metadata must be an object in {path}")
        yield GraphNode(
            id=str(row["id"]),
            label=str(row["label"]),
            type=str(row["type"]),
            weight=float(row["weight"]),
            metadata=metadata,
        )


def _iter_graph_edges(path: Path, *, limit: int | None) -> Iterator[GraphEdge]:
    for row in iter_jsonl(path, limit=limit):
        missing = {"source", "target", "relation", "weight"} - set(row)
        if missing:
            raise ValueError(f"Missing graph edge fields in {path}: {sorted(missing)}")
        metadata = row.get("metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError(f"graph_edges metadata must be an object in {path}")
        evidence_doc_ids = row.get("evidence_doc_ids", [])
        if evidence_doc_ids is None:
            evidence_doc_ids = []
        yield GraphEdge(
            source=str(row["source"]),
            target=str(row["target"]),
            relation=str(row["relation"]),
            weight=float(row["weight"]),
            metadata=metadata,
            evidence_doc_ids=tuple(str(item) for item in evidence_doc_ids),
        )
