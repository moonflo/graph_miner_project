"""Data loading helpers for text entities and OGB link prediction datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


SUPPORTED_OGB_LINK_DATASETS = {"ogbl-collab", "ogbl-ppa", "ogbl-citation2"}


def load_text_entities(path: str | Path, id_col: str = "id", text_col: str = "text") -> list[dict]:
    """Load custom entity text from CSV, JSON, or JSONL files."""

    path = Path(path)
    if path.suffix == ".csv":
        frame = pd.read_csv(path)
    elif path.suffix in {".json", ".jsonl"}:
        frame = pd.read_json(path, lines=path.suffix == ".jsonl")
    else:
        raise ValueError(f"Unsupported entity file format: {path.suffix}")

    missing_cols = {id_col, text_col} - set(frame.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {sorted(missing_cols)}")

    return [
        {"id": str(row[id_col]), "text": str(row[text_col])}
        for _, row in frame.iterrows()
    ]


def load_ogb_link_prediction_dataset(name: str, root: str | Path = "data/raw"):
    """Load an OGB link prediction dataset for evaluation-only workflows."""

    if name not in SUPPORTED_OGB_LINK_DATASETS:
        supported = ", ".join(sorted(SUPPORTED_OGB_LINK_DATASETS))
        raise ValueError(f"Unsupported OGB dataset '{name}'. Supported: {supported}")

    try:
        from ogb.linkproppred import LinkPropPredDataset
    except ImportError as exc:
        raise ImportError("Install the 'ogb' package to load OGB datasets.") from exc

    dataset = LinkPropPredDataset(name=name, root=str(root))
    graph = dataset[0]
    split_edge = dataset.get_edge_split()
    return graph, split_edge


def normalize_entities(entities: Iterable[dict]) -> list[dict]:
    """Normalize entity records into the expected {'id', 'text'} format."""

    normalized = []
    for index, entity in enumerate(entities):
        entity_id = str(entity.get("id", index))
        text = str(entity.get("text", entity_id))
        normalized.append({"id": entity_id, "text": text})
    return normalized
