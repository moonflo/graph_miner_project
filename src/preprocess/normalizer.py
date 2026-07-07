"""Normalization helpers for raw records and graph-ready identifiers."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any


TEXT_FIELDS = (
    "text",
    "content",
    "sentence",
    "document",
    "abstract",
    "body",
    "description",
    "question",
    "context",
    "passage",
    "paragraph",
)

TITLE_FIELDS = ("title", "name", "headline")
ID_FIELDS = ("id", "_id", "doc_id", "document_id", "uid", "guid", "original_id")
SPLIT_FIELDS = ("split", "partition", "subset", "set")


def normalize_whitespace(value: Any) -> str:
    """Convert a scalar or text-like value to a compact string."""

    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        value = " ".join(normalize_whitespace(item) for item in value)
    elif isinstance(value, dict):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True)
    else:
        value = str(value)
    return re.sub(r"\s+", " ", value).strip()


def normalize_entity_name(value: Any) -> str:
    return normalize_whitespace(value)


def normalize_split(value: Any) -> str:
    split = normalize_whitespace(value).lower()
    if split in {"train", "training"}:
        return "train"
    if split in {"dev", "valid", "validation", "val"}:
        return "dev"
    if split in {"test", "testing"}:
        return "test"
    return "unknown"


def infer_split_from_path(path: Path) -> str:
    for part in reversed(path.parts):
        split = normalize_split(part)
        if split != "unknown":
            return split
    return "unknown"


def sanitize_dataset_name(value: str) -> str:
    value = value.strip().replace("-", "_")
    value = re.sub(r"[^A-Za-z0-9_]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value or "dataset"


def stable_hash(*parts: Any, length: int = 16) -> str:
    payload = "\u241f".join(normalize_whitespace(part) for part in parts)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:length]


def stable_doc_id(dataset_name: str, split: str, index: int) -> str:
    dataset = sanitize_dataset_name(dataset_name)
    return f"{dataset}:{normalize_split(split)}:{index:08d}"


def stable_entity_id(dataset_name: str, name: str) -> str:
    dataset = sanitize_dataset_name(dataset_name)
    return f"{dataset}:entity:{stable_hash(name.lower())}"


def stable_relation_id(
    dataset_name: str,
    source_doc_id: str,
    index: int,
    head: str,
    relation_type: str,
    tail: str,
) -> str:
    dataset = sanitize_dataset_name(dataset_name)
    digest = stable_hash(source_doc_id, index, head.lower(), relation_type, tail.lower())
    return f"{dataset}:relation:{digest}"


def first_present(record: dict[str, Any], fields: tuple[str, ...]) -> tuple[str | None, Any]:
    lowered = {key.lower(): key for key in record}
    for field in fields:
        key = lowered.get(field.lower())
        if key is not None:
            return key, record[key]
    return None, None


def detect_text(record: Any) -> tuple[str | None, str]:
    if isinstance(record, str):
        return "text", normalize_whitespace(record)
    if not isinstance(record, dict):
        return None, normalize_whitespace(record)

    field, value = first_present(record, TEXT_FIELDS)
    if field is None:
        return None, ""
    return field, normalize_whitespace(value)


def detect_title(record: Any) -> str:
    if not isinstance(record, dict):
        return ""
    _, value = first_present(record, TITLE_FIELDS)
    return normalize_whitespace(value)


def detect_original_id(record: Any) -> str:
    if not isinstance(record, dict):
        return ""
    _, value = first_present(record, ID_FIELDS)
    return normalize_whitespace(value)


def detect_split(record: Any, fallback: str) -> str:
    if not isinstance(record, dict):
        return normalize_split(fallback)
    _, value = first_present(record, SPLIT_FIELDS)
    if value is None:
        return normalize_split(fallback)
    return normalize_split(value)


def field_names(record: Any) -> list[str]:
    if isinstance(record, dict):
        return sorted(str(key) for key in record)
    return []
