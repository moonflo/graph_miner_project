"""File discovery and raw record loading for the preprocessing layer."""

from __future__ import annotations

import csv
import gzip
import io
import json
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

from .normalizer import infer_split_from_path, sanitize_dataset_name
from .schema import DatasetInput, RawRecord


SUPPORTED_SUFFIXES = (
    ".json",
    ".jsonl",
    ".csv",
    ".txt",
    ".json.gz",
    ".jsonl.gz",
    ".csv.gz",
    ".txt.gz",
)

SKIPPED_DIR_NAMES = {
    "__pycache__",
    ".git",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "processed",
}


WarningCallback = Callable[[str], None]


def discover_dataset(input_path: str | Path) -> DatasetInput:
    """Resolve one input file or dataset directory into a single dataset."""

    path = Path(input_path)
    if path.is_file():
        return DatasetInput(
            name=sanitize_dataset_name(path.stem),
            path=path.parent,
            files=(path,),
            is_ogb=False,
        )

    if not path.exists():
        raise FileNotFoundError(f"Input path does not exist: {path}")

    if is_ogb_dataset_dir(path) or _has_direct_supported_files(path):
        return _dataset_from_dir(path)

    child_datasets = [
        child
        for child in path.iterdir()
        if child.is_dir()
        and not _should_skip_dir(child)
        and (is_ogb_dataset_dir(child) or _has_direct_supported_files(child))
    ]
    if child_datasets:
        examples = ", ".join(str(child) for child in child_datasets[:3])
        raise ValueError(
            "Input points to a dataset parent directory. "
            f"Pass one dataset directory instead, for example: {examples}"
        )

    raw_child_datasets = _child_dataset_dirs(path / "raw")
    if raw_child_datasets:
        examples = ", ".join(str(child) for child in raw_child_datasets[:3])
        raise ValueError(
            "Input points to a data root. "
            f"Pass one dataset directory instead, for example: {examples}"
        )

    if any(_iter_supported_files(path)):
        return _dataset_from_dir(path)

    raise ValueError(
        "Input must point to one dataset file or one dataset directory, "
        "for example data/raw/ogbl_citation2."
    )


def discover_datasets(input_path: str | Path) -> list[DatasetInput]:
    """Backward-compatible wrapper returning exactly one resolved dataset."""

    return [discover_dataset(input_path)]


def iter_raw_records(
    dataset: DatasetInput,
    *,
    warning_callback: WarningCallback | None = None,
) -> Iterator[RawRecord]:
    """Yield raw records from JSON, JSONL, CSV, and TXT files."""

    for file_path in dataset.files:
        split = infer_split_from_path(file_path)
        try:
            yield from _iter_file_records(
                dataset.name,
                file_path,
                split,
                warning_callback=warning_callback,
            )
        except Exception as exc:
            if warning_callback is not None:
                warning_callback(f"Failed to read {file_path}: {exc}")


def is_ogb_dataset_dir(path: Path) -> bool:
    return (path / "raw" / "edge.csv.gz").is_file() and (
        (path / "raw" / "num-node-list.csv.gz").is_file()
        or any((path / "mapping").glob("nodeidx2*.csv.gz"))
    )


def _dataset_from_dir(path: Path) -> DatasetInput:
    return DatasetInput(
        name=sanitize_dataset_name(path.name),
        path=path,
        files=tuple(sorted(_iter_supported_files(path))),
        is_ogb=is_ogb_dataset_dir(path),
    )


def _has_direct_supported_files(path: Path) -> bool:
    for child in path.iterdir():
        if child.is_file() and _has_supported_suffix(child):
            return True
    return False


def _child_dataset_dirs(path: Path) -> list[Path]:
    if not path.is_dir():
        return []
    return [
        child
        for child in path.iterdir()
        if child.is_dir()
        and not _should_skip_dir(child)
        and (is_ogb_dataset_dir(child) or _has_direct_supported_files(child))
    ]


def _should_skip_dir(path: Path) -> bool:
    return path.name.startswith(".") or path.name in SKIPPED_DIR_NAMES


def _iter_supported_files(root: Path) -> Iterator[Path]:
    for path in root.rglob("*"):
        if path.is_dir():
            continue
        if any(part in SKIPPED_DIR_NAMES or part.startswith(".") for part in path.parts):
            continue
        if _has_supported_suffix(path):
            yield path


def _has_supported_suffix(path: Path) -> bool:
    name = path.name.lower()
    return any(name.endswith(suffix) for suffix in SUPPORTED_SUFFIXES)


def _iter_file_records(
    dataset_name: str,
    file_path: Path,
    split: str,
    *,
    warning_callback: WarningCallback | None = None,
) -> Iterator[RawRecord]:
    name = file_path.name.lower()
    text = _read_text_with_fallback(file_path)

    if name.endswith(".jsonl") or name.endswith(".jsonl.gz"):
        yield from _iter_jsonl_records(dataset_name, file_path, split, text, warning_callback)
        return

    if name.endswith(".json") or name.endswith(".json.gz"):
        yield from _iter_json_records(dataset_name, file_path, split, text)
        return

    if name.endswith(".csv") or name.endswith(".csv.gz"):
        yield from _iter_csv_records(dataset_name, file_path, split, text)
        return

    if name.endswith(".txt") or name.endswith(".txt.gz"):
        yield RawRecord(
            dataset_name=dataset_name,
            data={"text": text, "filename": file_path.name},
            source_file=file_path,
            index=0,
            split=split,
            record_type="txt",
        )


def _iter_jsonl_records(
    dataset_name: str,
    file_path: Path,
    split: str,
    text: str,
    warning_callback: WarningCallback | None,
) -> Iterator[RawRecord]:
    record_index = 0
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError as exc:
            if warning_callback is not None:
                warning_callback(
                    f"Invalid JSONL in {file_path} line {line_number}: {exc.msg}"
                )
            continue
        yield RawRecord(
            dataset_name=dataset_name,
            data=data,
            source_file=file_path,
            index=record_index,
            split=split,
            record_type="jsonl",
        )
        record_index += 1


def _iter_json_records(
    dataset_name: str,
    file_path: Path,
    split: str,
    text: str,
) -> Iterator[RawRecord]:
    payload = json.loads(text)
    records = _records_from_json_payload(payload)
    for index, data in enumerate(records):
        yield RawRecord(
            dataset_name=dataset_name,
            data=data,
            source_file=file_path,
            index=index,
            split=split,
            record_type="json",
        )


def _records_from_json_payload(payload: Any) -> list[Any]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("data", "records", "samples", "documents", "items"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
        return [payload]
    return [{"text": payload}]


def _iter_csv_records(
    dataset_name: str,
    file_path: Path,
    split: str,
    text: str,
) -> Iterator[RawRecord]:
    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames is None:
        return
    for index, row in enumerate(reader):
        yield RawRecord(
            dataset_name=dataset_name,
            data=dict(row),
            source_file=file_path,
            index=index,
            split=split,
            record_type="csv",
        )


def _read_text_with_fallback(path: Path) -> str:
    last_error: UnicodeDecodeError | None = None
    for encoding in ("utf-8-sig", "utf-8"):
        try:
            if path.name.lower().endswith(".gz"):
                with gzip.open(path, "rt", encoding=encoding) as file_obj:
                    return file_obj.read()
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    raise RuntimeError(f"Unable to read {path}")
