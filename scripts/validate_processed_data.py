"""Validate prepared OGB processed data without mutating the data directory.

The script performs streaming checks over JSONL files, reads cached OGB split
metadata, and runs tiny NetworkX smoke tests. It writes only the markdown report
requested by the user.
"""

from __future__ import annotations

import argparse
import gc
import gzip
import json
import math
import numbers
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import networkx as nx


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_ROOT = PROJECT_ROOT / "data"
RAW_ROOT = DATA_ROOT / "raw"
PROCESSED_ROOT = DATA_ROOT / "processed"
REPORT_PATH = PROJECT_ROOT / "reports" / "data_validation_report.md"

EXPECTED_FILES = (
    "documents.jsonl",
    "entities.jsonl",
    "relations.jsonl",
    "triples.jsonl",
    "graph_nodes.jsonl",
    "graph_edges.jsonl",
    "stats.json",
)

JSONL_FILES = tuple(name for name in EXPECTED_FILES if name.endswith(".jsonl"))
NODE_FIELDS = ("id", "label", "type", "weight", "metadata")
EDGE_FIELDS = ("source", "target", "relation", "weight")
COUNT_KEYS = {
    "documents.jsonl": "num_documents",
    "entities.jsonl": "num_entities",
    "relations.jsonl": "num_relations",
    "triples.jsonl": "num_triples",
    "graph_nodes.jsonl": "num_graph_nodes",
    "graph_edges.jsonl": "num_graph_edges",
}
UNDIRECTED_CANONICAL = {"ogbl-collab", "ogbl-ppa"}


@dataclass
class FileCheck:
    name: str
    exists: bool
    size_bytes: int = 0
    line_count: int = 0
    valid_json: bool = True
    empty: bool = False
    errors: list[str] = field(default_factory=list)


@dataclass
class NodeCheck:
    duplicate_ids: int = 0
    duplicate_id_samples: list[str] = field(default_factory=list)
    missing_fields: Counter[str] = field(default_factory=Counter)
    missing_metadata_node_idx: int = 0
    invalid_metadata_node_idx: int = 0
    id_metadata_mismatch: int = 0
    id_format_counts: Counter[str] = field(default_factory=Counter)
    id_format_samples: dict[str, str] = field(default_factory=dict)


@dataclass
class EdgeCheck:
    missing_fields: Counter[str] = field(default_factory=Counter)
    missing_endpoints: int = 0
    missing_endpoint_samples: list[dict[str, str]] = field(default_factory=list)
    source_format_counts: Counter[str] = field(default_factory=Counter)
    target_format_counts: Counter[str] = field(default_factory=Counter)
    id_format_samples: dict[str, str] = field(default_factory=dict)
    self_loops: int = 0
    duplicate_edges: int = 0
    duplicate_edge_samples: list[dict[str, str]] = field(default_factory=list)
    reverse_edges_seen: int = 0
    invalid_weights: int = 0
    relation_values: Counter[str] = field(default_factory=Counter)


@dataclass
class SmokeResult:
    sample_nodes_read: int = 0
    sample_edges_read: int = 0
    graph_nodes: int = 0
    graph_edges: int = 0
    component_kind: str = "connected_components"
    components: int = 0
    average_degree: float = 0.0
    max_degree: int = 0
    algorithm_status: dict[str, str] = field(default_factory=dict)
    note: str = ""


@dataclass
class RawSplitResult:
    canonical_name: str
    raw_dir: str
    source: str = ""
    status: str = ""
    num_nodes: int | None = None
    edge_index_shape: tuple[int, ...] | None = None
    train_edges_shape: tuple[int, ...] | None = None
    valid_edges_shape: tuple[int, ...] | None = None
    test_edges_shape: tuple[int, ...] | None = None
    valid_negative_edges_shape: tuple[int, ...] | None = None
    test_negative_edges_shape: tuple[int, ...] | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class DatasetReport:
    name: str
    canonical_name: str
    path: Path
    file_checks: dict[str, FileCheck] = field(default_factory=dict)
    stats: dict[str, Any] = field(default_factory=dict)
    stats_valid: bool = False
    stats_errors: list[str] = field(default_factory=list)
    node_check: NodeCheck = field(default_factory=NodeCheck)
    edge_check: EdgeCheck = field(default_factory=EdgeCheck)
    count_mismatches: list[str] = field(default_factory=list)
    smoke: SmokeResult | None = None
    raw_split: RawSplitResult | None = None
    processed_from_full_graph_likely: bool | None = None
    leakage_notes: list[str] = field(default_factory=list)


def human_size(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.2f} {unit}"
        value /= 1024
    return f"{size} B"


def canonical_dataset_name(name: str) -> str:
    if name.startswith("ogbl_"):
        return "ogbl-" + name[len("ogbl_") :]
    return name


def parse_node_ref(value: Any) -> tuple[str, int | None]:
    if not isinstance(value, str):
        return "non_string", None
    if "::node::" in value:
        suffix = value.rsplit("::node::", 1)[1]
        try:
            return "double_colon_dataset::node::idx", int(suffix)
        except ValueError:
            return "double_colon_dataset::node::non_int", None
    if ":node:" in value:
        suffix = value.rsplit(":node:", 1)[1]
        try:
            return "single_colon_dataset:node:idx", int(suffix)
        except ValueError:
            return "single_colon_dataset:node:non_int", None
    return "other", None


def is_number(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, numbers.Number):
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


def read_json_line(line: bytes, path: Path, line_number: int, check: FileCheck) -> Any | None:
    try:
        return json.loads(line)
    except json.JSONDecodeError as exc:
        check.valid_json = False
        if len(check.errors) < 10:
            check.errors.append(f"line {line_number}: {exc.msg}")
        return None


def progress(dataset: str, filename: str, count: int, started_at: float) -> None:
    if count and count % 5_000_000 == 0:
        elapsed = time.time() - started_at
        print(
            f"[{dataset}] {filename}: parsed {count:,} lines in {elapsed:.1f}s",
            file=sys.stderr,
            flush=True,
        )


def validate_generic_jsonl(path: Path, dataset: str) -> FileCheck:
    check = FileCheck(name=path.name, exists=path.exists())
    if not check.exists:
        check.valid_json = False
        check.errors.append("missing file")
        return check

    check.size_bytes = path.stat().st_size
    check.empty = check.size_bytes == 0
    started_at = time.time()
    with path.open("rb") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                check.valid_json = False
                if len(check.errors) < 10:
                    check.errors.append(f"line {line_number}: empty line")
                continue
            read_json_line(line, path, line_number, check)
            check.line_count += 1
            progress(dataset, path.name, check.line_count, started_at)
    return check


def validate_nodes(path: Path, dataset: str) -> tuple[FileCheck, NodeCheck, set[str], int]:
    file_check = FileCheck(name=path.name, exists=path.exists())
    node_check = NodeCheck()
    node_ids: set[str] = set()
    max_node_idx = -1

    if not file_check.exists:
        file_check.valid_json = False
        file_check.errors.append("missing file")
        return file_check, node_check, node_ids, max_node_idx

    file_check.size_bytes = path.stat().st_size
    file_check.empty = file_check.size_bytes == 0
    started_at = time.time()

    with path.open("rb") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                file_check.valid_json = False
                if len(file_check.errors) < 10:
                    file_check.errors.append(f"line {line_number}: empty line")
                continue

            obj = read_json_line(line, path, line_number, file_check)
            file_check.line_count += 1
            progress(dataset, path.name, file_check.line_count, started_at)
            if not isinstance(obj, dict):
                file_check.valid_json = False
                if len(file_check.errors) < 10:
                    file_check.errors.append(f"line {line_number}: expected JSON object")
                continue

            for field_name in NODE_FIELDS:
                if field_name not in obj:
                    node_check.missing_fields[field_name] += 1

            node_id = obj.get("id")
            if isinstance(node_id, str):
                if node_id in node_ids:
                    node_check.duplicate_ids += 1
                    if len(node_check.duplicate_id_samples) < 5:
                        node_check.duplicate_id_samples.append(node_id)
                else:
                    node_ids.add(node_id)

            format_name, node_idx_from_id = parse_node_ref(node_id)
            node_check.id_format_counts[format_name] += 1
            node_check.id_format_samples.setdefault(format_name, str(node_id))

            metadata = obj.get("metadata")
            if not isinstance(metadata, dict) or "node_idx" not in metadata:
                node_check.missing_metadata_node_idx += 1
                continue

            try:
                metadata_node_idx = int(metadata["node_idx"])
            except (TypeError, ValueError):
                node_check.invalid_metadata_node_idx += 1
                continue

            max_node_idx = max(max_node_idx, metadata_node_idx)
            if node_idx_from_id is not None and node_idx_from_id != metadata_node_idx:
                node_check.id_metadata_mismatch += 1

    return file_check, node_check, node_ids, max_node_idx


def validate_edges(
    path: Path,
    dataset: str,
    node_ids: set[str],
    num_nodes_for_key: int,
) -> tuple[FileCheck, EdgeCheck]:
    file_check = FileCheck(name=path.name, exists=path.exists())
    edge_check = EdgeCheck()
    seen_edge_keys: set[int] = set()
    relation_ids: dict[str, int] = {}

    if not file_check.exists:
        file_check.valid_json = False
        file_check.errors.append("missing file")
        return file_check, edge_check

    file_check.size_bytes = path.stat().st_size
    file_check.empty = file_check.size_bytes == 0
    started_at = time.time()

    with path.open("rb") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                file_check.valid_json = False
                if len(file_check.errors) < 10:
                    file_check.errors.append(f"line {line_number}: empty line")
                continue

            obj = read_json_line(line, path, line_number, file_check)
            file_check.line_count += 1
            progress(dataset, path.name, file_check.line_count, started_at)
            if not isinstance(obj, dict):
                file_check.valid_json = False
                if len(file_check.errors) < 10:
                    file_check.errors.append(f"line {line_number}: expected JSON object")
                continue

            for field_name in EDGE_FIELDS:
                if field_name not in obj:
                    edge_check.missing_fields[field_name] += 1

            source = obj.get("source")
            target = obj.get("target")
            relation = str(obj.get("relation"))
            weight = obj.get("weight")
            edge_check.relation_values[relation] += 1

            source_format, source_idx = parse_node_ref(source)
            target_format, target_idx = parse_node_ref(target)
            edge_check.source_format_counts[source_format] += 1
            edge_check.target_format_counts[target_format] += 1
            edge_check.id_format_samples.setdefault(f"source:{source_format}", str(source))
            edge_check.id_format_samples.setdefault(f"target:{target_format}", str(target))

            if source not in node_ids or target not in node_ids:
                edge_check.missing_endpoints += 1
                if len(edge_check.missing_endpoint_samples) < 5:
                    edge_check.missing_endpoint_samples.append(
                        {"source": str(source), "target": str(target)}
                    )

            if source == target:
                edge_check.self_loops += 1

            if not is_number(weight):
                edge_check.invalid_weights += 1

            relation_id = relation_ids.setdefault(relation, len(relation_ids))
            if source_idx is not None and target_idx is not None:
                directed_pair = source_idx * num_nodes_for_key + target_idx
                reverse_pair = target_idx * num_nodes_for_key + source_idx
                edge_key = (directed_pair << 20) | relation_id
                reverse_key = (reverse_pair << 20) | relation_id
            else:
                edge_key = hash((source, target, relation_id))
                reverse_key = hash((target, source, relation_id))

            if source != target and reverse_key in seen_edge_keys:
                edge_check.reverse_edges_seen += 1

            if edge_key in seen_edge_keys:
                edge_check.duplicate_edges += 1
                if len(edge_check.duplicate_edge_samples) < 5:
                    edge_check.duplicate_edge_samples.append(
                        {
                            "source": str(source),
                            "target": str(target),
                            "relation": relation,
                        }
                    )
            else:
                seen_edge_keys.add(edge_key)

    return file_check, edge_check


def load_stats(path: Path) -> tuple[dict[str, Any], bool, list[str]]:
    if not path.exists():
        return {}, False, ["missing file"]
    if path.stat().st_size == 0:
        return {}, False, ["empty file"]
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except json.JSONDecodeError as exc:
        return {}, False, [f"invalid JSON: {exc.msg}"]
    if not isinstance(data, dict):
        return {}, False, ["stats.json is not an object"]
    return data, True, []


def compare_counts(report: DatasetReport) -> None:
    for filename, stats_key in COUNT_KEYS.items():
        file_check = report.file_checks.get(filename)
        if not file_check or not report.stats_valid:
            continue
        expected = report.stats.get(stats_key)
        if expected is None:
            report.count_mismatches.append(f"{stats_key} missing from stats.json")
            continue
        try:
            expected_int = int(expected)
        except (TypeError, ValueError):
            report.count_mismatches.append(f"{stats_key} is not an integer: {expected!r}")
            continue
        if file_check.line_count != expected_int:
            report.count_mismatches.append(
                f"{filename}: lines={file_check.line_count:,}, stats.{stats_key}={expected_int:,}"
            )


def discover_processed_datasets() -> list[Path]:
    if not PROCESSED_ROOT.exists():
        return []
    return sorted(
        path
        for path in PROCESSED_ROOT.iterdir()
        if path.is_dir() and path.name.startswith("ogbl")
    )


def read_gzip_scalar(path: Path) -> int | None:
    if not path.exists():
        return None
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        first = handle.readline().strip().split(",")[0]
    try:
        return int(first)
    except ValueError:
        return None


def array_shape(value: Any) -> tuple[int, ...] | None:
    if value is None:
        return None
    shape = getattr(value, "shape", None)
    if shape is None:
        try:
            return (len(value),)
        except TypeError:
            return None
    return tuple(int(item) for item in shape)


def positive_edge_shape(split: dict[str, Any]) -> tuple[int, ...] | None:
    if "edge" in split:
        return array_shape(split["edge"])
    if "source_node" in split and "target_node" in split:
        source_shape = array_shape(split["source_node"])
        if source_shape:
            return (source_shape[0], 2)
    return None


def negative_edge_shape(split: dict[str, Any]) -> tuple[int, ...] | None:
    if "edge_neg" in split:
        return array_shape(split["edge_neg"])
    if "source_node" in split and "target_node_neg" in split:
        source_shape = array_shape(split["source_node"])
        target_shape = array_shape(split["target_node_neg"])
        if not source_shape or not target_shape:
            return None
        if len(target_shape) == 1:
            return (source_shape[0], 2)
        return (source_shape[0] * target_shape[1], 2)
    return None


def load_project_dataset_split(canonical_name: str) -> RawSplitResult:
    result = RawSplitResult(
        canonical_name=canonical_name,
        raw_dir=str(RAW_ROOT / canonical_name.replace("-", "_")),
    )
    try:
        from utils.data_utils import load_dataset

        data = load_dataset(
            canonical_name,
            root=RAW_ROOT,
            confirm_download=False,
            num_negative_samples=None,
        )
    except Exception as exc:  # noqa: BLE001 - this is a report, not a library API.
        result.status = f"project loader failed: {type(exc).__name__}: {exc}"
        return result

    result.source = "utils.data_utils.load_dataset"
    result.status = "ok"
    result.num_nodes = int(data["num_nodes"])
    result.edge_index_shape = tuple(int(item) for item in data["edge_index"].shape)
    result.train_edges_shape = tuple(int(item) for item in data["train_edges"].shape)
    result.valid_edges_shape = tuple(int(item) for item in data["valid_edges"].shape)
    result.test_edges_shape = tuple(int(item) for item in data["test_edges"].shape)
    valid_neg = data.get("valid_negative_edges")
    test_neg = data.get("test_negative_edges")
    result.valid_negative_edges_shape = (
        None if valid_neg is None else tuple(int(item) for item in valid_neg.shape)
    )
    result.test_negative_edges_shape = (
        None if test_neg is None else tuple(int(item) for item in test_neg.shape)
    )
    return result


def load_manual_split(canonical_name: str) -> RawSplitResult:
    sanitized = canonical_name.replace("-", "_")
    raw_dir = RAW_ROOT / sanitized
    result = RawSplitResult(canonical_name=canonical_name, raw_dir=str(raw_dir))
    result.source = "cached raw files and split/*.pt"

    result.num_nodes = read_gzip_scalar(raw_dir / "raw" / "num-node-list.csv.gz")
    num_edges = read_gzip_scalar(raw_dir / "raw" / "num-edge-list.csv.gz")
    if num_edges is not None:
        result.edge_index_shape = (2, num_edges)

    split_root = raw_dir / "split"
    split_files = {
        path.name.removesuffix(".pt"): path
        for path in split_root.glob("*/*.pt")
        if path.name in {"train.pt", "valid.pt", "test.pt"}
    }
    if set(split_files) != {"train", "valid", "test"}:
        result.status = f"missing split files under {split_root}"
        return result

    try:
        import torch
    except ImportError as exc:
        result.status = f"manual split read failed: torch not available: {exc}"
        return result

    try:
        for split_name, split_path in sorted(split_files.items()):
            split = torch.load(split_path, weights_only=False)
            if not isinstance(split, dict):
                result.status = f"{split_path} did not load to a dict"
                return result
            if split_name == "train":
                result.train_edges_shape = positive_edge_shape(split)
            elif split_name == "valid":
                result.valid_edges_shape = positive_edge_shape(split)
                result.valid_negative_edges_shape = negative_edge_shape(split)
                if "target_node_neg" in split:
                    result.extra["valid_target_node_neg_shape"] = array_shape(
                        split["target_node_neg"]
                    )
            elif split_name == "test":
                result.test_edges_shape = positive_edge_shape(split)
                result.test_negative_edges_shape = negative_edge_shape(split)
                if "target_node_neg" in split:
                    result.extra["test_target_node_neg_shape"] = array_shape(
                        split["target_node_neg"]
                    )
            del split
            gc.collect()
    except Exception as exc:  # noqa: BLE001
        result.status = f"manual split read failed: {type(exc).__name__}: {exc}"
        return result

    result.status = "ok"
    return result


def load_raw_split(canonical_name: str) -> RawSplitResult:
    sanitized = canonical_name.replace("-", "_")
    processed_cache = RAW_ROOT / sanitized / "processed" / "data_processed"
    if processed_cache.exists():
        result = load_project_dataset_split(canonical_name)
        if result.status == "ok":
            return result

    result = load_manual_split(canonical_name)
    if not processed_cache.exists():
        result.extra["loader_note"] = (
            "Skipped project loader because OGB processed cache is absent; "
            "manual split read avoids creating data/raw cache files."
        )
    return result


def infer_full_graph_usage(dataset: DatasetReport) -> None:
    raw = dataset.raw_split
    graph_edges = dataset.file_checks.get("graph_edges.jsonl")
    relations = dataset.file_checks.get("relations.jsonl")
    if not raw or raw.status != "ok" or not graph_edges:
        dataset.processed_from_full_graph_likely = None
        dataset.leakage_notes.append("Could not compare processed graph_edges with raw split.")
        return

    raw_edge_count = raw.edge_index_shape[1] if raw.edge_index_shape and len(raw.edge_index_shape) == 2 else None
    train_count = raw.train_edges_shape[0] if raw.train_edges_shape else None
    graph_edge_count = graph_edges.line_count
    relation_count = relations.line_count if relations else None

    raw_matches_train = raw_edge_count is not None and train_count is not None and (
        raw_edge_count == train_count
        or (
            dataset.canonical_name in UNDIRECTED_CANONICAL
            and raw_edge_count == train_count * 2
        )
    )
    processed_matches_train = train_count is not None and (
        relation_count == train_count or graph_edge_count == train_count
    )

    if raw_matches_train and processed_matches_train:
        dataset.processed_from_full_graph_likely = False
        dataset.leakage_notes.append(
            "processed counts align with train_edges; no count-level evidence that valid/test positives are included in processed graph_edges."
        )
    elif raw_edge_count is not None and relation_count == raw_edge_count:
        dataset.processed_from_full_graph_likely = True
        dataset.leakage_notes.append(
            "relations/triples align with raw edge.csv.gz edge count, so processed data appears derived from the full raw graph."
        )
    elif raw_edge_count is not None and graph_edge_count == raw_edge_count:
        dataset.processed_from_full_graph_likely = True
        dataset.leakage_notes.append(
            "graph_edges count exactly matches raw edge.csv.gz edge count, so processed graph_edges appears to be the full raw graph."
        )
    elif train_count is not None and graph_edge_count == train_count:
        dataset.processed_from_full_graph_likely = False
        dataset.leakage_notes.append(
            "graph_edges count matches train split edge count."
        )
    elif train_count is not None:
        dataset.processed_from_full_graph_likely = True
        dataset.leakage_notes.append(
            f"graph_edges count ({graph_edge_count:,}) does not match train_edges ({train_count:,}); do not treat it as the official visible training graph."
        )

    if train_count is not None:
        dataset.leakage_notes.append(
            "Formal OGB evaluation should build the visible graph from train_edges and score valid/test positive plus negative edges."
        )


def read_sample_graph(dataset_dir: Path, sample_size: int) -> tuple[nx.Graph, int, int]:
    graph = nx.Graph()
    nodes_read = 0
    edges_read = 0

    nodes_path = dataset_dir / "graph_nodes.jsonl"
    if nodes_path.exists():
        with nodes_path.open("rb") as handle:
            for line in handle:
                if nodes_read >= sample_size:
                    break
                if not line.strip():
                    continue
                obj = json.loads(line)
                graph.add_node(
                    str(obj.get("id")),
                    label=obj.get("label"),
                    type=obj.get("type"),
                    weight=obj.get("weight"),
                    metadata=obj.get("metadata"),
                )
                nodes_read += 1

    edges_path = dataset_dir / "graph_edges.jsonl"
    if edges_path.exists():
        with edges_path.open("rb") as handle:
            for line in handle:
                if edges_read >= sample_size:
                    break
                if not line.strip():
                    continue
                obj = json.loads(line)
                weight = obj.get("weight", 1.0)
                graph.add_edge(
                    str(obj.get("source")),
                    str(obj.get("target")),
                    relation=obj.get("relation"),
                    weight=float(weight) if is_number(weight) else 1.0,
                )
                edges_read += 1

    return graph, nodes_read, edges_read


def run_smoke_test(dataset: DatasetReport, sample_size: int) -> SmokeResult:
    result = SmokeResult()
    graph, result.sample_nodes_read, result.sample_edges_read = read_sample_graph(
        dataset.path, sample_size
    )
    result.graph_nodes = graph.number_of_nodes()
    result.graph_edges = graph.number_of_edges()

    if graph.number_of_nodes() == 0:
        result.note = "No sample graph built."
        return result

    result.components = nx.number_connected_components(graph)
    degrees = [degree for _, degree in graph.degree()]
    result.average_degree = sum(degrees) / len(degrees) if degrees else 0.0
    result.max_degree = max(degrees) if degrees else 0

    try:
        from src.graph_algorithms import (
            adamic_adar_predictions,
            detect_louvain_communities,
            jaccard_predictions,
            resource_allocation_predictions,
        )
    except Exception as exc:  # noqa: BLE001
        result.algorithm_status["import"] = f"failed: {type(exc).__name__}: {exc}"
        return result

    algorithms = {
        "jaccard_predictions": lambda: jaccard_predictions(graph, top_n=5),
        "resource_allocation_predictions": lambda: resource_allocation_predictions(
            graph, top_n=5
        ),
        "adamic_adar_predictions": lambda: adamic_adar_predictions(graph, top_n=5),
        "detect_louvain_communities": lambda: detect_louvain_communities(graph),
    }

    for name, call in algorithms.items():
        try:
            output = call()
            if name == "detect_louvain_communities":
                result.algorithm_status[name] = f"ok ({len(output)} communities)"
            else:
                result.algorithm_status[name] = f"ok ({len(output)} predictions)"
        except Exception as exc:  # noqa: BLE001
            result.algorithm_status[name] = f"failed: {type(exc).__name__}: {exc}"

    if dataset.canonical_name == "ogbl-citation2":
        result.note = (
            "Smoke test uses an undirected NetworkX Graph because current algorithms "
            "accept nx.Graph; this is not a formal directed citation2 metric."
        )

    return result


def validate_dataset(dataset_dir: Path, sample_size: int) -> DatasetReport:
    dataset = DatasetReport(
        name=dataset_dir.name,
        canonical_name=canonical_dataset_name(dataset_dir.name),
        path=dataset_dir,
    )

    stats_path = dataset_dir / "stats.json"
    dataset.stats, dataset.stats_valid, dataset.stats_errors = load_stats(stats_path)
    dataset.file_checks["stats.json"] = FileCheck(
        name="stats.json",
        exists=stats_path.exists(),
        size_bytes=stats_path.stat().st_size if stats_path.exists() else 0,
        line_count=0,
        valid_json=dataset.stats_valid,
        empty=stats_path.exists() and stats_path.stat().st_size == 0,
        errors=dataset.stats_errors,
    )

    for filename in ("documents.jsonl", "entities.jsonl", "relations.jsonl", "triples.jsonl"):
        dataset.file_checks[filename] = validate_generic_jsonl(dataset_dir / filename, dataset.name)

    node_file, dataset.node_check, node_ids, max_node_idx = validate_nodes(
        dataset_dir / "graph_nodes.jsonl", dataset.name
    )
    dataset.file_checks["graph_nodes.jsonl"] = node_file

    num_nodes_for_key = max(max_node_idx + 1, len(node_ids), 1)
    edge_file, dataset.edge_check = validate_edges(
        dataset_dir / "graph_edges.jsonl",
        dataset.name,
        node_ids,
        num_nodes_for_key,
    )
    dataset.file_checks["graph_edges.jsonl"] = edge_file

    compare_counts(dataset)
    dataset.raw_split = load_raw_split(dataset.canonical_name)
    infer_full_graph_usage(dataset)
    dataset.smoke = run_smoke_test(dataset, sample_size)

    del node_ids
    gc.collect()
    return dataset


def ok_bad(condition: bool) -> str:
    return "OK" if condition else "ISSUE"


def fmt_shape(shape: tuple[int, ...] | None) -> str:
    if shape is None:
        return "N/A"
    return "x".join(f"{item:,}" for item in shape)


def top_counter(counter: Counter[str], limit: int = 5) -> str:
    if not counter:
        return "none"
    return ", ".join(f"{key}={value:,}" for key, value in counter.most_common(limit))


def file_inventory_table(dataset: DatasetReport) -> list[str]:
    lines = ["| File | Exists | Size | JSON/Stats valid | Lines | Empty |", "|---|---:|---:|---:|---:|---:|"]
    for filename in EXPECTED_FILES:
        check = dataset.file_checks.get(filename)
        if not check:
            lines.append(f"| `{filename}` | no | N/A | no | N/A | N/A |")
            continue
        lines.append(
            f"| `{filename}` | {'yes' if check.exists else 'no'} | {human_size(check.size_bytes)} | "
            f"{'yes' if check.valid_json else 'no'} | {check.line_count:,} | {'yes' if check.empty else 'no'} |"
        )
    return lines


def processed_checks_section(dataset: DatasetReport) -> list[str]:
    lines = [f"### {dataset.name}"]
    all_files = all(dataset.file_checks.get(name, FileCheck(name, False)).exists for name in EXPECTED_FILES)
    all_json = all(check.valid_json for check in dataset.file_checks.values())
    lines.append(f"- File presence: {ok_bad(all_files)}")
    lines.append(f"- JSONL/stats validity: {ok_bad(all_json)}")
    if dataset.stats_valid:
        warning_text = "; ".join(dataset.stats.get("warnings", [])) or "none"
        lines.append(f"- stats.json warning count: {dataset.stats.get('num_warnings', 0)}; warnings: {warning_text}")
    else:
        lines.append(f"- stats.json errors: {'; '.join(dataset.stats_errors)}")

    if dataset.count_mismatches:
        lines.append(f"- Count consistency: ISSUE; {'; '.join(dataset.count_mismatches)}")
    else:
        lines.append("- Count consistency: OK; JSONL line counts match stats.json counters.")

    node = dataset.node_check
    lines.append(
        "- graph_nodes fields/id checks: "
        f"missing_fields={dict(node.missing_fields) or 'none'}, "
        f"duplicate_ids={node.duplicate_ids:,}, "
        f"missing_metadata.node_idx={node.missing_metadata_node_idx:,}, "
        f"invalid_metadata.node_idx={node.invalid_metadata_node_idx:,}, "
        f"id_vs_metadata_mismatch={node.id_metadata_mismatch:,}."
    )
    lines.append(f"- graph_nodes id formats: {top_counter(node.id_format_counts)}")

    edge = dataset.edge_check
    lines.append(
        "- graph_edges field/endpoint checks: "
        f"missing_fields={dict(edge.missing_fields) or 'none'}, "
        f"missing_endpoints={edge.missing_endpoints:,}, "
        f"invalid_weights={edge.invalid_weights:,}."
    )
    lines.append(
        "- graph_edges structure checks: "
        f"self_loops={edge.self_loops:,}, duplicate_edges={edge.duplicate_edges:,}, "
        f"reverse_edges_seen={edge.reverse_edges_seen:,}, relations={top_counter(edge.relation_values)}."
    )
    lines.append(f"- source id formats: {top_counter(edge.source_format_counts)}")
    lines.append(f"- target id formats: {top_counter(edge.target_format_counts)}")

    if dataset.name == "ogbl_citation2":
        lines.append(
            "- Directionality note: citation2 is a directed link prediction task; reverse-edge count is reported for diagnostics only."
        )
    elif dataset.canonical_name in UNDIRECTED_CANONICAL:
        lines.append(
            "- Undirected note: reverse-edge duplicates are diagnostic only; they are not automatically treated as an error."
        )

    return lines


def split_section(dataset: DatasetReport) -> list[str]:
    raw = dataset.raw_split
    lines = [f"### {dataset.name}"]
    if raw is None:
        return lines + ["- Raw split status: not checked."]

    lines.append(f"- Raw split status: {raw.status}")
    lines.append(f"- Source: {raw.source or 'N/A'}")
    lines.append(f"- Raw dir: `{Path(raw.raw_dir).relative_to(PROJECT_ROOT) if Path(raw.raw_dir).is_absolute() else raw.raw_dir}`")
    lines.append(f"- num_nodes: {raw.num_nodes:,}" if raw.num_nodes is not None else "- num_nodes: N/A")
    lines.append(f"- edge_index shape: {fmt_shape(raw.edge_index_shape)}")
    lines.append(f"- train_edges shape: {fmt_shape(raw.train_edges_shape)}")
    lines.append(f"- valid_edges shape: {fmt_shape(raw.valid_edges_shape)}")
    lines.append(f"- test_edges shape: {fmt_shape(raw.test_edges_shape)}")
    lines.append(f"- valid_negative_edges shape: {fmt_shape(raw.valid_negative_edges_shape)}")
    lines.append(f"- test_negative_edges shape: {fmt_shape(raw.test_negative_edges_shape)}")
    for key, value in sorted(raw.extra.items()):
        lines.append(f"- {key}: {value}")

    for note in dataset.leakage_notes:
        lines.append(f"- Leakage/evaluation note: {note}")
    return lines


def smoke_section(dataset: DatasetReport) -> list[str]:
    smoke = dataset.smoke
    lines = [f"### {dataset.name}"]
    if smoke is None:
        return lines + ["- Smoke test was not run."]

    lines.append(f"- Sample read: nodes={smoke.sample_nodes_read:,}, edges={smoke.sample_edges_read:,}")
    lines.append(
        f"- NetworkX sample graph: nodes={smoke.graph_nodes:,}, edges={smoke.graph_edges:,}, "
        f"components={smoke.components:,}, average_degree={smoke.average_degree:.4f}, "
        f"max_degree={smoke.max_degree:,}."
    )
    for name, status in smoke.algorithm_status.items():
        lines.append(f"- {name}: {status}")
    if smoke.note:
        lines.append(f"- Note: {smoke.note}")
    return lines


def build_report(datasets: list[DatasetReport], sample_size: int) -> str:
    blocking: list[str] = []
    warnings: list[str] = []

    if not DATA_ROOT.exists():
        blocking.append("`data/` directory is missing.")
    if not PROCESSED_ROOT.exists():
        blocking.append("`data/processed/` directory is missing.")
    if not RAW_ROOT.exists():
        blocking.append("`data/raw/` directory is missing.")
    if len(datasets) < 3:
        warnings.append(f"Expected three OGB processed datasets, discovered {len(datasets)}.")

    for dataset in datasets:
        missing = [name for name in EXPECTED_FILES if not dataset.file_checks.get(name, FileCheck(name, False)).exists]
        if missing:
            blocking.append(f"{dataset.name}: missing files: {', '.join(missing)}.")
        invalid = [name for name, check in dataset.file_checks.items() if not check.valid_json]
        if invalid:
            blocking.append(f"{dataset.name}: invalid JSON/stats files: {', '.join(sorted(invalid))}.")
        if dataset.count_mismatches:
            blocking.append(f"{dataset.name}: stats count mismatches detected.")
        if dataset.node_check.duplicate_ids:
            blocking.append(f"{dataset.name}: duplicate graph node ids detected.")
        if dataset.edge_check.missing_endpoints:
            blocking.append(f"{dataset.name}: graph edges point to missing nodes.")
        if dataset.edge_check.missing_fields or dataset.node_check.missing_fields:
            blocking.append(f"{dataset.name}: required node/edge fields are missing.")
        if dataset.raw_split is None or dataset.raw_split.status != "ok":
            warnings.append(f"{dataset.name}: official split could not be fully verified.")
        if dataset.processed_from_full_graph_likely:
            warnings.append(
                f"{dataset.name}: processed graph_edges/relations appear derived from raw full graph; do not use them directly as the visible training graph for official metrics."
            )
        if dataset.canonical_name == "ogbl-citation2":
            warnings.append(
                "ogbl_citation2: smoke test uses undirected nx.Graph wrappers; formal citation2 evaluation must preserve directed semantics."
            )
        if dataset.edge_check.reverse_edges_seen and dataset.canonical_name in UNDIRECTED_CANONICAL:
            warnings.append(
                f"{dataset.name}: reverse edge pairs are present ({dataset.edge_check.reverse_edges_seen:,}); expected for some undirected encodings but should be deduplicated intentionally in algorithms."
            )

    warnings.append("Full-graph NetworkX link prediction is not scalable for these data sizes; keep full metrics split-based and candidate-limited.")
    warnings.append("Repository contains `__pycache__` directories; they are harmless but should stay ignored by git.")

    if blocking:
        recommendation = "可以进入但需要先修复上述 blocking issues。"
    else:
        recommendation = (
            "可以进入下一步传统图算法开发；正式 OGB 指标验证必须基于 train_edges 构建可见训练图，"
            "并用 valid/test positive + negative edges 评测，不能直接把 processed graph_edges 同时当训练图和测试来源。"
        )

    lines: list[str] = [
        "# Data Validation Report",
        "",
        f"Generated by `scripts/validate_processed_data.py` with sample_size={sample_size}.",
        "",
        "## 1. Summary",
        "",
        f"- Discovered `data/`: {'yes' if DATA_ROOT.exists() else 'no'}; `data/raw/`: {'yes' if RAW_ROOT.exists() else 'no'}; `data/processed/`: {'yes' if PROCESSED_ROOT.exists() else 'no'}.",
        f"- Discovered processed OGB datasets: {', '.join(dataset.name for dataset in datasets) or 'none'}.",
        f"- File integrity blockers: {len(blocking)}.",
        f"- Overall recommendation: {recommendation}",
        "",
        "## 2. Dataset Inventory",
        "",
    ]

    for dataset in datasets:
        lines.append(f"### {dataset.name}")
        lines.extend(file_inventory_table(dataset))
        lines.append("")

    lines.append("## 3. Processed File Checks")
    lines.append("")
    for dataset in datasets:
        lines.extend(processed_checks_section(dataset))
        lines.append("")

    lines.append("## 4. OGB Split / Leakage Risk")
    lines.append("")
    lines.append(
        "正式图算法评测不能把 `raw/edge.csv.gz` 或由其生成的 full `processed/graph_edges.jsonl` 直接当训练图；"
        "应使用官方 `train_edges` 构建可见图，并用 `valid/test` 正负边计算指标。"
    )
    lines.append("")
    for dataset in datasets:
        lines.extend(split_section(dataset))
        lines.append("")

    lines.append("## 5. NetworkX Smoke Test")
    lines.append("")
    for dataset in datasets:
        lines.extend(smoke_section(dataset))
        lines.append("")

    lines.append("## 6. Blocking Issues")
    lines.append("")
    if blocking:
        lines.extend(f"- {item}" for item in blocking)
    else:
        lines.append("- None for entering the next traditional graph algorithm development step.")
    lines.append("")

    lines.append("## 7. Non-blocking Warnings")
    lines.append("")
    if warnings:
        for item in warnings:
            lines.append(f"- {item}")
    else:
        lines.append("- None.")
    lines.append("")

    lines.append("## 8. Recommendation")
    lines.append("")
    lines.append(recommendation)
    lines.append("")

    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sample-size",
        type=int,
        default=500,
        help="Number of leading graph_nodes and graph_edges records used for NetworkX smoke tests.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=REPORT_PATH,
        help="Markdown report path.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_dirs = discover_processed_datasets()
    reports = [validate_dataset(path, args.sample_size) for path in dataset_dirs]
    report_text = build_report(reports, args.sample_size)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report_text, encoding="utf-8")
    print(args.report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
