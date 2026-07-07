"""Command-line preprocessing pipeline for graph construction inputs."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import sqlite3
from collections import defaultdict
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from .data_loader import DatasetInput, discover_dataset, iter_raw_records
from .normalizer import (
    detect_original_id,
    detect_split,
    detect_text,
    detect_title,
    field_names,
    normalize_entity_name,
    normalize_split,
    normalize_whitespace,
    sanitize_dataset_name,
    stable_doc_id,
    stable_entity_id,
    stable_relation_id,
)
from .schema import JsonDict, PreprocessStats, RawRecord


OUTPUT_FILES = (
    "documents.jsonl",
    "entities.jsonl",
    "relations.jsonl",
    "triples.jsonl",
    "graph_nodes.jsonl",
    "graph_edges.jsonl",
    "stats.json",
)

ENTITY_FIELDS = ("entities", "entity", "nodes", "mentions")
RELATION_FIELDS = (
    "relations",
    "relation",
    "edges",
    "edge",
    "triples",
    "triple",
    "spo_list",
    "spo",
    "links",
)
HEAD_FIELDS = ("head", "subject", "source", "source_node", "from", "h")
TAIL_FIELDS = ("tail", "object", "target", "target_node", "to", "t")
PREDICATE_FIELDS = ("relation_type", "predicate", "relation", "type", "label", "r")
EVIDENCE_FIELDS = ("evidence", "evidence_text", "sentence", "context", "text")
CONFIDENCE_FIELDS = ("confidence", "score", "probability")


def main() -> None:
    args = parse_args()
    try:
        stats = process_input(
            input_path=args.input,
            output_path=args.output,
            max_samples=args.max_samples,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"Error: {exc}") from exc

    print(
        f"{stats['dataset_name']}: "
        f"documents={stats['num_documents']} "
        f"entities={stats['num_entities']} "
        f"relations={stats['num_relations']} "
        f"graph_edges={stats['num_graph_edges']} "
        f"warnings={stats['num_warnings']}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess raw datasets into graph-construction JSONL files."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input file or one dataset directory, e.g. data/raw/ogbl_citation2.",
    )
    parser.add_argument(
        "--output",
        default="processed",
        help="Output root for processed/<dataset_name>/ files.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional smoke-test limit for raw records or OGB edge rows.",
    )
    return parser.parse_args()


def process_input(
    input_path: str | Path,
    output_path: str | Path,
    *,
    max_samples: int | None = None,
) -> JsonDict:
    """Process exactly one input dataset and return its stats dictionary."""

    dataset = discover_dataset(input_path)
    return process_dataset(dataset, output_root=output_path, max_samples=max_samples).to_dict()


def process_all(
    input_path: str | Path,
    output_path: str | Path,
    *,
    max_samples: int | None = None,
) -> list[JsonDict]:
    """Backward-compatible wrapper that still processes exactly one dataset."""

    return [process_input(input_path, output_path, max_samples=max_samples)]


def process_dataset(
    dataset: DatasetInput,
    *,
    output_root: str | Path,
    max_samples: int | None = None,
) -> PreprocessStats:
    """Process one discovered dataset into the standard output layout."""

    if dataset.is_ogb:
        return _process_ogb_dataset(dataset, Path(output_root), max_samples=max_samples)
    return _process_generic_dataset(dataset, Path(output_root), max_samples=max_samples)


def _process_generic_dataset(
    dataset: DatasetInput,
    output_root: Path,
    *,
    max_samples: int | None,
) -> PreprocessStats:
    stats = PreprocessStats(dataset_name=dataset.name)
    output_dir = _prepare_output_dir(output_root, dataset.name)

    entities_by_key: dict[str, JsonDict] = {}
    relations: list[JsonDict] = []

    with (output_dir / "documents.jsonl").open("w", encoding="utf-8") as documents_file:
        for raw in iter_raw_records(dataset, warning_callback=stats.add_warning):
            if max_samples is not None and stats.num_raw_samples >= max_samples:
                break

            stats.num_raw_samples += 1
            record_index = stats.num_raw_samples - 1
            try:
                document = _normalize_document(raw, record_index, stats)
                _write_jsonl(documents_file, document)
                stats.num_documents += 1

                entity_stubs, entity_fields = _extract_entities(raw.data)
                stats.detected_entity_fields.update(entity_fields)
                for entity_stub in entity_stubs:
                    _upsert_entity(
                        entities_by_key,
                        dataset.name,
                        name=entity_stub["name"],
                        entity_type=entity_stub["type"],
                        source_doc_id=document["doc_id"],
                        aliases=entity_stub.get("aliases", []),
                        metadata=entity_stub.get("metadata", {}),
                    )

                relation_stubs, relation_fields = _extract_relations(raw.data, stats)
                stats.detected_relation_fields.update(relation_fields)
                for relation_stub in relation_stubs:
                    relation = _materialize_relation(
                        dataset_name=dataset.name,
                        relation_stub=relation_stub,
                        source_doc_id=document["doc_id"],
                        relation_index=len(relations),
                        entities_by_key=entities_by_key,
                    )
                    relations.append(relation)
            except Exception as exc:
                stats.skipped_samples += 1
                stats.add_warning(
                    f"Skipped sample {raw.source_file}#{raw.index}: {type(exc).__name__}: {exc}"
                )

    entities = sorted(entities_by_key.values(), key=lambda item: item["entity_id"])
    _write_jsonl_file(output_dir / "entities.jsonl", entities)
    _write_jsonl_file(output_dir / "relations.jsonl", relations)

    triples = [_triple_from_relation(relation) for relation in relations]
    _write_jsonl_file(output_dir / "triples.jsonl", triples)

    graph_nodes = [_graph_node_from_entity(entity) for entity in entities]
    _write_jsonl_file(output_dir / "graph_nodes.jsonl", graph_nodes)

    graph_edges, skipped_edges = _graph_edges_from_relations(relations, entities)
    _write_jsonl_file(output_dir / "graph_edges.jsonl", graph_edges)

    stats.num_entities = len(entities)
    stats.num_relations = len(relations)
    stats.num_triples = len(triples)
    stats.num_graph_nodes = len(graph_nodes)
    stats.num_graph_edges = len(graph_edges)
    stats.skipped_edges += skipped_edges
    _write_stats(output_dir, stats)
    return stats


def _normalize_document(raw: RawRecord, record_index: int, stats: PreprocessStats) -> JsonDict:
    split = detect_split(raw.data, raw.split)
    doc_id = stable_doc_id(raw.dataset_name, split, record_index)
    text_field, text = detect_text(raw.data)
    if text_field is None:
        stats.add_warning(
            f"No text field found in {raw.source_file} record {raw.index}; kept empty text."
        )

    raw_fields: JsonDict = {
        "source_file": str(raw.source_file),
        "record_index": raw.index,
        "record_type": raw.record_type,
        "field_names": field_names(raw.data),
    }
    if text_field is not None:
        raw_fields["detected_text_field"] = text_field

    return {
        "doc_id": doc_id,
        "title": detect_title(raw.data),
        "text": text,
        "source": raw.dataset_name,
        "metadata": {
            "original_id": detect_original_id(raw.data),
            "split": split,
            "raw_fields": raw_fields,
        },
    }


def _extract_entities(record: Any) -> tuple[list[JsonDict], set[str]]:
    if not isinstance(record, dict):
        return [], set()

    detected_fields: set[str] = set()
    entities: list[JsonDict] = []
    lowered = {str(key).lower(): key for key in record}
    for field in ENTITY_FIELDS:
        key = lowered.get(field)
        if key is None:
            continue
        detected_fields.add(str(key))
        for item in _iter_entity_items(record[key]):
            parsed = _parse_entity_item(item)
            if parsed is not None:
                entities.append(parsed)
    return entities, detected_fields


def _iter_entity_items(value: Any) -> Iterator[Any]:
    if isinstance(value, list):
        for item in value:
            yield from _iter_entity_items(item)
        return
    if isinstance(value, tuple):
        for item in value:
            yield from _iter_entity_items(item)
        return
    if isinstance(value, dict):
        if _looks_like_entity_dict(value):
            yield value
            return
        for key, item in value.items():
            if isinstance(item, dict):
                merged = dict(item)
                merged.setdefault("id", key)
                yield merged
            else:
                yield {"id": key, "name": item}
        return
    yield value


def _looks_like_entity_dict(value: dict[str, Any]) -> bool:
    keys = {str(key).lower() for key in value}
    return bool(
        keys
        & {
            "name",
            "entity",
            "entity_name",
            "mention",
            "text",
            "label",
            "id",
            "node_id",
        }
    )


def _parse_entity_item(item: Any) -> JsonDict | None:
    if isinstance(item, dict):
        name = _entity_name_from_value(item)
        entity_type = _first_field_value(
            item, ("type", "entity_type", "node_type", "category", "kind")
        )
        aliases = item.get("aliases", [])
        if not isinstance(aliases, list):
            aliases = [aliases]
        metadata = {
            "raw_field_names": field_names(item),
        }
        original_id = _first_field_value(item, ("id", "node_id", "entity_id"))
        if original_id:
            metadata["original_id"] = normalize_whitespace(original_id)
    else:
        name = normalize_entity_name(item)
        entity_type = ""
        aliases = []
        metadata = {}

    if not name:
        return None

    normalized_aliases = [
        alias for alias in (normalize_entity_name(alias) for alias in aliases) if alias
    ]
    return {
        "name": name,
        "type": normalize_whitespace(entity_type) or "unknown",
        "aliases": normalized_aliases,
        "metadata": metadata,
    }


def _entity_name_from_value(value: Any) -> str:
    if not isinstance(value, dict):
        return normalize_entity_name(value)
    name = _first_field_value(
        value,
        (
            "name",
            "entity",
            "entity_name",
            "mention",
            "text",
            "label",
            "id",
            "node_id",
            "subject",
            "object",
            "source",
            "target",
            "head",
            "tail",
        ),
    )
    return normalize_entity_name(name)


def _extract_relations(
    record: Any,
    stats: PreprocessStats,
) -> tuple[list[JsonDict], set[str]]:
    if not isinstance(record, dict):
        return [], set()

    detected_fields: set[str] = set()
    relations: list[JsonDict] = []

    direct = None
    if _looks_like_top_level_relation(record):
        direct = _parse_relation_item(record, "record")
    if direct is not None:
        relations.append(direct)

    lowered = {str(key).lower(): key for key in record}
    for field in RELATION_FIELDS:
        key = lowered.get(field)
        if key is None:
            continue
        detected_fields.add(str(key))
        for item in _iter_relation_items(record[key]):
            parsed = _parse_relation_item(item, field)
            if parsed is None:
                stats.add_warning(f"Could not parse relation item from field '{key}'.")
                continue
            relations.append(parsed)

    return relations, detected_fields


def _iter_relation_items(value: Any) -> Iterator[Any]:
    if isinstance(value, (list, tuple)):
        if _looks_like_relation_sequence(value):
            yield value
            return
        for item in value:
            yield item
        return
    if isinstance(value, dict):
        if _looks_like_relation_dict(value):
            yield value
            return
        for key, item in value.items():
            if isinstance(item, dict):
                merged = dict(item)
                merged.setdefault("id", key)
                yield merged
            else:
                yield item
        return
    yield value


def _looks_like_relation_sequence(value: Any) -> bool:
    return (
        isinstance(value, (list, tuple))
        and 2 <= len(value) <= 4
        and all(not isinstance(item, (list, tuple, dict)) for item in value)
    )


def _looks_like_relation_dict(value: dict[str, Any]) -> bool:
    keys = {str(key).lower() for key in value}
    return bool(keys & set(HEAD_FIELDS)) and bool(keys & set(TAIL_FIELDS))


def _looks_like_top_level_relation(value: dict[str, Any]) -> bool:
    keys = {str(key).lower() for key in value}
    has_subject_object = "subject" in keys and "object" in keys
    has_head_tail = "head" in keys and "tail" in keys
    has_source_node_target_node = "source_node" in keys and "target_node" in keys
    has_from_to = "from" in keys and "to" in keys
    has_source_target_with_predicate = (
        "source" in keys
        and "target" in keys
        and bool(keys & {"relation", "predicate", "type", "relation_type"})
    )
    return any(
        (
            has_subject_object,
            has_head_tail,
            has_source_node_target_node,
            has_from_to,
            has_source_target_with_predicate,
        )
    )


def _parse_relation_item(item: Any, field_name: str) -> JsonDict | None:
    metadata: JsonDict = {"source_field": field_name}
    evidence = ""
    confidence = 1.0

    if isinstance(item, (list, tuple)):
        if len(item) < 2:
            return None
        if field_name in {"edges", "edge", "links"}:
            head = normalize_entity_name(item[0])
            tail = normalize_entity_name(item[1])
            relation_type = "related_to"
            if len(item) >= 3:
                if _is_number_like(item[2]):
                    metadata["edge_value"] = item[2]
                else:
                    relation_type = normalize_whitespace(item[2]) or "related_to"
        else:
            if len(item) < 3:
                return None
            head = normalize_entity_name(item[0])
            relation_type = normalize_whitespace(item[1]) or "related_to"
            tail = normalize_entity_name(item[2])
    elif isinstance(item, dict):
        head = _entity_name_from_value(_first_field_value(item, HEAD_FIELDS))
        tail = _entity_name_from_value(_first_field_value(item, TAIL_FIELDS))
        relation_type = (
            normalize_whitespace(_first_field_value(item, PREDICATE_FIELDS)) or "related_to"
        )
        evidence = normalize_whitespace(_first_field_value(item, EVIDENCE_FIELDS))
        confidence = _safe_float(_first_field_value(item, CONFIDENCE_FIELDS), 1.0)
        metadata["raw_field_names"] = field_names(item)
    else:
        return None

    if not head or not tail:
        return None

    return {
        "head": head,
        "tail": tail,
        "relation_type": relation_type,
        "evidence": evidence,
        "confidence": confidence,
        "metadata": metadata,
    }


def _materialize_relation(
    *,
    dataset_name: str,
    relation_stub: JsonDict,
    source_doc_id: str,
    relation_index: int,
    entities_by_key: dict[str, JsonDict],
) -> JsonDict:
    head_entity = _upsert_entity(
        entities_by_key,
        dataset_name,
        name=relation_stub["head"],
        entity_type="unknown",
        source_doc_id=source_doc_id,
        aliases=[],
        metadata={"inferred_from_relation": True},
    )
    tail_entity = _upsert_entity(
        entities_by_key,
        dataset_name,
        name=relation_stub["tail"],
        entity_type="unknown",
        source_doc_id=source_doc_id,
        aliases=[],
        metadata={"inferred_from_relation": True},
    )

    return {
        "relation_id": stable_relation_id(
            dataset_name,
            source_doc_id,
            relation_index,
            relation_stub["head"],
            relation_stub["relation_type"],
            relation_stub["tail"],
        ),
        "head": relation_stub["head"],
        "head_id": head_entity["entity_id"],
        "tail": relation_stub["tail"],
        "tail_id": tail_entity["entity_id"],
        "relation_type": relation_stub["relation_type"],
        "evidence": relation_stub.get("evidence", ""),
        "source_doc_id": source_doc_id,
        "confidence": relation_stub.get("confidence", 1.0),
        "metadata": relation_stub.get("metadata", {}),
    }


def _upsert_entity(
    entities_by_key: dict[str, JsonDict],
    dataset_name: str,
    *,
    name: str,
    entity_type: str,
    source_doc_id: str,
    aliases: Iterable[str],
    metadata: JsonDict,
) -> JsonDict:
    normalized_name = normalize_entity_name(name)
    if not normalized_name:
        raise ValueError("Entity name is empty")

    key = normalized_name.casefold()
    entity_type = normalize_whitespace(entity_type) or "unknown"
    aliases = [alias for alias in (normalize_entity_name(alias) for alias in aliases) if alias]

    if key not in entities_by_key:
        entities_by_key[key] = {
            "entity_id": stable_entity_id(dataset_name, normalized_name),
            "name": normalized_name,
            "type": entity_type,
            "aliases": aliases,
            "source_doc_ids": [source_doc_id] if source_doc_id else [],
            "metadata": dict(metadata),
        }
        return entities_by_key[key]

    entity = entities_by_key[key]
    if source_doc_id and source_doc_id not in entity["source_doc_ids"]:
        entity["source_doc_ids"].append(source_doc_id)
    for alias in aliases:
        if alias not in entity["aliases"]:
            entity["aliases"].append(alias)

    existing_type = entity.get("type", "unknown")
    if existing_type == "unknown" and entity_type != "unknown":
        entity["type"] = entity_type
    elif entity_type not in {"unknown", existing_type}:
        conflicts = entity["metadata"].setdefault("type_conflicts", [])
        if entity_type not in conflicts:
            conflicts.append(entity_type)

    if metadata.get("inferred_from_relation"):
        entity["metadata"].setdefault("inferred_from_relation", True)
    return entity


def _triple_from_relation(relation: JsonDict) -> JsonDict:
    return {
        "subject": relation["head"],
        "predicate": relation["relation_type"],
        "object": relation["tail"],
        "source_doc_id": relation.get("source_doc_id", ""),
        "evidence": relation.get("evidence", ""),
    }


def _graph_node_from_entity(entity: JsonDict) -> JsonDict:
    return {
        "id": entity["entity_id"],
        "label": entity["name"],
        "type": entity.get("type", "unknown"),
        "weight": float(len(entity.get("source_doc_ids", []))),
        "metadata": entity.get("metadata", {}),
    }


def _graph_edges_from_relations(
    relations: list[JsonDict],
    entities: list[JsonDict],
) -> tuple[list[JsonDict], int]:
    entity_name_to_id = {
        normalize_entity_name(entity["name"]).casefold(): entity["entity_id"]
        for entity in entities
    }
    edge_map: dict[tuple[str, str, str], JsonDict] = {}
    skipped_edges = 0

    for relation in relations:
        source = relation.get("head_id") or entity_name_to_id.get(
            normalize_entity_name(relation.get("head", "")).casefold()
        )
        target = relation.get("tail_id") or entity_name_to_id.get(
            normalize_entity_name(relation.get("tail", "")).casefold()
        )
        relation_type = relation.get("relation_type") or "related_to"
        if not source or not target:
            skipped_edges += 1
            continue

        key = (source, target, relation_type)
        if key not in edge_map:
            edge_map[key] = {
                "source": source,
                "target": target,
                "relation": relation_type,
                "weight": 0.0,
                "evidence_doc_ids": [],
                "metadata": {"confidence_sum": 0.0, "relation_count": 0},
            }

        edge = edge_map[key]
        edge["weight"] += 1.0
        edge["metadata"]["relation_count"] += 1
        edge["metadata"]["confidence_sum"] += _safe_float(relation.get("confidence"), 1.0)
        source_doc_id = relation.get("source_doc_id")
        if source_doc_id and source_doc_id not in edge["evidence_doc_ids"]:
            edge["evidence_doc_ids"].append(source_doc_id)

    return sorted(edge_map.values(), key=lambda item: (item["source"], item["target"], item["relation"])), skipped_edges


def _process_ogb_dataset(
    dataset: DatasetInput,
    output_root: Path,
    *,
    max_samples: int | None,
) -> PreprocessStats:
    stats = PreprocessStats(dataset_name=dataset.name)
    output_dir = _prepare_output_dir(output_root, dataset.name)

    (output_dir / "documents.jsonl").write_text("", encoding="utf-8")
    stats.add_warning(
        "OGB graph dataset has no raw document text; documents.jsonl is empty."
    )

    mapping_file = _find_ogb_mapping_file(dataset.path)
    mapping = _read_ogb_mapping(mapping_file) if mapping_file is not None else {}
    num_nodes = _infer_ogb_num_nodes(dataset.path, mapping)
    entity_type = _infer_ogb_entity_type(dataset.name, mapping_file)
    relation_type = _infer_ogb_relation_type(dataset.name)
    if mapping_file is not None:
        stats.detected_entity_fields.add(f"mapping/{mapping_file.name}")
    stats.detected_relation_fields.add("raw/edge.csv.gz")

    graph_edge_db = _create_edge_aggregation_db()
    degree_by_node: defaultdict[int, float] = defaultdict(float)
    seen_nodes: set[int] = set()

    with (output_dir / "relations.jsonl").open("w", encoding="utf-8") as relations_file:
        with (output_dir / "triples.jsonl").open("w", encoding="utf-8") as triples_file:
            for edge_index, source_idx, target_idx, weight, edge_metadata in _iter_ogb_edges(
                dataset.path,
                stats,
            ):
                if max_samples is not None and edge_index >= max_samples:
                    break

                source_id = _ogb_entity_id(dataset.name, source_idx)
                target_id = _ogb_entity_id(dataset.name, target_idx)
                head = mapping.get(source_idx, str(source_idx))
                tail = mapping.get(target_idx, str(target_idx))
                relation = {
                    "relation_id": stable_relation_id(
                        dataset.name,
                        "",
                        edge_index,
                        head,
                        relation_type,
                        tail,
                    ),
                    "head": head,
                    "head_id": source_id,
                    "tail": tail,
                    "tail_id": target_id,
                    "relation_type": relation_type,
                    "evidence": "",
                    "source_doc_id": "",
                    "confidence": 1.0,
                    "metadata": edge_metadata,
                }
                _write_jsonl(relations_file, relation)
                _write_jsonl(triples_file, _triple_from_relation(relation))

                stats.num_raw_samples += 1
                stats.num_relations += 1
                stats.num_triples += 1

                seen_nodes.update((source_idx, target_idx))
                degree_by_node[source_idx] += weight
                degree_by_node[target_idx] += weight
                _add_ogb_graph_edge(
                    graph_edge_db,
                    source_id=source_id,
                    target_id=target_id,
                    relation_type=relation_type,
                    weight=weight,
                )

    node_indices: Iterable[int]
    if max_samples is None:
        node_indices = range(num_nodes)
    else:
        node_indices = sorted(seen_nodes)
        if not seen_nodes:
            node_indices = range(min(num_nodes, max_samples))

    with (output_dir / "entities.jsonl").open("w", encoding="utf-8") as entities_file:
        with (output_dir / "graph_nodes.jsonl").open("w", encoding="utf-8") as nodes_file:
            for node_idx in node_indices:
                name = mapping.get(node_idx, str(node_idx))
                entity = {
                    "entity_id": _ogb_entity_id(dataset.name, node_idx),
                    "name": name,
                    "type": entity_type,
                    "aliases": [],
                    "source_doc_ids": [],
                    "metadata": {
                        "node_idx": node_idx,
                        "ogb_dataset": dataset.name,
                    },
                }
                _write_jsonl(entities_file, entity)
                _write_jsonl(
                    nodes_file,
                    {
                        "id": entity["entity_id"],
                        "label": name,
                        "type": entity_type,
                        "weight": float(degree_by_node.get(node_idx, 0.0)),
                        "metadata": entity["metadata"],
                    },
                )
                stats.num_entities += 1
                stats.num_graph_nodes += 1

    stats.num_graph_edges = _write_aggregated_graph_edges(
        graph_edge_db,
        output_dir / "graph_edges.jsonl",
    )
    graph_edge_db.close()
    _write_stats(output_dir, stats)
    return stats


def _create_edge_aggregation_db() -> sqlite3.Connection:
    connection = sqlite3.connect("")
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute(
        """
        CREATE TABLE edge_agg (
            source TEXT NOT NULL,
            target TEXT NOT NULL,
            relation TEXT NOT NULL,
            weight REAL NOT NULL,
            relation_count INTEGER NOT NULL,
            PRIMARY KEY (source, target, relation)
        )
        """
    )
    return connection


def _add_ogb_graph_edge(
    connection: sqlite3.Connection,
    *,
    source_id: str,
    target_id: str,
    relation_type: str,
    weight: float,
) -> None:
    cursor = connection.execute(
        """
        UPDATE edge_agg
        SET weight = weight + ?, relation_count = relation_count + 1
        WHERE source = ? AND target = ? AND relation = ?
        """,
        (weight, source_id, target_id, relation_type),
    )
    if cursor.rowcount == 0:
        connection.execute(
            """
            INSERT INTO edge_agg (source, target, relation, weight, relation_count)
            VALUES (?, ?, ?, ?, 1)
            """,
            (source_id, target_id, relation_type, weight),
        )


def _write_aggregated_graph_edges(connection: sqlite3.Connection, path: Path) -> int:
    connection.commit()
    count = 0
    with path.open("w", encoding="utf-8") as file_obj:
        cursor = connection.execute(
            """
            SELECT source, target, relation, weight, relation_count
            FROM edge_agg
            ORDER BY source, target, relation
            """
        )
        for source, target, relation, weight, relation_count in cursor:
            _write_jsonl(
                file_obj,
                {
                    "source": source,
                    "target": target,
                    "relation": relation,
                    "weight": float(weight),
                    "evidence_doc_ids": [],
                    "metadata": {"relation_count": int(relation_count)},
                },
            )
            count += 1
    return count



def _iter_ogb_edges(
    dataset_path: Path,
    stats: PreprocessStats,
) -> Iterator[tuple[int, int, int, float, JsonDict]]:
    edge_path = dataset_path / "raw" / "edge.csv.gz"
    weight_iter = _iter_optional_ogb_column(dataset_path / "raw" / "edge_weight.csv.gz")
    year_iter = _iter_optional_ogb_column(dataset_path / "raw" / "edge_year.csv.gz")

    with gzip.open(edge_path, "rt", encoding="utf-8") as edge_file:
        reader = csv.reader(edge_file)
        for edge_index, row in enumerate(reader):
            try:
                source_idx = int(row[0])
                target_idx = int(row[1])
            except (IndexError, ValueError):
                stats.skipped_edges += 1
                stats.add_warning(f"Skipped invalid OGB edge row {edge_index}: {row}")
                continue

            raw_weight = next(weight_iter, None)
            raw_year = next(year_iter, None)
            weight = _safe_float(raw_weight, 1.0)
            metadata: JsonDict = {"edge_index": edge_index}
            if raw_weight is not None:
                metadata["edge_weight"] = weight
            if raw_year is not None:
                metadata["edge_year"] = normalize_whitespace(raw_year)
            yield edge_index, source_idx, target_idx, weight, metadata


def _iter_optional_ogb_column(path: Path) -> Iterator[str]:
    if not path.is_file():
        return iter(())

    def _iterator() -> Iterator[str]:
        with gzip.open(path, "rt", encoding="utf-8") as file_obj:
            for line in file_obj:
                value = line.strip()
                if value:
                    yield value

    return _iterator()


def _infer_ogb_num_nodes(dataset_path: Path, mapping: dict[int, str]) -> int:
    num_nodes = _read_ogb_num_nodes(dataset_path)
    if num_nodes > 0:
        return num_nodes
    if mapping:
        return max(mapping) + 1
    return 0


def _read_ogb_num_nodes(dataset_path: Path) -> int:
    path = dataset_path / "raw" / "num-node-list.csv.gz"
    if not path.is_file():
        return 0
    with gzip.open(path, "rt", encoding="utf-8") as file_obj:
        for line in file_obj:
            value = line.strip()
            if value:
                return int(value)
    return 0


def _find_ogb_mapping_file(dataset_path: Path) -> Path | None:
    mapping_dir = dataset_path / "mapping"
    if not mapping_dir.is_dir():
        return None
    matches = sorted(mapping_dir.glob("nodeidx2*.csv.gz"))
    return matches[0] if matches else None


def _read_ogb_mapping(path: Path) -> dict[int, str]:
    mapping: dict[int, str] = {}
    with gzip.open(path, "rt", encoding="utf-8") as file_obj:
        reader = csv.DictReader(file_obj)
        if not reader.fieldnames or len(reader.fieldnames) < 2:
            return mapping
        node_field = reader.fieldnames[0]
        name_field = reader.fieldnames[1]
        for row in reader:
            try:
                mapping[int(row[node_field])] = normalize_entity_name(row[name_field])
            except (KeyError, ValueError):
                continue
    return mapping


def _infer_ogb_entity_type(dataset_name: str, mapping_file: Path | None) -> str:
    dataset_name = sanitize_dataset_name(dataset_name)
    if "collab" in dataset_name:
        return "author"
    if "ppa" in dataset_name:
        return "protein"
    if "citation" in dataset_name:
        return "paper"
    if mapping_file is None:
        return "unknown"
    stem = mapping_file.name.split(".csv")[0].replace("nodeidx2", "")
    return stem.removesuffix("id").replace("_", "") or "unknown"


def _infer_ogb_relation_type(dataset_name: str) -> str:
    dataset_name = sanitize_dataset_name(dataset_name)
    if "collab" in dataset_name:
        return "collaborates_with"
    if "ppa" in dataset_name:
        return "associated_with"
    if "citation" in dataset_name:
        return "cites"
    return "related_to"


def _ogb_entity_id(dataset_name: str, node_idx: int) -> str:
    return f"{sanitize_dataset_name(dataset_name)}:node:{node_idx}"


def _first_field_value(record: dict[str, Any], fields: Iterable[str]) -> Any:
    lowered = {str(key).lower(): key for key in record}
    for field in fields:
        key = lowered.get(field.lower())
        if key is not None:
            return record[key]
    return None


def _safe_float(value: Any, default: float) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _is_number_like(value: Any) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def _prepare_output_dir(output_root: Path, dataset_name: str) -> Path:
    output_dir = output_root / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename in OUTPUT_FILES:
        path = output_dir / filename
        if filename.endswith(".jsonl"):
            path.write_text("", encoding="utf-8")
    return output_dir


def _write_jsonl_file(path: Path, rows: Iterable[JsonDict]) -> None:
    with path.open("w", encoding="utf-8") as file_obj:
        for row in rows:
            _write_jsonl(file_obj, row)


def _write_jsonl(file_obj: Any, row: JsonDict) -> None:
    file_obj.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
    file_obj.write("\n")


def _write_stats(output_dir: Path, stats: PreprocessStats) -> None:
    (output_dir / "stats.json").write_text(
        json.dumps(stats.to_dict(), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
