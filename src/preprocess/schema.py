"""Shared schemas for the preprocessing layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


JsonDict = dict[str, Any]


@dataclass(frozen=True)
class RawRecord:
    """A raw input sample plus loader provenance."""

    dataset_name: str
    data: Any
    source_file: Path
    index: int
    split: str = "unknown"
    record_type: str = "record"


@dataclass(frozen=True)
class DatasetInput:
    """A dataset directory or single-file dataset discovered under the input root."""

    name: str
    path: Path
    files: tuple[Path, ...] = ()
    is_ogb: bool = False


@dataclass
class PreprocessStats:
    """Counters and warnings written to stats.json."""

    dataset_name: str
    num_raw_samples: int = 0
    num_documents: int = 0
    num_entities: int = 0
    num_relations: int = 0
    num_triples: int = 0
    num_graph_nodes: int = 0
    num_graph_edges: int = 0
    skipped_samples: int = 0
    skipped_edges: int = 0
    warnings: list[str] = field(default_factory=list)
    detected_entity_fields: set[str] = field(default_factory=set)
    detected_relation_fields: set[str] = field(default_factory=set)
    _warning_count: int = 0

    def add_warning(self, message: str) -> None:
        self._warning_count += 1
        if len(self.warnings) < 200:
            self.warnings.append(message)
        elif len(self.warnings) == 200:
            self.warnings.append("Additional warnings suppressed; see num_warnings.")

    def to_dict(self) -> JsonDict:
        return {
            "dataset_name": self.dataset_name,
            "num_raw_samples": self.num_raw_samples,
            "num_documents": self.num_documents,
            "num_entities": self.num_entities,
            "num_relations": self.num_relations,
            "num_triples": self.num_triples,
            "num_graph_nodes": self.num_graph_nodes,
            "num_graph_edges": self.num_graph_edges,
            "num_warnings": self._warning_count,
            "warnings": self.warnings,
            "skipped_samples": self.skipped_samples,
            "skipped_edges": self.skipped_edges,
            "detected_entity_fields": sorted(self.detected_entity_fields),
            "detected_relation_fields": sorted(self.detected_relation_fields),
        }
