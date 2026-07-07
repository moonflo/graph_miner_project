from __future__ import annotations

import gzip
import json
import tempfile
import unittest
from pathlib import Path

from src.preprocess.preprocess import process_all


class PreprocessSmokeTest(unittest.TestCase):
    def test_jsonl_with_triples_generates_graph_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_dir = root / "data" / "demo"
            dataset_dir.mkdir(parents=True)
            input_file = dataset_dir / "samples.jsonl"
            input_file.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "id": "doc-1",
                                "title": "Demo",
                                "text": "Alice works with Bob.",
                                "entities": [
                                    {"name": "Alice", "type": "person"},
                                    {"name": "Bob", "type": "person"},
                                ],
                                "triples": [
                                    {
                                        "subject": "Alice",
                                        "predicate": "works_with",
                                        "object": "Bob",
                                        "evidence": "Alice works with Bob.",
                                    }
                                ],
                            }
                        ),
                        json.dumps(
                            {
                                "id": "doc-2",
                                "content": "Carol studies graph mining.",
                                "entities": [{"name": "Carol", "type": "person"}],
                            }
                        ),
                    ]
                ),
                encoding="utf-8",
            )

            stats = process_all(dataset_dir, root / "processed")

            self.assertEqual(stats[0]["num_documents"], 2)
            output_dir = root / "processed" / "demo"
            self.assertEqual(len(_read_jsonl(output_dir / "documents.jsonl")), 2)
            self.assertEqual(len(_read_jsonl(output_dir / "relations.jsonl")), 1)
            self.assertEqual(len(_read_jsonl(output_dir / "triples.jsonl")), 1)
            self.assertEqual(len(_read_jsonl(output_dir / "graph_nodes.jsonl")), 3)

            edges = _read_jsonl(output_dir / "graph_edges.jsonl")
            self.assertEqual(len(edges), 1)
            self.assertEqual(edges[0]["relation"], "works_with")
            self.assertEqual(edges[0]["weight"], 1.0)

    def test_missing_entity_relation_fields_does_not_crash(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_dir = root / "data" / "plain"
            dataset_dir.mkdir(parents=True)
            (dataset_dir / "samples.jsonl").write_text(
                json.dumps({"id": "plain-1", "text": "A document without annotations."}),
                encoding="utf-8",
            )

            stats = process_all(dataset_dir, root / "processed")

            self.assertEqual(stats[0]["num_documents"], 1)
            self.assertEqual(stats[0]["num_entities"], 0)
            self.assertEqual(stats[0]["num_relations"], 0)
            output_dir = root / "processed" / "plain"
            self.assertEqual(_read_jsonl(output_dir / "entities.jsonl"), [])
            self.assertEqual(_read_jsonl(output_dir / "graph_edges.jsonl"), [])

    def test_top_level_source_target_metadata_is_not_a_relation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_dir = root / "data" / "metadata"
            dataset_dir.mkdir(parents=True)
            (dataset_dir / "samples.jsonl").write_text(
                json.dumps(
                    {
                        "id": "row-1",
                        "text": "This is only a document.",
                        "source": "newswire",
                        "target": "internal-review",
                    }
                ),
                encoding="utf-8",
            )

            stats = process_all(dataset_dir, root / "processed")
            output_dir = root / "processed" / "metadata"

            self.assertEqual(stats[0]["num_documents"], 1)
            self.assertEqual(stats[0]["num_relations"], 0)
            self.assertEqual(_read_jsonl(output_dir / "relations.jsonl"), [])

    def test_missing_text_field_keeps_document_with_warning(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_dir = root / "data" / "no_text"
            dataset_dir.mkdir(parents=True)
            (dataset_dir / "records.jsonl").write_text(
                json.dumps({"id": "row-1", "label": "no text here"}),
                encoding="utf-8",
            )

            stats = process_all(dataset_dir, root / "processed")
            output_dir = root / "processed" / "no_text"
            documents = _read_jsonl(output_dir / "documents.jsonl")

            self.assertEqual(stats[0]["num_documents"], 1)
            self.assertGreaterEqual(stats[0]["num_warnings"], 1)
            self.assertEqual(documents[0]["text"], "")

    def test_parent_directory_with_child_datasets_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            parent = root / "data"
            dataset_dir = parent / "demo"
            dataset_dir.mkdir(parents=True)
            (dataset_dir / "samples.jsonl").write_text(
                json.dumps({"id": "demo-1", "text": "One dataset."}),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "dataset parent directory"):
                process_all(parent, root / "processed")

    def test_data_root_with_raw_child_datasets_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_dir = root / "data" / "raw" / "demo"
            dataset_dir.mkdir(parents=True)
            (dataset_dir / "samples.jsonl").write_text(
                json.dumps({"id": "demo-1", "text": "One dataset."}),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "data root"):
                process_all(root / "data", root / "processed")

    def test_minimal_ogb_dataset_generates_graph_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_dir = root / "data" / "raw" / "ogbl_tiny"
            raw_dir = dataset_dir / "raw"
            mapping_dir = dataset_dir / "mapping"
            raw_dir.mkdir(parents=True)
            mapping_dir.mkdir(parents=True)

            _write_gzip_text(raw_dir / "edge.csv.gz", "0,1\n1,2\n")
            _write_gzip_text(raw_dir / "num-node-list.csv.gz", "3\n")
            _write_gzip_text(
                mapping_dir / "nodeidx2paperid.csv.gz",
                "node idx,paper id\n0,paper-a\n1,paper-b\n2,paper-c\n",
            )

            stats = process_all(dataset_dir, root / "processed")
            output_dir = root / "processed" / "ogbl_tiny"

            self.assertEqual(_read_jsonl(output_dir / "documents.jsonl"), [])
            self.assertEqual(len(_read_jsonl(output_dir / "entities.jsonl")), 3)
            self.assertEqual(len(_read_jsonl(output_dir / "relations.jsonl")), 2)
            self.assertEqual(len(_read_jsonl(output_dir / "triples.jsonl")), 2)
            self.assertEqual(len(_read_jsonl(output_dir / "graph_nodes.jsonl")), 3)
            self.assertEqual(len(_read_jsonl(output_dir / "graph_edges.jsonl")), 2)

            stats_payload = json.loads(
                (output_dir / "stats.json").read_text(encoding="utf-8")
            )
            self.assertEqual(stats_payload["num_raw_samples"], 2)
            self.assertEqual(stats_payload["num_entities"], 3)
            self.assertEqual(stats_payload["num_graph_nodes"], 3)
            self.assertEqual(stats_payload["num_warnings"], 1)
            self.assertIn("OGB graph dataset", stats_payload["warnings"][0])

    def test_ogb_num_nodes_falls_back_to_mapping_size(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_dir = root / "data" / "raw" / "ogbl_mapping_only"
            raw_dir = dataset_dir / "raw"
            mapping_dir = dataset_dir / "mapping"
            raw_dir.mkdir(parents=True)
            mapping_dir.mkdir(parents=True)

            _write_gzip_text(raw_dir / "edge.csv.gz", "0,2\n")
            _write_gzip_text(
                mapping_dir / "nodeidx2paperid.csv.gz",
                "node idx,paper id\n0,paper-a\n1,paper-b\n2,paper-c\n",
            )

            stats = process_all(dataset_dir, root / "processed")
            output_dir = root / "processed" / "ogbl_mapping_only"

            self.assertEqual(stats[0]["num_raw_samples"], 1)
            self.assertEqual(len(_read_jsonl(output_dir / "graph_nodes.jsonl")), 3)


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _write_gzip_text(path: Path, text: str) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as file_obj:
        file_obj.write(text)


if __name__ == "__main__":
    unittest.main()
