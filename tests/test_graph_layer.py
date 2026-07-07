from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import networkx as nx
import numpy as np

from src.graph.candidates import (
    candidates_from_split,
    edges_to_node_pairs,
    normalize_edge_array,
    sample_non_edges,
)
from src.graph.dataset_registry import (
    get_dataset_config,
    list_supported_datasets,
    normalize_dataset_name,
    require_supported_dataset,
)
from src.graph.graph_factory import (
    build_networkx_graph_from_processed,
    build_networkx_graph_from_train_split,
    infer_directed,
)
from src.graph.ogb_split_loader import load_ogb_split
from src.graph.processed_loader import (
    load_graph_edges,
    load_graph_nodes,
    load_stats,
    resolve_processed_dataset_dir,
)
from src.graph.schemas import OGBSplitData


class DatasetRegistryTest(unittest.TestCase):
    def test_registry_lists_only_manual_datasets(self) -> None:
        self.assertEqual(
            list_supported_datasets(),
            ["ogbl_citation2", "ogbl_collab", "ogbl_ppa"],
        )

    def test_names_normalize_and_resolve(self) -> None:
        self.assertEqual(normalize_dataset_name("ogbl-citation2"), "ogbl_citation2")
        self.assertEqual(require_supported_dataset("ogbl-collab"), "ogbl_collab")
        self.assertTrue(get_dataset_config("ogbl-citation2").directed)
        self.assertFalse(get_dataset_config("ogbl_ppa").directed)

    def test_unknown_dataset_has_clear_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "Supported datasets"):
            get_dataset_config("ogbl-new")

    def test_registry_does_not_scan_processed_directories(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            processed_root = Path(temp_dir) / "processed"
            (processed_root / "ogbl_new").mkdir(parents=True)

            self.assertEqual(
                list_supported_datasets(),
                ["ogbl_citation2", "ogbl_collab", "ogbl_ppa"],
            )
            with self.assertRaisesRegex(ValueError, "Unsupported dataset"):
                resolve_processed_dataset_dir("ogbl_new", processed_root)


class ProcessedLoaderTest(unittest.TestCase):
    def test_load_nodes_edges_stats_and_limits(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            processed_root = Path(temp_dir) / "processed"
            dataset_dir = make_processed_dataset(processed_root, "ogbl_collab")

            self.assertEqual(
                resolve_processed_dataset_dir("ogbl-collab", processed_root),
                dataset_dir,
            )
            nodes = list(load_graph_nodes("ogbl-collab", processed_root, limit=1))
            edges = list(load_graph_edges("ogbl-collab", processed_root, limit=1))
            stats = load_stats("ogbl-collab", processed_root)

            self.assertEqual(len(nodes), 1)
            self.assertEqual(nodes[0].node_idx, 0)
            self.assertEqual(len(edges), 1)
            self.assertEqual(edges[0].relation, "collaborates_with")
            self.assertEqual(stats["num_graph_nodes"], 3)

    def test_missing_processed_file_has_clear_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            processed_root = Path(temp_dir) / "processed"
            (processed_root / "ogbl_collab").mkdir(parents=True)

            with self.assertRaisesRegex(FileNotFoundError, "graph_nodes.jsonl"):
                list(load_graph_nodes("ogbl_collab", processed_root))


class GraphFactoryTest(unittest.TestCase):
    def test_processed_graph_uses_registry_directed_default(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            processed_root = Path(temp_dir) / "processed"
            make_processed_dataset(processed_root, "ogbl_citation2", relation="cites")

            graph = build_networkx_graph_from_processed(
                "ogbl-citation2",
                processed_root=processed_root,
                limit_edges=1,
                limit_nodes=3,
            )

            self.assertIsInstance(graph, nx.DiGraph)
            self.assertTrue(graph.is_directed())
            self.assertEqual(graph.number_of_edges(), 1)
            self.assertEqual(graph.graph["dataset_name"], "ogbl_citation2")

    def test_processed_graph_can_be_undirected_and_limited(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            processed_root = Path(temp_dir) / "processed"
            make_processed_dataset(processed_root, "ogbl_ppa", relation="associated_with")

            graph = build_networkx_graph_from_processed(
                "ogbl_ppa",
                processed_root=processed_root,
                limit_edges=1,
                limit_nodes=2,
            )

            self.assertIsInstance(graph, nx.Graph)
            self.assertFalse(graph.is_directed())
            self.assertEqual(graph.number_of_edges(), 1)
            self.assertEqual(infer_directed("ogbl_ppa"), False)

    def test_train_split_graph_uses_train_edges_only(self) -> None:
        split = OGBSplitData(
            dataset_name="ogbl_ppa",
            ogb_name="ogbl-ppa",
            num_nodes=4,
            train_edges=np.asarray([[0, 1], [1, 2], [2, 3]], dtype=np.int64),
            valid_edges=np.asarray([[0, 2]], dtype=np.int64),
            test_edges=np.asarray([[0, 3]], dtype=np.int64),
        )

        with patch("src.graph.graph_factory.load_ogb_split", return_value=split):
            graph = build_networkx_graph_from_train_split("ogbl_ppa", limit_edges=2)

        self.assertFalse(graph.is_directed())
        self.assertEqual(graph.number_of_edges(), 2)
        self.assertFalse(graph.has_edge(2, 3))
        self.assertEqual(graph.graph["edge_source"], "official train_edges")


class CandidatesTest(unittest.TestCase):
    def test_sample_non_edges_does_not_return_existing_edges(self) -> None:
        graph = nx.Graph()
        graph.add_edges_from([(0, 1), (1, 2)])

        samples = sample_non_edges(graph, 1, seed=7)

        self.assertEqual(len(samples), 1)
        self.assertFalse(graph.has_edge(*samples[0]))
        self.assertNotEqual(samples[0][0], samples[0][1])

    def test_edge_array_normalization_and_pairs(self) -> None:
        array = normalize_edge_array(np.asarray([[0, 1, 2], [1, 2, 3]]))

        self.assertEqual(array.shape, (3, 2))
        self.assertEqual(edges_to_node_pairs(array, limit=2), [(0, 1), (1, 2)])

    def test_candidates_preserve_citation_target_neg_matrix(self) -> None:
        split = OGBSplitData(
            dataset_name="ogbl_citation2",
            ogb_name="ogbl-citation2",
            num_nodes=4,
            train_edges=np.asarray([[0, 1]], dtype=np.int64),
            valid_edges=np.asarray([[1, 2]], dtype=np.int64),
            test_edges=np.asarray([[2, 3]], dtype=np.int64),
            valid_source_nodes=np.asarray([1], dtype=np.int64),
            valid_target_nodes=np.asarray([2], dtype=np.int64),
            valid_target_node_neg=np.asarray([[0, 3]], dtype=np.int64),
        )

        candidates = candidates_from_split(split, "valid")

        self.assertEqual(candidates.positive_edges.shape, (1, 2))
        self.assertIsNone(candidates.negative_edges)
        self.assertEqual(candidates.target_node_neg.shape, (1, 2))
        self.assertIn("Target-negative matrix", candidates.notes)

    def test_load_ogb_split_requires_existing_raw_dir(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(FileNotFoundError, "Raw directory"):
                load_ogb_split("ogbl_collab", Path(temp_dir) / "raw")


def make_processed_dataset(
    processed_root: Path,
    dataset_name: str,
    *,
    relation: str = "collaborates_with",
) -> Path:
    dataset_dir = processed_root / dataset_name
    dataset_dir.mkdir(parents=True)
    _write_jsonl(
        dataset_dir / "graph_nodes.jsonl",
        [
            node_record(dataset_name, 0),
            node_record(dataset_name, 1),
            node_record(dataset_name, 2),
        ],
    )
    _write_jsonl(
        dataset_dir / "graph_edges.jsonl",
        [
            edge_record(dataset_name, 0, 1, relation),
            edge_record(dataset_name, 1, 2, relation),
        ],
    )
    (dataset_dir / "stats.json").write_text(
        json.dumps(
            {
                "dataset_name": dataset_name,
                "num_graph_nodes": 3,
                "num_graph_edges": 2,
            }
        ),
        encoding="utf-8",
    )
    return dataset_dir


def node_record(dataset_name: str, node_idx: int) -> dict:
    return {
        "id": f"{dataset_name}:node:{node_idx}",
        "label": f"node-{node_idx}",
        "type": "node",
        "weight": 1.0,
        "metadata": {"node_idx": node_idx},
    }


def edge_record(dataset_name: str, source: int, target: int, relation: str) -> dict:
    return {
        "source": f"{dataset_name}:node:{source}",
        "target": f"{dataset_name}:node:{target}",
        "relation": relation,
        "weight": 1.0,
        "metadata": {},
        "evidence_doc_ids": [],
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as file_obj:
        for row in rows:
            file_obj.write(json.dumps(row))
            file_obj.write("\n")


if __name__ == "__main__":
    unittest.main()
