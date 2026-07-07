from __future__ import annotations

import unittest
from unittest.mock import patch

import networkx as nx
import numpy as np

from src.algorithms.evaluation import (
    evaluate_ogb_style,
    hits_at_k_from_scores,
    mrr_from_citation2_scores,
)
from src.algorithms.link_prediction import (
    METHOD_TO_NETWORKX_FUNCTION,
    score_candidate_pairs,
    to_simple_undirected_for_topology,
)
from src.algorithms.scoring import score_candidates_for_dataset
from src.graph.graph_factory import build_visible_graph
from src.graph.schemas import OGBSplitData


class ClassicalLinkPredictionTest(unittest.TestCase):
    def test_four_methods_return_sorted_candidate_scores(self) -> None:
        graph = nx.Graph()
        graph.add_edges_from([(0, 1), (1, 2), (2, 3)])
        candidates = [(0, 2), (0, 4)]

        for method in METHOD_TO_NETWORKX_FUNCTION:
            with self.subTest(method=method):
                scores = score_candidate_pairs(graph, candidates, method)

                self.assertEqual(len(scores), 2)
                self.assertEqual([score.method for score in scores], [method, method])
                self.assertGreaterEqual(scores[0].score, scores[1].score)

    def test_directed_graph_is_projected_before_networkx_scoring(self) -> None:
        graph = nx.DiGraph()
        graph.add_edges_from([(0, 1), (1, 2)])

        scores = score_candidate_pairs(graph, [(0, 2)], "jaccard")
        topology_graph = to_simple_undirected_for_topology(graph)

        self.assertEqual(len(scores), 1)
        self.assertFalse(topology_graph.is_directed())
        self.assertEqual(topology_graph.graph["topology_projection"], "simple_undirected")

    def test_self_loop_is_removed_before_adamic_adar(self) -> None:
        graph = nx.Graph()
        graph.add_edges_from([(0, 1), (1, 2), (2, 2)])

        scores = score_candidate_pairs(graph, [(0, 2)], "adamic_adar")
        topology_graph = to_simple_undirected_for_topology(graph)

        self.assertEqual(len(scores), 1)
        self.assertEqual(nx.number_of_selfloops(topology_graph), 0)

    def test_candidate_limited_scoring_passes_ebunch(self) -> None:
        graph = nx.path_graph(3)

        def fake_jaccard(scored_graph, ebunch=None):
            self.assertIsNotNone(ebunch)
            return [(source, target, 0.0) for source, target in ebunch]

        with patch("networkx.jaccard_coefficient", side_effect=fake_jaccard) as mocked:
            scores = score_candidate_pairs(graph, [(0, 2)], "jaccard")

        self.assertEqual(len(scores), 1)
        mocked.assert_called_once()

    def test_missing_candidate_endpoint_gets_zero_score(self) -> None:
        graph = nx.Graph()
        graph.add_edge(0, 1)

        scores = score_candidate_pairs(graph, [(7, 8)], "preferential_attachment")

        self.assertEqual(len(scores), 1)
        self.assertEqual(scores[0].score, 0.0)


class OgbAwareScoringTest(unittest.TestCase):
    def test_citation2_scores_preserve_negative_target_matrix(self) -> None:
        split = OGBSplitData(
            dataset_name="ogbl_citation2",
            ogb_name="ogbl-citation2",
            num_nodes=6,
            train_edges=np.asarray([[0, 1], [1, 2], [2, 3]], dtype=np.int64),
            valid_edges=np.asarray([[0, 2], [3, 4]], dtype=np.int64),
            test_edges=np.empty((0, 2), dtype=np.int64),
            valid_source_nodes=np.asarray([0, 3], dtype=np.int64),
            valid_target_nodes=np.asarray([2, 4], dtype=np.int64),
            valid_target_node_neg=np.asarray([[5, 1], [0, 5]], dtype=np.int64),
        )

        with (
            patch("src.algorithms.scoring.load_ogb_split", return_value=split),
            patch("src.graph.graph_factory.load_ogb_split", return_value=split),
        ):
            result = score_candidates_for_dataset(
                "ogbl_citation2",
                "jaccard",
                "valid",
                raw_root="unused",
                limit_pos=2,
                limit_neg_per_pos=2,
            )

        self.assertEqual(result.pos_scores.shape, (2,))
        self.assertEqual(result.neg_scores_matrix.shape, (2, 2))
        self.assertIsNone(result.neg_scores)
        self.assertEqual(result.metric_name, "MRR")
        self.assertIn("target_node_neg", result.notes)

    def test_formal_scoring_includes_isolated_nodes_and_missing_endpoints(self) -> None:
        split = OGBSplitData(
            dataset_name="ogbl_collab",
            ogb_name="ogbl-collab",
            num_nodes=5,
            train_edges=np.asarray([[0, 1]], dtype=np.int64),
            valid_edges=np.asarray([[3, 4]], dtype=np.int64),
            test_edges=np.empty((0, 2), dtype=np.int64),
            valid_neg_edges=np.asarray([[9, 10]], dtype=np.int64),
        )

        with (
            patch("src.algorithms.scoring.load_ogb_split", return_value=split),
            patch("src.graph.graph_factory.load_ogb_split", return_value=split),
        ):
            graph = build_visible_graph(
                "ogbl_collab",
                raw_root="unused",
                include_isolated_nodes=True,
            )
            result = score_candidates_for_dataset(
                "ogbl_collab",
                "jaccard",
                "valid",
                raw_root="unused",
                limit_pos=1,
                limit_neg_per_pos=1,
            )

        self.assertEqual(graph.number_of_nodes(), 5)
        self.assertTrue(result.graph_metadata["include_isolated_nodes"])
        self.assertEqual(result.pos_scores.shape, (1,))
        self.assertEqual(result.neg_scores.shape, (1,))


class EvaluationTest(unittest.TestCase):
    def test_hits_at_k_supports_global_and_rowwise_negatives(self) -> None:
        global_hits = hits_at_k_from_scores(
            np.asarray([0.9, 0.4]),
            np.asarray([0.8, 0.7, 0.1]),
            2,
        )
        rowwise_hits = hits_at_k_from_scores(
            np.asarray([0.8, 0.2]),
            np.asarray([[0.9, 0.7], [0.1, 0.3]]),
            1,
        )

        self.assertEqual(global_hits, 0.5)
        self.assertEqual(rowwise_hits, 0.0)

    def test_citation2_mrr_uses_rowwise_rank(self) -> None:
        mrr = mrr_from_citation2_scores(
            np.asarray([0.8, 0.2]),
            np.asarray([[0.9, 0.7], [0.1, 0.3]]),
        )

        self.assertEqual(mrr, 0.5)

    def test_evaluate_ogb_style_routes_metrics(self) -> None:
        collab = {
            "pos_scores": np.asarray([0.9]),
            "neg_scores": np.asarray([0.1, 0.2]),
        }
        citation = {
            "pos_scores": np.asarray([0.9]),
            "neg_scores_matrix": np.asarray([[0.1, 0.2]]),
        }

        self.assertEqual(evaluate_ogb_style("ogbl_collab", collab), {"Hits@50": 1.0})
        self.assertEqual(evaluate_ogb_style("ogbl_citation2", citation), {"MRR": 1.0})


if __name__ == "__main__":
    unittest.main()
