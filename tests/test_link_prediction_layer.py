from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
import math
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import networkx as nx
import numpy as np

from scripts import (
    eval_ogb_official,
    run_collab_experiments,
    run_final_link_prediction_report,
    smoke_link_prediction,
)
from src.algorithms.evaluation import (
    evaluate_ogb_style,
    hits_at_k_from_scores,
    mrr_from_citation2_scores,
)
from src.algorithms.link_prediction import (
    METHOD_TO_NETWORKX_FUNCTION,
    normalize_method,
    score_candidate_pairs,
    score_candidate_pairs_in_order,
    to_simple_undirected_for_topology,
)
from src.algorithms.scoring import (
    DatasetCandidateScores,
    OfficialOGBResult,
    score_candidates_for_dataset,
    score_multiple_methods_for_dataset,
    score_ogb_official_for_dataset,
    score_ogb_official_multiple_methods,
)
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

    def test_weighted_methods_return_deterministic_scores(self) -> None:
        graph = nx.Graph()
        graph.add_edge(0, 1, weight=2.0)
        graph.add_edge(2, 1, weight=4.0)
        graph.add_edge(0, 3, weight=6.0)
        graph.add_edge(2, 3, weight=2.0)

        methods = [
            "common_neighbors",
            "weighted_common_neighbors",
            "weighted_resource_allocation",
            "weighted_adamic_adar",
        ]
        scores = {
            method: score_candidate_pairs_in_order(graph, [(0, 2)], method)[0].score
            for method in methods
        }

        self.assertEqual(scores["common_neighbors"], 2.0)
        self.assertEqual(scores["weighted_common_neighbors"], 7.0)
        self.assertAlmostEqual(scores["weighted_resource_allocation"], 1.0)
        expected_adamic = (3.0 / math.log1p(6.0)) + (4.0 / math.log1p(8.0))
        self.assertAlmostEqual(scores["weighted_adamic_adar"], expected_adamic)

    def test_time_decay_methods_use_year_and_degrade_without_year(self) -> None:
        graph = nx.Graph()
        graph.graph["max_train_year"] = 2019
        graph.add_edge(0, 1, weight=2.0, max_year=2019)
        graph.add_edge(2, 1, weight=4.0, max_year=2018)

        decayed_cn = score_candidate_pairs_in_order(
            graph,
            [(0, 2)],
            "time_decay_common_neighbors",
            decay=0.5,
        )[0]
        decayed_ra = score_candidate_pairs_in_order(
            graph,
            [(0, 2)],
            "time_decay_resource_allocation",
            decay=0.5,
        )[0]

        no_year_graph = nx.Graph()
        no_year_graph.add_edge(0, 1, weight=2.0)
        no_year_graph.add_edge(2, 1, weight=4.0)
        degraded = score_candidate_pairs_in_order(
            no_year_graph,
            [(0, 2)],
            "time_decay_common_neighbors",
            decay=0.5,
        )[0]

        self.assertEqual(decayed_cn.score, 2.0)
        self.assertEqual(decayed_ra.score, 0.5)
        self.assertEqual(degraded.score, 3.0)
        self.assertEqual(normalize_method("cn"), "common_neighbors")


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

    def test_multiple_method_scoring_reuses_loaded_split_and_graph(self) -> None:
        split = OGBSplitData(
            dataset_name="ogbl_collab",
            ogb_name="ogbl-collab",
            num_nodes=4,
            train_edges=np.asarray([[0, 1], [1, 2]], dtype=np.int64),
            valid_edges=np.asarray([[0, 2]], dtype=np.int64),
            test_edges=np.empty((0, 2), dtype=np.int64),
            valid_neg_edges=np.asarray([[0, 3]], dtype=np.int64),
        )
        graph = nx.Graph()
        graph.add_edges_from([(0, 1), (1, 2)])

        with (
            patch("src.algorithms.scoring.load_ogb_split", return_value=split) as load_mock,
            patch(
                "src.algorithms.scoring.build_networkx_graph_from_train_split",
                return_value=graph,
            ) as build_mock,
        ):
            results = score_multiple_methods_for_dataset(
                "ogbl_collab",
                ["jaccard", "common_neighbors"],
                "valid",
                raw_root="unused",
                limit_pos=1,
                limit_neg_per_pos=1,
            )

        self.assertEqual([result.method for result in results], ["jaccard", "common_neighbors"])
        load_mock.assert_called_once()
        build_mock.assert_called_once()

    def test_limited_positive_split_tracks_negative_budget_metadata(self) -> None:
        split = OGBSplitData(
            dataset_name="ogbl_collab",
            ogb_name="ogbl-collab",
            num_nodes=8,
            train_edges=np.asarray([[0, 1], [1, 2]], dtype=np.int64),
            valid_edges=np.asarray([[0, 2], [0, 3], [1, 3]], dtype=np.int64),
            test_edges=np.empty((0, 2), dtype=np.int64),
            valid_neg_edges=np.asarray(
                [[0, 4], [0, 5], [1, 4], [2, 4], [3, 5], [4, 6], [5, 7]],
                dtype=np.int64,
            ),
        )

        with patch("src.algorithms.scoring.load_ogb_split", return_value=split):
            result = score_candidates_for_dataset(
                "ogbl_collab",
                "jaccard",
                "valid",
                raw_root="unused",
                limit_pos=2,
                limit_neg_per_pos=3,
            )

        self.assertFalse(result.positive_split_full)
        self.assertEqual(result.positive_count, 2)
        self.assertEqual(result.requested_negative_count, 6)
        self.assertEqual(result.available_negative_count, 7)
        self.assertEqual(result.negative_count, 6)
        self.assertFalse(result.negative_truncated)

    def test_full_positive_split_ignores_limit_and_reports_truncation(self) -> None:
        split = OGBSplitData(
            dataset_name="ogbl_collab",
            ogb_name="ogbl-collab",
            num_nodes=8,
            train_edges=np.asarray([[0, 1], [1, 2]], dtype=np.int64),
            valid_edges=np.asarray([[0, 2], [0, 3], [1, 3]], dtype=np.int64),
            test_edges=np.empty((0, 2), dtype=np.int64),
            valid_neg_edges=np.asarray(
                [[0, 4], [0, 5], [1, 4], [2, 4], [3, 5]],
                dtype=np.int64,
            ),
        )

        with patch("src.algorithms.scoring.load_ogb_split", return_value=split):
            result = score_candidates_for_dataset(
                "ogbl_collab",
                "jaccard",
                "valid",
                raw_root="unused",
                limit_pos=1,
                limit_neg_per_pos=2,
                full_positive_split=True,
            )

        self.assertTrue(result.positive_split_full)
        self.assertEqual(result.positive_count, 3)
        self.assertEqual(result.requested_negative_count, 6)
        self.assertEqual(result.available_negative_count, 5)
        self.assertEqual(result.negative_count, 5)
        self.assertTrue(result.negative_truncated)

    def test_official_collab_scoring_preserves_rowwise_negative_shape(self) -> None:
        split = OGBSplitData(
            dataset_name="ogbl_collab",
            ogb_name="ogbl-collab",
            num_nodes=7,
            train_edges=np.asarray([[0, 1], [1, 2], [2, 3]], dtype=np.int64),
            valid_edges=np.asarray([[0, 2], [0, 3], [1, 3]], dtype=np.int64),
            test_edges=np.empty((0, 2), dtype=np.int64),
            valid_edge_neg=np.asarray(
                [
                    [[0, 4], [0, 5]],
                    [[0, 6], [1, 4]],
                    [[1, 5], [2, 6]],
                ],
                dtype=np.int64,
            ),
            valid_neg_edges=np.asarray(
                [[0, 4], [0, 5], [0, 6], [1, 4], [1, 5], [2, 6]],
                dtype=np.int64,
            ),
        )
        fake_evaluator = FakeEvaluator({"hits@50": 0.75})

        with (
            patch("src.algorithms.scoring.load_ogb_split", return_value=split) as load_mock,
            patch(
                "src.algorithms.scoring.build_networkx_graph_from_train_split",
                wraps=lambda *args, **kwargs: nx.Graph([(0, 1), (1, 2), (2, 3)]),
            ) as build_mock,
            patch("src.algorithms.scoring._make_ogb_evaluator", return_value=fake_evaluator),
        ):
            results = score_ogb_official_multiple_methods(
                "ogbl_collab",
                ["adamic_adar", "time_decay_common_neighbors"],
                "valid",
                raw_root="unused",
                limit_pos=2,
                limit_neg_per_pos=1,
                decay=0.8,
                batch_size=1,
            )

        self.assertEqual([result.method for result in results], ["adamic_adar", "time_decay_common_neighbors"])
        self.assertEqual(results[0].pos_used, 2)
        self.assertEqual(results[0].neg_per_pos_used, 1)
        self.assertEqual(results[0].total_neg_used, 2)
        self.assertEqual(results[0].y_pred_pos_shape, (2,))
        self.assertEqual(results[0].y_pred_neg_shape, (2, 1))
        self.assertEqual(results[0].negative_layout, "per_positive")
        self.assertEqual(fake_evaluator.calls, [((2,), (2, 1)), ((2,), (2, 1))])
        load_mock.assert_called_once()
        build_mock.assert_called_once()

    def test_official_collab_strict_mode_rejects_2d_edge_neg(self) -> None:
        split = OGBSplitData(
            dataset_name="ogbl_collab",
            ogb_name="ogbl-collab",
            num_nodes=5,
            train_edges=np.asarray([[0, 1], [1, 2]], dtype=np.int64),
            valid_edges=np.asarray([[0, 2], [1, 3]], dtype=np.int64),
            test_edges=np.empty((0, 2), dtype=np.int64),
            valid_edge_neg=np.asarray([[0, 4], [1, 4]], dtype=np.int64),
            valid_neg_edges=np.asarray([[0, 4], [1, 4]], dtype=np.int64),
        )

        with (
            patch("src.algorithms.scoring.load_ogb_split", return_value=split),
            patch(
                "src.algorithms.scoring._make_ogb_evaluator",
                return_value=FakeEvaluator({"hits@50": 0.0}),
            ),
        ):
            with self.assertRaisesRegex(ValueError, "requires edge_neg shape"):
                score_ogb_official_for_dataset(
                    "ogbl_collab",
                    "adamic_adar",
                    "valid",
                    raw_root="unused",
                    require_per_positive_negatives=True,
                )

    def test_official_collab_shared_pool_fallback_calls_ogb_evaluator(self) -> None:
        split = OGBSplitData(
            dataset_name="ogbl_collab",
            ogb_name="ogbl-collab",
            num_nodes=5,
            train_edges=np.asarray([[0, 1], [1, 2]], dtype=np.int64),
            valid_edges=np.asarray([[0, 2], [1, 3]], dtype=np.int64),
            test_edges=np.empty((0, 2), dtype=np.int64),
            valid_edge_neg=np.asarray([[0, 4], [1, 4], [2, 4]], dtype=np.int64),
            valid_neg_edges=np.asarray([[0, 4], [1, 4], [2, 4]], dtype=np.int64),
        )
        fake_evaluator = FakeEvaluator({"hits@50": 1.0})

        with (
            patch("src.algorithms.scoring.load_ogb_split", return_value=split),
            patch(
                "src.algorithms.scoring.build_networkx_graph_from_train_split",
                wraps=lambda *args, **kwargs: nx.Graph([(0, 1), (1, 2)]),
            ),
            patch("src.algorithms.scoring._make_ogb_evaluator", return_value=fake_evaluator),
        ):
            result = score_ogb_official_for_dataset(
                "ogbl_collab",
                "adamic_adar",
                "valid",
                raw_root="unused",
                limit_pos=2,
                limit_neg_per_pos=2,
            )

        self.assertEqual(result.negative_layout, "shared_pool")
        self.assertIsNone(result.neg_per_pos_used)
        self.assertEqual(result.total_neg_used, 2)
        self.assertEqual(result.y_pred_neg_shape, (2,))
        self.assertEqual(fake_evaluator.calls, [((2,), (2,))])
        self.assertIn("shared negative pool", result.notes)


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


class SmokeLinkPredictionCliTest(unittest.TestCase):
    def test_full_positive_flag_changes_output_summary(self) -> None:
        args = smoke_link_prediction.parse_args(["--full-positive-split", "--limit-pos", "7"])

        self.assertTrue(args.full_positive_split)
        self.assertIn("full", smoke_link_prediction.positive_limit_summary(args))

    def test_default_limit_pos_is_labeled_as_smoke_value(self) -> None:
        args = smoke_link_prediction.parse_args([])

        self.assertFalse(args.full_positive_split)
        self.assertFalse(args.limit_pos_explicit)
        self.assertIn("default smoke-test value", smoke_link_prediction.positive_limit_summary(args))

    def test_smoke_output_labels_legacy_candidate_limited_mode(self) -> None:
        output = io.StringIO()
        with (
            patch("sys.argv", ["smoke_link_prediction.py", "--datasets", "ogbl_collab"]),
            patch("scripts.smoke_link_prediction.score_multiple_methods_for_dataset", return_value=[]),
            redirect_stdout(output),
        ):
            exit_code = smoke_link_prediction.main()

        self.assertEqual(exit_code, 0)
        self.assertIn("Evaluation mode: candidate-limited legacy smoke test", output.getvalue())
        self.assertIn("Not directly comparable with OGB leaderboard.", output.getvalue())


class CollabExperimentRunnerTest(unittest.TestCase):
    def test_runner_writes_reports_with_tiny_mock_grid(self) -> None:
        with TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "reports"
            args = run_collab_experiments.parse_args(
                [
                    "--raw-root",
                    "data/raw",
                    "--positive-limits",
                    "10",
                    "20",
                    "--decays",
                    "0.8",
                    "1.0",
                    "--neg-per-pos",
                    "3",
                    "--output-dir",
                    str(output_dir),
                ]
            )

            with patch(
                "scripts.run_collab_experiments.score_multiple_methods_for_dataset",
                side_effect=fake_collab_scores,
            ) as score_mock:
                rows = run_collab_experiments.run_experiments(args)
                paths = run_collab_experiments.write_outputs(rows, args.output_dir)

            self.assertEqual(score_mock.call_count, 4)
            self.assertEqual(len(rows), 18)
            self.assertTrue(paths["csv"].is_file())
            self.assertTrue(paths["markdown"].is_file())
            self.assertTrue(paths["summary"].is_file())
            self.assertIn("requested_neg", paths["csv"].read_text(encoding="utf-8"))
            self.assertIn(
                "Best method by Hits@50",
                paths["summary"].read_text(encoding="utf-8"),
            )


class FinalLinkPredictionReportTest(unittest.TestCase):
    def test_parse_application_output_captures_metrics_and_fallback_ratio(self) -> None:
        spec = run_final_link_prediction_report.ExperimentSpec(
            name="experiment_3_local_2hop_candidate_pool",
            label="Experiment 3",
            eval_mode="application_candidate",
            limit_pos=10000,
            limit_neg_per_pos=50,
            methods=("adamic_adar", "resource_allocation"),
            raw_output_name="experiment_3_local_2hop_candidate_pool.txt",
            application_neg_sampling="source_fixed_2hop",
            tie_policy="average",
            seed=42,
        )
        output = """
## ogbl_collab
- effective_pos: 10,000
- skipped_pos: 0
- fallback_random_negative_ratio: 0.123456
- resource_allocation: Hits@1=0.401200, Hits@5=0.483600, Hits@10=0.527700, Hits@20=0.574900, Hits@50=0.701300, MRR=0.452193, MeanRank=21.320200, PosZeroRate=0.356000, NegZeroRate=0.065586, AvgTiesWithPos=2.964200, AvgGreaterThanPos=18.838100 graph=235,868 nodes/1,285,465 edges weight=yes year=yes max_train_year=2017 decay=0.9 topology_edges=1,285,465
Completed successfully.
"""

        experiment = run_final_link_prediction_report.parse_experiment_output(spec, output)

        self.assertEqual(experiment["status"], "success")
        self.assertAlmostEqual(experiment["fallback_random_negative_ratio"], 0.123456)
        self.assertEqual(experiment["effective_pos"], 10000)
        self.assertEqual(experiment["skipped_pos"], 0)
        self.assertAlmostEqual(
            experiment["results"]["resource_allocation"]["hits@50"],
            0.701300,
        )
        self.assertEqual(experiment["settings"]["candidate_size"], 51)

    def test_runner_writes_final_reports_with_mocked_subprocess(self) -> None:
        with TemporaryDirectory() as temp_dir:
            args = run_final_link_prediction_report.parse_args(
                [
                    "--output-dir",
                    str(Path(temp_dir) / "final"),
                    "--skip-legacy",
                    "--main-limit-pos",
                    "10",
                    "--random-limit-pos",
                    "5",
                ]
            )
            fake_output = """
- effective_pos: 5
- skipped_pos: 0
- adamic_adar: Hits@1=0.400000, Hits@5=0.500000, Hits@10=0.550000, Hits@20=0.600000, Hits@50=0.700000, MRR=0.450000, MeanRank=20.000000, PosZeroRate=0.300000, NegZeroRate=0.100000, AvgTiesWithPos=1.000000, AvgGreaterThanPos=18.000000 graph=10 nodes/20 edges weight=yes year=yes max_train_year=2019 decay=0.9 topology_edges=20
- resource_allocation: Hits@1=0.410000, Hits@5=0.510000, Hits@10=0.560000, Hits@20=0.610000, Hits@50=0.710000, MRR=0.455000, MeanRank=19.000000, PosZeroRate=0.290000, NegZeroRate=0.090000, AvgTiesWithPos=0.900000, AvgGreaterThanPos=17.000000 graph=10 nodes/20 edges weight=yes year=yes max_train_year=2019 decay=0.9 topology_edges=20
Completed successfully.
"""

            completed = subprocess.CompletedProcess(
                args=["python"],
                returncode=0,
                stdout=fake_output,
                stderr="",
            )
            with patch("scripts.run_final_link_prediction_report.subprocess.run", return_value=completed):
                experiments = run_final_link_prediction_report.run_experiments(
                    args,
                    args.output_dir / "raw_outputs",
                )
                payload = run_final_link_prediction_report.build_json_payload(
                    args,
                    experiments,
                    "2026-07-08T00:00:00+00:00",
                )
                rows = run_final_link_prediction_report.summary_rows(experiments)
                args.output_dir.mkdir(parents=True, exist_ok=True)
                run_final_link_prediction_report.write_csv(
                    args.output_dir / "final_link_prediction_summary.csv",
                    rows,
                )
                (args.output_dir / "final_link_prediction_results.json").write_text(
                    json.dumps(payload, ensure_ascii=False),
                    encoding="utf-8",
                )
                (args.output_dir / "final_link_prediction_report.md").write_text(
                    run_final_link_prediction_report.markdown_report(
                        args,
                        experiments,
                        "2026-07-08T00:00:00+00:00",
                    ),
                    encoding="utf-8",
                )

            self.assertEqual(len(experiments), 2)
            self.assertTrue((args.output_dir / "final_link_prediction_report.md").is_file())
            self.assertTrue((args.output_dir / "final_link_prediction_results.json").is_file())
            self.assertTrue((args.output_dir / "final_link_prediction_summary.csv").is_file())
            self.assertTrue(
                (
                    args.output_dir
                    / "raw_outputs"
                    / "experiment_2_random_candidate_pool.txt"
                ).is_file()
            )
            self.assertIn(
                "Final Link Prediction Experiment Summary",
                (args.output_dir / "final_link_prediction_report.md").read_text(encoding="utf-8"),
            )
            self.assertIn("status", (args.output_dir / "final_link_prediction_summary.csv").read_text(encoding="utf-8"))
            self.assertIn("\"settings\"", json.dumps(payload, ensure_ascii=False))


class OfficialEvalCliTest(unittest.TestCase):
    def test_markdown_table_contains_required_official_columns(self) -> None:
        args = eval_ogb_official.parse_args(
            [
                "--raw-root",
                "data/raw",
                "--limit-pos",
                "2",
                "--limit-neg-per-pos",
                "3",
            ]
        )
        row = eval_ogb_official.row_from_result(
            OfficialOGBResult(
                dataset="ogbl_collab",
                split="valid",
                method="adamic_adar",
                decay=0.8,
                pos_used=2,
                neg_per_pos_used=3,
                total_neg_used=6,
                official_mode=True,
                hits_at_50=0.5,
                graph_metadata={
                    "num_nodes": 5,
                    "num_edges": 2,
                    "has_edge_weight": True,
                    "has_edge_year": True,
                    "max_train_year": 2019,
                },
                y_pred_pos_shape=(2,),
                y_pred_neg_shape=(2, 3),
                edge_neg_shape=(2, 3, 2),
                negative_layout="per_positive",
            )
        )

        markdown = eval_ogb_official.markdown_table([row], args)

        self.assertIn("dataset | split | method | decay", markdown)
        self.assertIn("hits@50", markdown)
        self.assertIn("OGB official-style", markdown)


class FakeEvaluator:
    def __init__(self, result: dict[str, float]) -> None:
        self.result = result
        self.calls: list[tuple[tuple[int, ...], tuple[int, ...]]] = []

    def eval(self, input_dict):
        self.calls.append(
            (
                tuple(int(item) for item in input_dict["y_pred_pos"].shape),
                tuple(int(item) for item in input_dict["y_pred_neg"].shape),
            )
        )
        return self.result


def fake_collab_scores(
    dataset_name,
    methods,
    split,
    raw_root,
    limit_pos,
    limit_neg_per_pos,
    limit_train_edges,
    full_positive_split,
    decay,
    continue_on_error=False,
):
    positive_count = 30 if full_positive_split else int(limit_pos)
    requested_neg = positive_count * int(limit_neg_per_pos)
    available_neg = max(0, requested_neg - 1)
    used_neg = min(requested_neg, available_neg)
    results = []
    for method in methods:
        results.append(
            DatasetCandidateScores(
                dataset_name=dataset_name,
                method=method,
                split=split,
                pos_scores=np.ones(positive_count, dtype=float),
                neg_scores=np.zeros(used_neg, dtype=float),
                metric_name="Hits@50",
                graph_metadata={
                    "num_nodes": 8,
                    "num_edges": 3,
                    "has_edge_weight": True,
                    "has_edge_year": True,
                    "max_train_year": 2019,
                    "topology_num_edges": 3,
                },
                requested_negative_count=requested_neg,
                available_negative_count=available_neg,
                negative_truncated=used_neg < requested_neg,
                positive_split_full=full_positive_split,
            )
        )
    return results


if __name__ == "__main__":
    unittest.main()
