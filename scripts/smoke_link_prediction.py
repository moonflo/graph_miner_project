"""Smoke-test classical candidate-limited link prediction on local OGB data."""

from __future__ import annotations

import argparse
import random
import sys
from collections.abc import Hashable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

from src.algorithms.evaluation import evaluate_ogb_style
from src.algorithms.link_prediction import (
    METHOD_TO_NETWORKX_FUNCTION,
    SUPPORTED_LINK_PREDICTION_METHODS,
    LinkPredictionScore,
    normalize_method,
    score_candidate_pairs_in_order,
    to_simple_undirected_for_topology,
)
from src.algorithms.scoring import DatasetCandidateScores, score_multiple_methods_for_dataset
from src.graph.candidates import candidates_from_split, edges_to_node_pairs
from src.graph.dataset_registry import list_supported_datasets, require_supported_dataset
from src.graph.graph_factory import build_networkx_graph_from_train_split
from src.graph.ogb_split_loader import load_ogb_split


DEFAULT_SMOKE_TRAIN_EDGE_LIMIT = 50_000
DEFAULT_APPLICATION_METRICS = (
    "hits@1",
    "hits@5",
    "hits@10",
    "hits@20",
    "hits@50",
    "mrr",
    "mean_rank",
)
APPLICATION_METRIC_LABELS = {
    "hits@1": "Hits@1",
    "hits@5": "Hits@5",
    "hits@10": "Hits@10",
    "hits@20": "Hits@20",
    "hits@50": "Hits@50",
    "mrr": "MRR",
    "mean_rank": "MeanRank",
}
APPLICATION_NEG_SAMPLING_METHODS = (
    "source_fixed_random",
    "source_fixed_2hop",
    "random_pair",
)
TIE_POLICIES = ("optimistic", "average", "pessimistic")


@dataclass(frozen=True)
class ApplicationCandidateBatch:
    dataset_name: str
    split: str
    graph: Any
    topology_graph: Any
    candidate_pairs: list[tuple[Hashable, Hashable]]
    effective_pos: int
    skipped_pos: int
    candidate_size: int
    fallback_random_negatives: int
    graph_metadata: dict[str, Any]


@dataclass(frozen=True)
class ApplicationCandidateResult:
    method: str
    metrics: dict[str, float | None]
    diagnostics: dict[str, float | None]
    effective_pos: int
    skipped_pos: int
    candidate_size: int
    fallback_random_negatives: int
    graph_metadata: dict[str, Any]


@dataclass(frozen=True)
class NegativeSampleResult:
    negatives: list[tuple[Hashable, Hashable]]
    fallback_random_negatives: int = 0


@dataclass(frozen=True)
class ApplicationRankingStats:
    ranks: list[float]
    pos_zero_rate: float | None
    neg_zero_rate: float | None
    avg_ties_with_pos: float | None
    avg_greater_than_pos: float | None


def main() -> int:
    args = parse_args()
    dataset_names = [require_supported_dataset(name) for name in args.datasets]
    methods = args.methods
    train_edge_limit = None if args.full_train_graph else args.limit_train_edges

    if args.eval_mode == "application_candidate":
        return run_application_candidate_mode(args, dataset_names, methods, train_edge_limit)

    return run_legacy_mode(args, dataset_names, methods, train_edge_limit)


def run_legacy_mode(
    args: argparse.Namespace,
    dataset_names: list[str],
    methods: list[str],
    train_edge_limit: int | None,
) -> int:
    positive_limit_text = positive_limit_summary(args)

    print("# Classical Link Prediction Smoke Test")
    print("Evaluation mode: candidate-limited legacy smoke test")
    print("Not directly comparable with OGB leaderboard.")
    print()
    print(f"- raw_root: `{args.raw_root}`")
    print(f"- split: {args.split}")
    print(f"- positive_split: {positive_limit_text}")
    print(f"- limit_neg_per_pos: {format_value(args.limit_neg_per_pos)}")
    print(f"- limit_train_edges: {format_value(train_edge_limit)}")
    print(f"- decay: {args.decay}")
    print(f"- datasets: {', '.join(dataset_names)}")
    print(f"- methods: {', '.join(methods)}")
    print()

    failures = 0
    for dataset_name in dataset_names:
        print(f"## {dataset_name}")
        try:
            results = score_multiple_methods_for_dataset(
                dataset_name=dataset_name,
                methods=methods,
                split=args.split,
                raw_root=args.raw_root,
                limit_pos=args.limit_pos,
                limit_neg_per_pos=args.limit_neg_per_pos,
                limit_train_edges=train_edge_limit,
                full_positive_split=args.full_positive_split,
                decay=args.decay,
            )
            for result in results:
                print_result(result.method, result)
        except Exception as exc:  # noqa: BLE001 - smoke output should stay readable.
            failures += 1
            print(f"- ERROR {type(exc).__name__}: {exc}")
        print()

    if failures:
        print(f"Completed with {failures} failure(s).")
        return 1
    print("Completed successfully.")
    return 0


def run_application_candidate_mode(
    args: argparse.Namespace,
    dataset_names: list[str],
    methods: list[str],
    train_edge_limit: int | None,
) -> int:
    print("# Classical Link Prediction Smoke Test")
    print("Evaluation mode: application_candidate")
    print("Not directly comparable with OGB leaderboard.")
    print("This evaluates per-positive candidate-pool ranking.")
    print()
    print(f"- raw_root: `{args.raw_root}`")
    print(f"- split: {args.split}")
    print(f"- limit_pos: {application_limit_pos_text(args)}")
    print(f"- limit_neg_per_pos: {format_value(args.limit_neg_per_pos)}")
    print(f"- application_neg_sampling: {args.application_neg_sampling}")
    print(f"- tie_policy: {args.tie_policy}")
    print(f"- seed: {args.seed}")
    print(f"- candidate_size = 1 + limit_neg_per_pos: {format_value(1 + args.limit_neg_per_pos)}")
    print(f"- limit_train_edges: {format_value(train_edge_limit)}")
    print(f"- decay: {args.decay}")
    print(f"- datasets: {', '.join(dataset_names)}")
    print(f"- methods: {', '.join(methods)}")
    print(f"- metrics: {', '.join(APPLICATION_METRIC_LABELS[name] for name in args.metrics)}")
    print()

    failures = 0
    for dataset_name in dataset_names:
        print(f"## {dataset_name}")
        try:
            batch = build_application_candidate_batch(
                dataset_name=dataset_name,
                split=args.split,
                raw_root=args.raw_root,
                limit_pos=None if args.full_positive_split else args.limit_pos,
                limit_neg_per_pos=args.limit_neg_per_pos,
                limit_train_edges=train_edge_limit,
                neg_sampling=args.application_neg_sampling,
                seed=args.seed,
                decay=args.decay,
            )
            print(f"- split: {args.split}")
            print(f"- limit_pos: {application_limit_pos_text(args)}")
            print(f"- limit_neg_per_pos: {format_value(args.limit_neg_per_pos)}")
            print(f"- application_neg_sampling: {args.application_neg_sampling}")
            print(f"- tie_policy: {args.tie_policy}")
            print(f"- seed: {args.seed}")
            print(f"- effective_pos: {format_value(batch.effective_pos)}")
            print(f"- skipped_pos: {format_value(batch.skipped_pos)}")
            print(f"- candidate_size = 1 + limit_neg_per_pos: {format_value(batch.candidate_size)}")
            if args.application_neg_sampling == "source_fixed_2hop":
                print(f"- fallback_random_negatives: {format_value(batch.fallback_random_negatives)}")
                print(
                    "- fallback_random_negative_ratio: "
                    f"{format_metric_value(fallback_random_negative_ratio(batch))}"
                )
            for result in score_application_candidate_methods(
                batch=batch,
                methods=methods,
                decay=args.decay,
                metrics=args.metrics,
                tie_policy=args.tie_policy,
            ):
                print_application_result(result, args.metrics)
        except Exception as exc:  # noqa: BLE001 - smoke output should stay readable.
            failures += 1
            print(f"- ERROR {type(exc).__name__}: {exc}")
        print()

    if failures:
        print(f"Completed with {failures} failure(s).")
        return 1
    print("Completed successfully.")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    raw_argv = sys.argv[1:] if argv is None else argv
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", default="data/raw")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list_supported_datasets(),
        help="Registry dataset names, e.g. ogbl_collab ogbl_ppa ogbl_citation2.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=list(METHOD_TO_NETWORKX_FUNCTION),
        choices=sorted((*SUPPORTED_LINK_PREDICTION_METHODS, "all")),
        help="Scoring methods. Use 'all' to run every supported method.",
    )
    parser.add_argument("--decay", type=float, default=0.9)
    parser.add_argument("--split", choices=("valid", "test"), default="valid")
    parser.add_argument(
        "--eval-mode",
        choices=("legacy", "application_candidate"),
        default="legacy",
        help="Evaluation mode. 'legacy' preserves the existing smoke-test protocol.",
    )
    parser.add_argument(
        "--application-neg-sampling",
        choices=APPLICATION_NEG_SAMPLING_METHODS,
        default="source_fixed_random",
        help="Negative sampling strategy for --eval-mode application_candidate.",
    )
    parser.add_argument(
        "--tie-policy",
        choices=TIE_POLICIES,
        default="average",
        help="Tie handling for application_candidate ranking.",
    )
    parser.add_argument(
        "--metrics",
        default=",".join(DEFAULT_APPLICATION_METRICS),
        help=(
            "Comma-separated metrics for application_candidate mode. Supported: "
            f"{', '.join(DEFAULT_APPLICATION_METRICS)}."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--limit-pos",
        type=int,
        default=100,
        help=(
            "Number of positive edges to use. Default is a small smoke-test "
            "value, not full split. Use --full-positive-split to evaluate all "
            "positives."
        ),
    )
    parser.add_argument(
        "--full-positive-split",
        action="store_true",
        help="Use all positive edges in the current split and ignore --limit-pos.",
    )
    parser.add_argument(
        "--limit-neg-per-pos",
        type=int,
        default=100,
        help=(
            "Negative sample budget multiplier. For OGB splits with a global "
            "negative pool, the actual number of negatives is capped by the "
            "available official negatives."
        ),
    )
    parser.add_argument(
        "--limit-train-edges",
        type=int,
        default=DEFAULT_SMOKE_TRAIN_EDGE_LIMIT,
        help="Limit official train_edges for smoke graph construction.",
    )
    parser.add_argument(
        "--full-train-graph",
        action="store_true",
        help="Use all official train_edges instead of the smoke train-edge limit.",
    )
    args = parser.parse_args(raw_argv)
    args.limit_pos_explicit = any(
        item == "--limit-pos" or item.startswith("--limit-pos=") for item in raw_argv
    )
    try:
        args.methods = resolve_methods(args.methods)
        args.metrics = parse_application_metrics(args.metrics)
    except ValueError as exc:
        parser.error(str(exc))
    if args.eval_mode == "application_candidate" and args.limit_neg_per_pos < 0:
        parser.error("--limit-neg-per-pos must be non-negative in application_candidate mode")
    return args


def resolve_methods(raw_methods: list[str]) -> list[str]:
    if raw_methods == ["all"]:
        return list(SUPPORTED_LINK_PREDICTION_METHODS)
    if "all" in raw_methods:
        raise ValueError("Use either '--methods all' or explicit method names, not both.")
    return [normalize_method(method) for method in raw_methods]


def print_result(method: str, result: DatasetCandidateScores) -> None:
    metric = evaluate_ogb_style(result.dataset_name, result)
    metric_text = ", ".join(f"{name}={value:.6f}" for name, value in metric.items())
    graph_nodes = result.graph_metadata.get("num_nodes")
    graph_edges = result.graph_metadata.get("num_edges")
    metadata_text = metadata_summary(result)
    negative_text = negative_summary(result)

    if result.dataset_name == "ogbl_citation2":
        print(
            f"- {method}: pos={shape(result.pos_scores)} "
            f"neg_matrix={shape(result.neg_scores_matrix)} {metric_text} "
            f"graph={format_value(graph_nodes)} nodes/{format_value(graph_edges)} edges "
            f"{metadata_text} {negative_text}"
        )
        return

    print(
        f"- {method}: pos={shape(result.pos_scores)} neg={shape(result.neg_scores)} "
        f"{metric_text} "
        f"graph={format_value(graph_nodes)} nodes/{format_value(graph_edges)} edges "
        f"{metadata_text} {negative_text}"
    )


def build_application_candidate_batch(
    *,
    dataset_name: str,
    split: str,
    raw_root: str,
    limit_pos: int | None,
    limit_neg_per_pos: int,
    limit_train_edges: int | None,
    neg_sampling: str,
    seed: int,
    decay: float,
) -> ApplicationCandidateBatch:
    split_data = load_ogb_split(dataset_name, raw_root)
    candidates = candidates_from_split(split_data, split)  # type: ignore[arg-type]
    graph = build_networkx_graph_from_train_split(
        dataset_name,
        raw_root=raw_root,
        limit_edges=limit_train_edges,
        include_isolated_nodes=True,
        split_data=split_data,
    )
    topology_graph = to_simple_undirected_for_topology(graph, include_isolated_nodes=False)
    positive_pairs = edges_to_node_pairs(candidates.positive_edges, limit=limit_pos)
    rng = random.Random(seed)
    candidate_pairs: list[tuple[Hashable, Hashable]] = []
    skipped_pos = 0
    fallback_random_negatives = 0

    for positive_pair in positive_pairs:
        sample_result = sample_application_negatives(
            graph=graph,
            topology_graph=topology_graph,
            positive_pair=positive_pair,
            count=limit_neg_per_pos,
            neg_sampling=neg_sampling,
            rng=rng,
        )
        negatives = sample_result.negatives
        if len(negatives) < limit_neg_per_pos:
            skipped_pos += 1
            continue
        fallback_random_negatives += sample_result.fallback_random_negatives
        candidate_pairs.extend([positive_pair, *negatives])

    candidate_size = 1 + limit_neg_per_pos
    return ApplicationCandidateBatch(
        dataset_name=dataset_name,
        split=split,
        graph=graph,
        topology_graph=topology_graph,
        candidate_pairs=candidate_pairs,
        effective_pos=len(candidate_pairs) // candidate_size if candidate_size else 0,
        skipped_pos=skipped_pos,
        candidate_size=candidate_size,
        fallback_random_negatives=fallback_random_negatives,
        graph_metadata=application_graph_metadata(graph, topology_graph, decay),
    )


def sample_application_negatives(
    *,
    graph,
    topology_graph,
    positive_pair: tuple[Hashable, Hashable],
    count: int,
    neg_sampling: str,
    rng: random.Random,
) -> NegativeSampleResult:
    if count == 0:
        return NegativeSampleResult([])

    nodes = list(graph.nodes)
    if len(nodes) < 2:
        return NegativeSampleResult([])

    directed = graph.is_directed()
    source, target = positive_pair
    positive_key = candidate_key(source, target, directed)
    used_keys = {positive_key}
    negatives: list[tuple[Hashable, Hashable]] = []

    if neg_sampling == "source_fixed_2hop":
        for candidate_target in source_fixed_two_hop_targets(topology_graph, source, rng):
            if len(negatives) >= count:
                break
            add_valid_negative(
                graph,
                (source, candidate_target),
                directed,
                used_keys,
                negatives,
            )
        before_fallback = len(negatives)
        if len(negatives) < count:
            fill_source_fixed_random_negatives(
                graph,
                nodes,
                source,
                count,
                directed,
                used_keys,
                negatives,
                rng,
            )
        return NegativeSampleResult(
            negatives,
            fallback_random_negatives=len(negatives) - before_fallback,
        )

    if neg_sampling == "source_fixed_random":
        fill_source_fixed_random_negatives(
            graph,
            nodes,
            source,
            count,
            directed,
            used_keys,
            negatives,
            rng,
        )
        return NegativeSampleResult(negatives)

    if neg_sampling == "random_pair":
        fill_random_pair_negatives(
            graph,
            nodes,
            count,
            directed,
            used_keys,
            negatives,
            rng,
        )
        return NegativeSampleResult(negatives)

    raise ValueError(f"Unsupported application negative sampling: {neg_sampling}")


def source_fixed_two_hop_targets(
    topology_graph,
    source: Hashable,
    rng: random.Random,
) -> list[Hashable]:
    if source not in topology_graph:
        return []

    targets: set[Hashable] = set()
    for neighbor in topology_graph.neighbors(source):
        targets.update(topology_graph.neighbors(neighbor))
    targets.discard(source)

    target_list = sorted(targets, key=repr)
    rng.shuffle(target_list)
    return target_list


def fill_source_fixed_random_negatives(
    graph,
    nodes: Sequence[Hashable],
    source: Hashable,
    count: int,
    directed: bool,
    used_keys: set[tuple[Hashable, Hashable]],
    negatives: list[tuple[Hashable, Hashable]],
    rng: random.Random,
) -> None:
    max_attempts = max((count - len(negatives)) * 100, 1000)
    attempts = 0
    while len(negatives) < count and attempts < max_attempts:
        attempts += 1
        add_valid_negative(
            graph,
            (source, rng.choice(nodes)),
            directed,
            used_keys,
            negatives,
        )


def fill_random_pair_negatives(
    graph,
    nodes: Sequence[Hashable],
    count: int,
    directed: bool,
    used_keys: set[tuple[Hashable, Hashable]],
    negatives: list[tuple[Hashable, Hashable]],
    rng: random.Random,
) -> None:
    max_attempts = max((count - len(negatives)) * 100, 1000)
    attempts = 0
    while len(negatives) < count and attempts < max_attempts:
        attempts += 1
        add_valid_negative(
            graph,
            tuple(rng.sample(nodes, 2)),
            directed,
            used_keys,
            negatives,
        )


def add_valid_negative(
    graph,
    candidate: tuple[Hashable, Hashable],
    directed: bool,
    used_keys: set[tuple[Hashable, Hashable]],
    negatives: list[tuple[Hashable, Hashable]],
) -> bool:
    candidate_source, candidate_target = candidate
    if candidate_source == candidate_target:
        return False
    key = candidate_key(candidate_source, candidate_target, directed)
    if key in used_keys:
        return False
    if existing_train_edge(graph, candidate_source, candidate_target):
        return False
    used_keys.add(key)
    negatives.append(candidate)
    return True


def score_application_candidate_methods(
    *,
    batch: ApplicationCandidateBatch,
    methods: Sequence[str],
    decay: float,
    metrics: Sequence[str],
    tie_policy: str,
) -> list[ApplicationCandidateResult]:
    results: list[ApplicationCandidateResult] = []
    for method in methods:
        ranking_stats = empty_application_ranking_stats()
        if batch.candidate_pairs:
            predictions = score_candidate_pairs_in_order(
                batch.graph,
                batch.candidate_pairs,
                method,
                decay=decay,
                topology_graph=batch.topology_graph,
            )
            ranking_stats = application_ranking_stats(
                predictions,
                batch.candidate_size,
                tie_policy,
            )
        results.append(
            ApplicationCandidateResult(
                method=method,
                metrics=application_metrics_from_ranks(ranking_stats.ranks, metrics),
                diagnostics=application_diagnostics_from_stats(ranking_stats),
                effective_pos=batch.effective_pos,
                skipped_pos=batch.skipped_pos,
                candidate_size=batch.candidate_size,
                fallback_random_negatives=batch.fallback_random_negatives,
                graph_metadata=batch.graph_metadata,
            )
        )
    return results


def application_ranking_stats(
    predictions: Sequence[LinkPredictionScore],
    candidate_size: int,
    tie_policy: str,
) -> ApplicationRankingStats:
    ranks: list[float] = []
    pos_zero_count = 0
    neg_zero_count = 0
    neg_score_count = 0
    ties_with_pos_total = 0
    greater_than_pos_total = 0

    for start in range(0, len(predictions), candidate_size):
        candidate_scores = [item.score for item in predictions[start : start + candidate_size]]
        positive_score = candidate_scores[0]
        negative_scores = candidate_scores[1:]
        greater = sum(score > positive_score for score in negative_scores)
        equal = sum(score == positive_score for score in negative_scores)
        if positive_score == 0:
            pos_zero_count += 1
        neg_zero_count += sum(score == 0 for score in negative_scores)
        neg_score_count += len(negative_scores)
        ties_with_pos_total += equal
        greater_than_pos_total += greater
        ranks.append(rank_from_ties(greater, equal, tie_policy))

    rank_count = len(ranks)
    if rank_count == 0:
        return empty_application_ranking_stats()
    return ApplicationRankingStats(
        ranks=ranks,
        pos_zero_rate=pos_zero_count / rank_count,
        neg_zero_rate=neg_zero_count / neg_score_count if neg_score_count else None,
        avg_ties_with_pos=ties_with_pos_total / rank_count,
        avg_greater_than_pos=greater_than_pos_total / rank_count,
    )


def empty_application_ranking_stats() -> ApplicationRankingStats:
    return ApplicationRankingStats(
        ranks=[],
        pos_zero_rate=None,
        neg_zero_rate=None,
        avg_ties_with_pos=None,
        avg_greater_than_pos=None,
    )


def rank_from_ties(greater: int, equal: int, tie_policy: str) -> float:
    if tie_policy == "optimistic":
        return float(1 + greater)
    if tie_policy == "average":
        return 1 + greater + (equal / 2)
    if tie_policy == "pessimistic":
        return float(1 + greater + equal)
    raise ValueError(f"Unsupported tie policy: {tie_policy}")


def application_metrics_from_ranks(
    ranks: Sequence[float],
    metrics: Sequence[str],
) -> dict[str, float | None]:
    if not ranks:
        return {metric: None for metric in metrics}

    rank_count = len(ranks)
    values: dict[str, float | None] = {}
    for metric in metrics:
        if metric.startswith("hits@"):
            cutoff = int(metric.split("@", 1)[1])
            values[metric] = sum(rank <= cutoff for rank in ranks) / rank_count
        elif metric == "mrr":
            values[metric] = sum(1.0 / rank for rank in ranks) / rank_count
        elif metric == "mean_rank":
            values[metric] = sum(ranks) / rank_count
        else:
            raise ValueError(f"Unsupported application metric: {metric}")
    return values


def application_diagnostics_from_stats(
    stats: ApplicationRankingStats,
) -> dict[str, float | None]:
    return {
        "PosZeroRate": stats.pos_zero_rate,
        "NegZeroRate": stats.neg_zero_rate,
        "AvgTiesWithPos": stats.avg_ties_with_pos,
        "AvgGreaterThanPos": stats.avg_greater_than_pos,
    }


def print_application_result(
    result: ApplicationCandidateResult,
    metrics: Sequence[str],
) -> None:
    metric_text = ", ".join(
        f"{APPLICATION_METRIC_LABELS[name]}={format_metric_value(result.metrics[name])}"
        for name in metrics
    )
    diagnostic_text = ", ".join(
        f"{name}={format_metric_value(value)}" for name, value in result.diagnostics.items()
    )
    graph_nodes = result.graph_metadata.get("num_nodes")
    graph_edges = result.graph_metadata.get("num_edges")
    print(
        f"- {result.method}: {metric_text}, {diagnostic_text} "
        f"graph={format_value(graph_nodes)} nodes/{format_value(graph_edges)} edges "
        f"{metadata_summary_from_dict(result.graph_metadata)}"
    )


def candidate_key(
    source: Hashable,
    target: Hashable,
    directed: bool,
) -> tuple[Hashable, Hashable]:
    if directed:
        return source, target
    left, right = sorted((source, target), key=repr)
    return left, right


def existing_train_edge(graph, source: Hashable, target: Hashable) -> bool:
    if graph.has_edge(source, target):
        return True
    if not graph.is_directed() and graph.has_edge(target, source):
        return True
    return False


def application_graph_metadata(graph, topology_graph, decay: float) -> dict[str, Any]:
    return {
        "dataset_name": graph.graph.get("dataset_name"),
        "graph_source": graph.graph.get("source"),
        "num_nodes": graph.number_of_nodes(),
        "num_edges": graph.number_of_edges(),
        "include_isolated_nodes": graph.graph.get("include_isolated_nodes"),
        "has_edge_weight": bool(graph.graph.get("has_edge_weight")),
        "has_edge_year": bool(graph.graph.get("has_edge_year")),
        "max_train_year": graph.graph.get("max_train_year"),
        "decay": decay,
        "topology_num_nodes": topology_graph.number_of_nodes(),
        "topology_num_edges": topology_graph.number_of_edges(),
    }


def parse_application_metrics(raw_metrics: str) -> list[str]:
    metrics = [normalize_application_metric(item) for item in raw_metrics.split(",")]
    if not metrics or any(not metric for metric in metrics):
        raise ValueError("--metrics must contain at least one metric")
    unsupported = [metric for metric in metrics if metric not in DEFAULT_APPLICATION_METRICS]
    if unsupported:
        supported = ", ".join(DEFAULT_APPLICATION_METRICS)
        raise ValueError(f"Unsupported --metrics value(s): {unsupported}. Supported: {supported}")
    return metrics


def normalize_application_metric(metric: str) -> str:
    normalized = metric.strip().lower().replace("-", "_")
    if normalized == "meanrank":
        return "mean_rank"
    return normalized


def metadata_summary(result: DatasetCandidateScores) -> str:
    return metadata_summary_from_dict(result.graph_metadata)


def metadata_summary_from_dict(metadata: dict[str, Any]) -> str:
    return (
        f"weight={yes_no(metadata.get('has_edge_weight'))} "
        f"year={yes_no(metadata.get('has_edge_year'))} "
        f"max_train_year={format_metadata_value(metadata.get('max_train_year'))} "
        f"decay={metadata.get('decay')} "
        f"topology_edges={format_metadata_value(metadata.get('topology_num_edges'))}"
    )


def negative_summary(result: DatasetCandidateScores) -> str:
    return (
        f"positive_split={positive_split_label(result.positive_split_full)} "
        f"requested_neg={format_metadata_value(result.requested_negative_count)} "
        f"available_neg={format_metadata_value(result.available_negative_count)} "
        f"used_neg={format_value(result.negative_count)} "
        f"neg_truncated={yes_no_optional(result.negative_truncated)}"
    )


def positive_limit_summary(args: argparse.Namespace) -> str:
    if args.full_positive_split:
        return "full (ignores --limit-pos)"
    if args.limit_pos_explicit:
        return f"limited, limit_pos={format_value(args.limit_pos)} (not full split)"
    return (
        f"limited, limit_pos={format_value(args.limit_pos)} "
        "(default smoke-test value, not full split)"
    )


def application_limit_pos_text(args: argparse.Namespace) -> str:
    if args.full_positive_split:
        return "full (ignores --limit-pos)"
    return format_value(args.limit_pos)


def fallback_random_negative_ratio(batch: ApplicationCandidateBatch) -> float | None:
    negative_count = batch.effective_pos * max(batch.candidate_size - 1, 0)
    if negative_count == 0:
        return None
    return batch.fallback_random_negatives / negative_count


def positive_split_label(is_full: bool) -> str:
    return "full" if is_full else "limited"


def yes_no(value: Any) -> str:
    return "yes" if bool(value) else "no"


def yes_no_optional(value: bool | None) -> str:
    if value is None:
        return "N/A"
    return yes_no(value)


def shape(value: Any) -> str:
    if value is None:
        return "N/A"
    value_shape = getattr(value, "shape", None)
    if value_shape is None:
        return "N/A"
    return "x".join(f"{int(item):,}" for item in value_shape)


def format_value(value: Any) -> str:
    if value is None:
        return "full"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def format_metadata_value(value: Any) -> str:
    if value is None:
        return "N/A"
    return format_value(value)


def format_metric_value(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:.6f}"


if __name__ == "__main__":
    raise SystemExit(main())
