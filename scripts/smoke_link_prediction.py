"""Smoke-test classical candidate-limited link prediction on local OGB data."""

from __future__ import annotations

import argparse
import sys
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
    normalize_method,
)
from src.algorithms.scoring import DatasetCandidateScores, score_multiple_methods_for_dataset
from src.graph.dataset_registry import list_supported_datasets, require_supported_dataset


DEFAULT_SMOKE_TRAIN_EDGE_LIMIT = 50_000


def main() -> int:
    args = parse_args()
    dataset_names = [require_supported_dataset(name) for name in args.datasets]
    methods = args.methods
    train_edge_limit = None if args.full_train_graph else args.limit_train_edges
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
    except ValueError as exc:
        parser.error(str(exc))
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


def metadata_summary(result: DatasetCandidateScores) -> str:
    metadata = result.graph_metadata
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


if __name__ == "__main__":
    raise SystemExit(main())
