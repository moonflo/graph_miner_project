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
from src.algorithms.link_prediction import METHOD_TO_NETWORKX_FUNCTION
from src.algorithms.scoring import DatasetCandidateScores, score_candidates_for_dataset
from src.graph.dataset_registry import list_supported_datasets, require_supported_dataset


DEFAULT_SMOKE_TRAIN_EDGE_LIMIT = 50_000


def main() -> int:
    args = parse_args()
    dataset_names = [require_supported_dataset(name) for name in args.datasets]
    methods = args.methods
    train_edge_limit = None if args.full_train_graph else args.limit_train_edges

    print("# Classical Link Prediction Smoke Test")
    print()
    print(f"- raw_root: `{args.raw_root}`")
    print(f"- split: {args.split}")
    print(f"- limit_pos: {format_value(args.limit_pos)}")
    print(f"- limit_neg_per_pos: {format_value(args.limit_neg_per_pos)}")
    print(f"- limit_train_edges: {format_value(train_edge_limit)}")
    print(f"- datasets: {', '.join(dataset_names)}")
    print(f"- methods: {', '.join(methods)}")
    print()

    failures = 0
    for dataset_name in dataset_names:
        print(f"## {dataset_name}")
        for method in methods:
            try:
                result = score_candidates_for_dataset(
                    dataset_name=dataset_name,
                    method=method,
                    split=args.split,
                    raw_root=args.raw_root,
                    limit_pos=args.limit_pos,
                    limit_neg_per_pos=args.limit_neg_per_pos,
                    limit_train_edges=train_edge_limit,
                )
                print_result(method, result)
            except Exception as exc:  # noqa: BLE001 - smoke output should stay readable.
                failures += 1
                print(f"- {method}: ERROR {type(exc).__name__}: {exc}")
        print()

    if failures:
        print(f"Completed with {failures} failure(s).")
        return 1
    print("Completed successfully.")
    return 0


def parse_args() -> argparse.Namespace:
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
        choices=sorted(METHOD_TO_NETWORKX_FUNCTION),
    )
    parser.add_argument("--split", choices=("valid", "test"), default="valid")
    parser.add_argument("--limit-pos", type=int, default=100)
    parser.add_argument("--limit-neg-per-pos", type=int, default=100)
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
    return parser.parse_args()


def print_result(method: str, result: DatasetCandidateScores) -> None:
    metric = evaluate_ogb_style(result.dataset_name, result)
    metric_text = ", ".join(f"{name}={value:.6f}" for name, value in metric.items())
    graph_nodes = result.graph_metadata.get("num_nodes")
    graph_edges = result.graph_metadata.get("num_edges")

    if result.dataset_name == "ogbl_citation2":
        print(
            f"- {method}: pos={shape(result.pos_scores)} "
            f"neg_matrix={shape(result.neg_scores_matrix)} {metric_text} "
            f"graph={format_value(graph_nodes)} nodes/{format_value(graph_edges)} edges"
        )
        return

    print(
        f"- {method}: pos={shape(result.pos_scores)} neg={shape(result.neg_scores)} "
        f"{metric_text} "
        f"graph={format_value(graph_nodes)} nodes/{format_value(graph_edges)} edges"
    )


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


if __name__ == "__main__":
    raise SystemExit(main())
