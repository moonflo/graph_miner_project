"""Smoke-test the graph construction layer on local processed and OGB data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.graph.candidates import candidates_from_split
from src.graph.dataset_registry import (
    get_dataset_config,
    list_supported_datasets,
    require_supported_dataset,
)
from src.graph.graph_factory import (
    build_networkx_graph_from_processed,
    build_networkx_graph_from_train_split,
)
from src.graph.graph_stats import compute_basic_graph_stats, summarize_split
from src.graph.ogb_split_loader import load_ogb_split


def main() -> int:
    args = parse_args()
    dataset_names = (
        [require_supported_dataset(name) for name in args.datasets]
        if args.datasets
        else list_supported_datasets()
    )

    print("# Graph Build Smoke Test")
    print()
    print(f"- processed_root: `{args.processed_root}`")
    print(f"- raw_root: `{args.raw_root}`")
    print(f"- limit_edges: {args.limit_edges:,}")
    print(f"- datasets: {', '.join(dataset_names)}")
    print()

    failures = 0
    for dataset_name in dataset_names:
        print(f"## {dataset_name}")
        try:
            run_dataset_smoke(dataset_name, args.processed_root, args.raw_root, args.limit_edges)
        except Exception as exc:  # noqa: BLE001 - keep smoke output readable.
            failures += 1
            print(f"- ERROR: {type(exc).__name__}: {exc}")
        print()

    if failures:
        print(f"Completed with {failures} dataset failure(s).")
        return 1
    print("Completed successfully.")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-root", default="data/processed")
    parser.add_argument("--raw-root", default="data/raw")
    parser.add_argument("--limit-edges", type=int, default=5_000)
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Optional registry dataset names, e.g. ogbl_collab ogbl_ppa.",
    )
    return parser.parse_args()


def run_dataset_smoke(
    dataset_name: str,
    processed_root: str,
    raw_root: str,
    limit_edges: int,
) -> None:
    config = get_dataset_config(dataset_name)
    print(f"- registry.ogb_name: `{config.ogb_name}`")
    print(f"- registry.directed: {config.directed}")
    print(f"- registry.processed_graph_is_aggregated: {config.processed_graph_is_aggregated}")

    processed_graph = build_networkx_graph_from_processed(
        dataset_name,
        processed_root=processed_root,
        limit_nodes=limit_edges,
        limit_edges=limit_edges,
    )
    processed_stats = compute_basic_graph_stats(processed_graph)
    print("- processed graph:")
    print_stats(processed_stats)
    print(f"  - directed matches registry: {processed_graph.is_directed() == config.directed}")

    train_graph = build_networkx_graph_from_train_split(
        dataset_name,
        raw_root=raw_root,
        limit_edges=limit_edges,
    )
    train_stats = compute_basic_graph_stats(train_graph)
    print("- train visible graph:")
    print_stats(train_stats)
    print(f"  - directed matches registry: {train_graph.is_directed() == config.directed}")

    split_data = load_ogb_split(dataset_name, raw_root=raw_root)
    split_summary = summarize_split(split_data)
    print("- official split:")
    for key in (
        "source",
        "num_nodes",
        "train_edges",
        "valid_edges",
        "test_edges",
        "valid_neg_edges",
        "test_neg_edges",
        "valid_target_node_neg",
        "test_target_node_neg",
    ):
        print(f"  - {key}: {format_value(split_summary[key])}")

    for split_name in ("valid", "test"):
        candidates = candidates_from_split(split_data, split_name)  # type: ignore[arg-type]
        print(f"- {split_name} candidates:")
        print(f"  - positive_edges: {format_value(shape(candidates.positive_edges))}")
        print(f"  - negative_edges: {format_value(shape(candidates.negative_edges))}")
        print(f"  - target_node_neg: {format_value(shape(candidates.target_node_neg))}")
        if candidates.notes:
            print(f"  - notes: {candidates.notes}")


def print_stats(stats: Any) -> None:
    print(f"  - nodes: {stats.num_nodes:,}")
    print(f"  - edges: {stats.num_edges:,}")
    print(f"  - directed: {stats.is_directed}")
    print(f"  - average_degree: {stats.average_degree:.4f}")
    print(f"  - max_degree: {stats.max_degree:,}")
    print(f"  - self_loops: {stats.self_loops:,}")
    if stats.num_components is not None:
        print(f"  - {stats.component_type}: {stats.num_components:,}")
    elif stats.components_skipped:
        print("  - components: skipped")


def shape(value: Any) -> tuple[int, ...] | None:
    if value is None:
        return None
    value_shape = getattr(value, "shape", None)
    if value_shape is None:
        return None
    return tuple(int(item) for item in value_shape)


def format_value(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, tuple):
        return "x".join(f"{item:,}" for item in value)
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
