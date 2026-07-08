"""Run OGB Evaluator-backed official-style evaluation for ogbl-collab."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

from src.algorithms.link_prediction import normalize_method
from src.algorithms.scoring import (
    OfficialOGBResult,
    score_ogb_official_multiple_methods,
)
from src.graph.dataset_registry import require_supported_dataset


DEFAULT_METHODS = (
    "adamic_adar",
    "resource_allocation",
    "time_decay_common_neighbors",
    "time_decay_resource_allocation",
)

RESULT_COLUMNS = (
    "dataset",
    "split",
    "method",
    "decay",
    "pos_used",
    "neg_per_pos_used",
    "total_neg_used",
    "hits@50",
    "nodes",
    "edges",
    "has_weight",
    "has_year",
    "max_train_year",
    "runtime_seconds",
    "official_mode",
    "official_negatives_available",
    "edge_neg_shape",
    "negative_layout",
    "y_pred_pos_shape",
    "y_pred_neg_shape",
    "error",
)


def main() -> int:
    args = parse_args()
    try:
        results = score_ogb_official_multiple_methods(
            dataset_name=args.dataset,
            methods=args.methods,
            split_name=args.split,
            raw_root=args.raw_root,
            full_train_graph=args.full_train_graph,
            limit_pos=args.limit_pos,
            limit_neg_per_pos=args.limit_neg_per_pos,
            decay=args.decay,
            batch_size=args.batch_size,
            require_per_positive_negatives=args.strict_per_positive_negatives,
            continue_on_error=True,
        )
    except (FileNotFoundError, ValueError, ImportError) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    rows = [row_from_result(result) for result in results]
    write_outputs(rows, args.output, args.csv_output, args)
    print(markdown_table(rows, args))
    print(f"Saved Markdown: {args.output}")
    print(f"Saved CSV: {args.csv_output}")
    return 1 if any(row.get("error") for row in rows) else 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", default="data/raw")
    parser.add_argument("--dataset", default="ogbl_collab")
    parser.add_argument("--split", choices=("valid", "test"), default="valid")
    parser.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS))
    parser.add_argument("--decay", type=float, default=0.8)
    parser.add_argument("--limit-pos", type=int, default=1000000)
    parser.add_argument("--limit-neg-per-pos", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument(
        "--full-train-graph",
        action="store_true",
        default=True,
        help="Use all official train_edges. This is the default for official mode.",
    )
    parser.add_argument(
        "--strict-per-positive-negatives",
        action="store_true",
        help=(
            "Require edge_neg shape [num_pos, num_neg, 2]. Without this flag, "
            "2D ogbl-collab shared-pool negatives use the installed OGB "
            "Hits@50 evaluator fallback."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/ogb_official_eval.md"),
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=Path("reports/ogb_official_eval.csv"),
    )
    args = parser.parse_args(argv)
    args.dataset = require_supported_dataset(args.dataset)
    if args.dataset != "ogbl_collab":
        parser.error("official evaluation currently supports only ogbl_collab")
    args.methods = [normalize_method(method) for method in args.methods]
    return args


def row_from_result(result: OfficialOGBResult) -> dict[str, str]:
    metadata = result.graph_metadata
    return {
        "dataset": result.dataset,
        "split": result.split,
        "method": result.method,
        "decay": format_float(result.decay),
        "pos_used": format_int(result.pos_used),
        "neg_per_pos_used": format_optional_int(result.neg_per_pos_used),
        "total_neg_used": format_int(result.total_neg_used),
        "hits@50": format_float(result.hits_at_50),
        "nodes": format_optional_int(metadata.get("num_nodes")),
        "edges": format_optional_int(metadata.get("num_edges")),
        "has_weight": bool_text(metadata.get("has_edge_weight")),
        "has_year": bool_text(metadata.get("has_edge_year")),
        "max_train_year": format_optional_int(metadata.get("max_train_year")),
        "runtime_seconds": f"{result.runtime_seconds:.3f}",
        "official_mode": bool_text(result.official_mode),
        "official_negatives_available": bool_text(result.official_negatives_available),
        "edge_neg_shape": shape_text(result.edge_neg_shape),
        "negative_layout": result.negative_layout,
        "y_pred_pos_shape": shape_text(result.y_pred_pos_shape),
        "y_pred_neg_shape": shape_text(result.y_pred_neg_shape),
        "error": result.error,
    }


def write_outputs(
    rows: list[dict[str, str]],
    markdown_path: Path,
    csv_path: Path,
    args: argparse.Namespace,
) -> None:
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(markdown_table(rows, args), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows: list[dict[str, str]], args: argparse.Namespace) -> str:
    lines = [
        "# ogbl-collab OGB Official Evaluation",
        "",
        "Evaluation mode: OGB official-style via `ogb.linkproppred.Evaluator`.",
        "Use this report for leaderboard-aligned comparisons before legacy smoke results.",
        "",
        sanity_note(args.split),
        "",
        "| " + " | ".join(RESULT_COLUMNS) + " |",
        "| " + " | ".join("---" for _ in RESULT_COLUMNS) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(markdown_cell(row.get(column, "")) for column in RESULT_COLUMNS)
            + " |"
        )
    lines.append("")
    lines.append(
        "If `negative_layout` is `shared_pool`, the split exposed 2D `edge_neg` "
        "and `y_pred_neg` was passed to the installed OGB Hits evaluator as a "
        "1D shared negative pool, not as per-positive row-wise negatives."
    )
    lines.append("")
    return "\n".join(lines)


def sanity_note(split: str) -> str:
    target = "0.6349" if split == "valid" else "0.6417"
    return (
        "Sanity check: official Adamic-Adar for ogbl-collab is expected near "
        f"`{target}` Hits@50 on the {split} split. Large deviations usually "
        "mean the negative layout, train graph boundary, valid/test visibility, "
        "or undirected projection needs inspection."
    )


def bool_text(value: Any) -> str:
    return "yes" if bool(value) else "no"


def format_optional_int(value: Any) -> str:
    if value is None:
        return "N/A"
    return format_int(value)


def format_int(value: Any) -> str:
    try:
        return str(int(value))
    except (TypeError, ValueError):
        return ""


def format_float(value: Any) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return ""


def shape_text(shape: tuple[int, ...] | None) -> str:
    if shape is None:
        return "N/A"
    return "x".join(str(item) for item in shape)


def markdown_cell(value: Any) -> str:
    return str(value).replace("\n", " ").replace("|", "\\|")


if __name__ == "__main__":
    raise SystemExit(main())
