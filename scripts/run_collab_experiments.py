"""Run repeatable ogbl-collab classical link prediction experiments."""

from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

from src.algorithms.evaluation import evaluate_ogb_style
from src.algorithms.scoring import DatasetCandidateScores, score_multiple_methods_for_dataset
from src.graph.dataset_registry import require_supported_dataset


DATASET_NAME = "ogbl_collab"
DEFAULT_TRAIN_EDGE_LIMIT = 50_000
DEFAULT_DECAYS = (0.7, 0.8, 0.85, 0.9, 0.95, 1.0)
DEFAULT_POSITIVE_LIMITS = (1000, 5000, 10000, 20000)
DECAY_SWEEP_POSITIVE_LIMIT = 1000
SCALE_SWEEP_DECAY = 0.8

DECAY_SWEEP_METHODS = (
    "weighted_resource_allocation",
    "weighted_adamic_adar",
    "time_decay_common_neighbors",
    "time_decay_resource_allocation",
)
SCALE_SWEEP_METHODS = (
    "adamic_adar",
    "resource_allocation",
    "weighted_resource_allocation",
    "time_decay_common_neighbors",
    "time_decay_resource_allocation",
)

RESULT_COLUMNS = (
    "experiment_group",
    "dataset",
    "split",
    "method",
    "decay",
    "limit_pos",
    "positive_split_full",
    "pos_used",
    "requested_neg",
    "available_neg",
    "neg_used",
    "negative_truncated",
    "hits_at_50",
    "num_nodes",
    "num_edges",
    "has_weight",
    "has_year",
    "max_train_year",
    "topology_edges",
    "runtime_seconds",
    "error",
)


@dataclass(frozen=True)
class ExperimentConfig:
    experiment_group: str
    methods: tuple[str, ...]
    decay: float
    limit_pos: int | None
    full_positive_split: bool = False


def main() -> int:
    args = parse_args()
    try:
        rows = run_experiments(args)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    paths = write_outputs(rows, args.output_dir)
    print()
    print("Saved collab experiment reports:")
    print(f"- CSV: {paths['csv']}")
    print(f"- Markdown: {paths['markdown']}")
    print(f"- Best summary: {paths['summary']}")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", default="data/raw")
    parser.add_argument("--split", choices=("valid", "test"), default="valid")
    parser.add_argument("--output-dir", type=Path, default=Path("reports/collab_experiments"))
    parser.add_argument(
        "--full-train-graph",
        action="store_true",
        help="Use all official train_edges instead of the smoke train-edge limit.",
    )
    parser.add_argument(
        "--limit-train-edges",
        type=int,
        default=DEFAULT_TRAIN_EDGE_LIMIT,
        help="Train-edge cap used unless --full-train-graph is set.",
    )
    parser.add_argument("--decays", nargs="+", type=float, default=list(DEFAULT_DECAYS))
    parser.add_argument(
        "--positive-limits",
        nargs="+",
        type=int,
        default=list(DEFAULT_POSITIVE_LIMITS),
    )
    parser.add_argument("--neg-per-pos", type=int, default=50)
    parser.add_argument("--skip-decay-sweep", action="store_true")
    parser.add_argument("--skip-scale-sweep", action="store_true")
    parser.add_argument(
        "--include-full-positive",
        action="store_true",
        help="Append one full-positive split run to the scale sweep.",
    )
    args = parser.parse_args(argv)
    if args.skip_decay_sweep and args.skip_scale_sweep:
        parser.error("At least one of decay sweep or scale sweep must run.")
    return args


def run_experiments(args: argparse.Namespace) -> list[dict[str, str]]:
    dataset_name = require_supported_dataset(DATASET_NAME)
    train_edge_limit = None if args.full_train_graph else args.limit_train_edges
    rows: list[dict[str, str]] = []

    for config in iter_experiment_configs(args):
        print(progress_start(config))
        started = time.perf_counter()
        try:
            results = score_multiple_methods_for_dataset(
                dataset_name=dataset_name,
                methods=config.methods,
                split=args.split,
                raw_root=args.raw_root,
                limit_pos=config.limit_pos,
                limit_neg_per_pos=args.neg_per_pos,
                limit_train_edges=train_edge_limit,
                full_positive_split=config.full_positive_split,
                decay=config.decay,
                continue_on_error=True,
            )
        except (FileNotFoundError, ValueError):
            raise
        except Exception as exc:  # noqa: BLE001 - preserve the rest of the experiment grid.
            runtime = time.perf_counter() - started
            config_rows = [
                error_row(args, config, method, runtime, exc) for method in config.methods
            ]
        else:
            runtime = time.perf_counter() - started
            config_rows = [row_from_result(config, result, runtime) for result in results]

        rows.extend(config_rows)
        print(progress_done(config, config_rows))

    return rows


def iter_experiment_configs(args: argparse.Namespace) -> list[ExperimentConfig]:
    configs: list[ExperimentConfig] = []
    if not args.skip_decay_sweep:
        for decay in args.decays:
            configs.append(
                ExperimentConfig(
                    experiment_group="decay_sweep",
                    methods=DECAY_SWEEP_METHODS,
                    decay=decay,
                    limit_pos=DECAY_SWEEP_POSITIVE_LIMIT,
                )
            )

    if not args.skip_scale_sweep:
        for limit_pos in args.positive_limits:
            configs.append(
                ExperimentConfig(
                    experiment_group="scale_sweep",
                    methods=SCALE_SWEEP_METHODS,
                    decay=SCALE_SWEEP_DECAY,
                    limit_pos=limit_pos,
                )
            )
        if args.include_full_positive:
            configs.append(
                ExperimentConfig(
                    experiment_group="scale_sweep",
                    methods=SCALE_SWEEP_METHODS,
                    decay=SCALE_SWEEP_DECAY,
                    limit_pos=None,
                    full_positive_split=True,
                )
            )
    return configs


def row_from_result(
    config: ExperimentConfig,
    result: DatasetCandidateScores,
    runtime_seconds: float,
) -> dict[str, str]:
    metadata = result.graph_metadata
    hits_at_50 = ""
    if not result.error:
        metric = evaluate_ogb_style(result.dataset_name, result)
        hits_at_50 = format_float(metric.get("Hits@50"))

    return {
        "experiment_group": config.experiment_group,
        "dataset": result.dataset_name,
        "split": result.split,
        "method": result.method,
        "decay": format_float(config.decay),
        "limit_pos": limit_pos_label(config),
        "positive_split_full": bool_text(result.positive_split_full),
        "pos_used": format_int(result.positive_count) if not result.error else "",
        "requested_neg": format_optional_int(result.requested_negative_count),
        "available_neg": format_optional_int(result.available_negative_count),
        "neg_used": format_int(result.negative_count) if not result.error else "",
        "negative_truncated": optional_bool_text(result.negative_truncated),
        "hits_at_50": hits_at_50,
        "num_nodes": format_optional_int(metadata.get("num_nodes")),
        "num_edges": format_optional_int(metadata.get("num_edges")),
        "has_weight": bool_text(metadata.get("has_edge_weight")),
        "has_year": bool_text(metadata.get("has_edge_year")),
        "max_train_year": format_optional_int(metadata.get("max_train_year")),
        "topology_edges": format_optional_int(metadata.get("topology_num_edges")),
        "runtime_seconds": f"{runtime_seconds:.3f}",
        "error": result.error,
    }


def error_row(
    args: argparse.Namespace,
    config: ExperimentConfig,
    method: str,
    runtime_seconds: float,
    exc: Exception,
) -> dict[str, str]:
    return {
        "experiment_group": config.experiment_group,
        "dataset": DATASET_NAME,
        "split": args.split,
        "method": method,
        "decay": format_float(config.decay),
        "limit_pos": limit_pos_label(config),
        "positive_split_full": bool_text(config.full_positive_split),
        "pos_used": "",
        "requested_neg": "",
        "available_neg": "",
        "neg_used": "",
        "negative_truncated": "",
        "hits_at_50": "",
        "num_nodes": "",
        "num_edges": "",
        "has_weight": "",
        "has_year": "",
        "max_train_year": "",
        "topology_edges": "",
        "runtime_seconds": f"{runtime_seconds:.3f}",
        "error": f"{type(exc).__name__}: {exc}",
    }


def write_outputs(rows: list[dict[str, str]], output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "collab_experiments.csv"
    markdown_path = output_dir / "collab_experiments.md"
    summary_path = output_dir / "collab_best_summary.md"

    with csv_path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    markdown_path.write_text(markdown_table(rows), encoding="utf-8")
    summary_path.write_text(best_summary(rows), encoding="utf-8")
    return {"csv": csv_path, "markdown": markdown_path, "summary": summary_path}


def markdown_table(rows: list[dict[str, str]]) -> str:
    lines = ["# ogbl-collab Classical Link Prediction Experiments", ""]
    lines.append("| " + " | ".join(RESULT_COLUMNS) + " |")
    lines.append("| " + " | ".join("---" for _ in RESULT_COLUMNS) + " |")
    for row in rows:
        lines.append("| " + " | ".join(markdown_cell(row.get(column, "")) for column in RESULT_COLUMNS) + " |")
    lines.append("")
    return "\n".join(lines)


def best_summary(rows: list[dict[str, str]]) -> str:
    lines = ["# ogbl-collab Best Summary", ""]
    decay_rows = filter_success(rows, experiment_group="decay_sweep")
    scale_rows = filter_success(rows, experiment_group="scale_sweep")

    lines.append("## Best method by Hits@50 for decay sweep")
    lines.append(best_row_sentence(best_row(decay_rows)))
    lines.append("")

    lines.append("## Best decay for time_decay_common_neighbors")
    lines.append(best_row_sentence(best_row(decay_rows, method="time_decay_common_neighbors"), include_decay=True))
    lines.append("")

    lines.append("## Best decay for time_decay_resource_allocation")
    lines.append(best_row_sentence(best_row(decay_rows, method="time_decay_resource_allocation"), include_decay=True))
    lines.append("")

    lines.append("## Best method for each positive limit in scale sweep")
    lines.extend(best_by_limit_table(scale_rows))
    lines.append("")

    lines.append("## Improvement of best time-decay method over adamic_adar and resource_allocation")
    lines.extend(time_decay_improvement_table(scale_rows))
    lines.append("")
    return "\n".join(lines)


def progress_start(config: ExperimentConfig) -> str:
    if config.experiment_group == "decay_sweep":
        return (
            f"[decay-sweep] decay={config.decay} methods={len(config.methods)} "
            f"limit_pos={limit_pos_label(config)}"
        )
    return (
        f"[scale-sweep] limit_pos={limit_pos_label(config)} decay={config.decay} "
        f"methods={len(config.methods)}"
    )


def progress_done(config: ExperimentConfig, rows: list[dict[str, str]]) -> str:
    label = config.experiment_group.replace("_", "-")
    best = best_row(filter_success(rows))
    if best is None:
        return f"[{label}] completed with no successful Hits@50 result"
    return f"[{label}] current best={best['method']} Hits@50={best['hits_at_50']}"


def filter_success(
    rows: list[dict[str, str]],
    *,
    experiment_group: str | None = None,
) -> list[dict[str, str]]:
    filtered = [row for row in rows if row.get("hits_at_50") and not row.get("error")]
    if experiment_group is not None:
        filtered = [row for row in filtered if row.get("experiment_group") == experiment_group]
    return filtered


def best_row(
    rows: list[dict[str, str]],
    *,
    method: str | None = None,
) -> dict[str, str] | None:
    candidates = rows
    if method is not None:
        candidates = [row for row in candidates if row.get("method") == method]
    if not candidates:
        return None
    return max(candidates, key=lambda row: metric_value(row))


def best_row_sentence(row: dict[str, str] | None, *, include_decay: bool = False) -> str:
    if row is None:
        return "No successful rows."
    parts = [
        f"- method: `{row['method']}`",
        f"Hits@50: `{row['hits_at_50']}`",
        f"limit_pos: `{row['limit_pos']}`",
    ]
    if include_decay:
        parts.append(f"decay: `{row['decay']}`")
    return ", ".join(parts)


def best_by_limit_table(rows: list[dict[str, str]]) -> list[str]:
    lines = ["| limit_pos | best_method | hits_at_50 | decay |", "| --- | --- | --- | --- |"]
    for limit_pos in sorted_limit_labels({row["limit_pos"] for row in rows}):
        row = best_row([item for item in rows if item["limit_pos"] == limit_pos])
        if row is None:
            continue
        lines.append(
            f"| {markdown_cell(limit_pos)} | {markdown_cell(row['method'])} | "
            f"{row['hits_at_50']} | {row['decay']} |"
        )
    if len(lines) == 2:
        lines.append("| N/A | N/A | N/A | N/A |")
    return lines


def time_decay_improvement_table(rows: list[dict[str, str]]) -> list[str]:
    lines = [
        "| limit_pos | best_time_decay | time_decay_hits | adamic_adar_hits | "
        "delta_vs_adamic_adar | resource_allocation_hits | delta_vs_resource_allocation |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for limit_pos in sorted_limit_labels({row["limit_pos"] for row in rows}):
        limit_rows = [row for row in rows if row["limit_pos"] == limit_pos]
        time_rows = [row for row in limit_rows if row["method"].startswith("time_decay_")]
        time_best = best_row(time_rows)
        adamic = best_row([row for row in limit_rows if row["method"] == "adamic_adar"])
        resource = best_row([row for row in limit_rows if row["method"] == "resource_allocation"])
        if time_best is None:
            continue
        time_hits = metric_value(time_best)
        adamic_hits = metric_value(adamic) if adamic else None
        resource_hits = metric_value(resource) if resource else None
        lines.append(
            "| "
            + " | ".join(
                (
                    markdown_cell(limit_pos),
                    markdown_cell(time_best["method"]),
                    format_float(time_hits),
                    format_float(adamic_hits),
                    format_delta(time_hits, adamic_hits),
                    format_float(resource_hits),
                    format_delta(time_hits, resource_hits),
                )
            )
            + " |"
        )
    if len(lines) == 2:
        lines.append("| N/A | N/A | N/A | N/A | N/A | N/A | N/A |")
    return lines


def sorted_limit_labels(values: set[str]) -> list[str]:
    def sort_key(value: str) -> tuple[int, int | str]:
        if value == "full":
            return (1, value)
        try:
            return (0, int(value))
        except ValueError:
            return (0, value)

    return sorted(values, key=sort_key)


def metric_value(row: dict[str, str] | None) -> float:
    if row is None:
        return float("-inf")
    try:
        return float(row["hits_at_50"])
    except (KeyError, TypeError, ValueError):
        return float("-inf")


def limit_pos_label(config: ExperimentConfig) -> str:
    if config.full_positive_split or config.limit_pos is None:
        return "full"
    return str(config.limit_pos)


def bool_text(value: Any) -> str:
    return "yes" if bool(value) else "no"


def optional_bool_text(value: bool | None) -> str:
    if value is None:
        return ""
    return bool_text(value)


def format_optional_int(value: Any) -> str:
    if value is None:
        return ""
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


def format_delta(left: float | None, right: float | None) -> str:
    if left is None or right is None or right == float("-inf"):
        return ""
    return format_float(left - right)


def markdown_cell(value: Any) -> str:
    return str(value).replace("\n", " ").replace("|", "\\|")


if __name__ == "__main__":
    raise SystemExit(main())
