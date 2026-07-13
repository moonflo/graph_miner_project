"""Run final all-in-one link prediction experiments and summarize outputs."""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

from src.graph.dataset_registry import require_supported_dataset


NOTE = "application_candidate results are not directly comparable with the OGB leaderboard."
DEFAULT_METHODS = (
    "jaccard",
    "adamic_adar",
    "resource_allocation",
    "preferential_attachment",
)
LEGACY_METHODS = (
    "jaccard",
    "adamic_adar",
    "resource_allocation",
    "preferential_attachment",
    "time_decay_common_neighbors",
    "time_decay_resource_allocation",
)
CSV_COLUMNS = (
    "experiment_name",
    "status",
    "eval_mode",
    "application_neg_sampling",
    "tie_policy",
    "limit_pos",
    "limit_neg_per_pos",
    "candidate_size",
    "method",
    "hits@1",
    "hits@5",
    "hits@10",
    "hits@20",
    "hits@50",
    "mrr",
    "mean_rank",
    "pos_zero_rate",
    "neg_zero_rate",
    "avg_ties_with_pos",
    "avg_greater_than_pos",
    "fallback_random_negative_ratio",
)
METRIC_LABEL_TO_KEY = {
    "Hits@1": "hits@1",
    "Hits@5": "hits@5",
    "Hits@10": "hits@10",
    "Hits@20": "hits@20",
    "Hits@50": "hits@50",
    "MRR": "mrr",
    "MeanRank": "mean_rank",
    "PosZeroRate": "pos_zero_rate",
    "NegZeroRate": "neg_zero_rate",
    "AvgTiesWithPos": "avg_ties_with_pos",
    "AvgGreaterThanPos": "avg_greater_than_pos",
}


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    label: str
    eval_mode: str
    limit_pos: int
    limit_neg_per_pos: int
    methods: tuple[str, ...]
    raw_output_name: str
    application_neg_sampling: str | None = None
    tie_policy: str | None = None
    seed: int | None = None

    @property
    def candidate_size(self) -> int | None:
        if self.eval_mode != "application_candidate":
            return None
        return 1 + self.limit_neg_per_pos


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_output_dir = args.output_dir / "raw_outputs"

    experiments = run_experiments(args, raw_output_dir)
    generated_at = datetime.now(timezone.utc).isoformat()
    payload = build_json_payload(args, experiments, generated_at)
    rows = summary_rows(experiments)

    report_path = args.output_dir / "final_link_prediction_report.md"
    json_path = args.output_dir / "final_link_prediction_results.json"
    csv_path = args.output_dir / "final_link_prediction_summary.csv"

    report_path.write_text(markdown_report(args, experiments, generated_at), encoding="utf-8")
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_csv(csv_path, rows)

    print("Saved final link prediction experiment outputs:")
    print(f"- Markdown: {report_path}")
    print(f"- JSON: {json_path}")
    print(f"- CSV: {csv_path}")
    print(f"- Raw outputs: {raw_output_dir}")
    return 1 if any(exp["status"] == "failed" for exp in experiments) else 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", default="data/raw")
    parser.add_argument("--dataset", default="ogbl_collab")
    parser.add_argument("--split", choices=("valid", "test"), default="valid")
    parser.add_argument("--output-dir", type=Path, default=Path("reports/final_link_prediction"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--random-limit-pos", type=int, default=1000)
    parser.add_argument("--main-limit-pos", type=int, default=10000)
    parser.add_argument("--limit-neg-per-pos", type=int, default=50)
    parser.add_argument("--decay", type=float, default=0.9)
    parser.add_argument("--skip-legacy", action="store_true")
    parser.add_argument("--run-full-valid", action="store_true")
    args = parser.parse_args(argv)
    args.dataset = require_supported_dataset(args.dataset)
    if args.random_limit_pos <= 0 or args.main_limit_pos <= 0:
        parser.error("--random-limit-pos and --main-limit-pos must be positive")
    if args.limit_neg_per_pos < 0:
        parser.error("--limit-neg-per-pos must be non-negative")
    return args


def run_experiments(args: argparse.Namespace, raw_output_dir: Path) -> list[dict[str, Any]]:
    raw_output_dir.mkdir(parents=True, exist_ok=True)
    experiments: list[dict[str, Any]] = []
    for spec in experiment_specs(args):
        print(f"[final-link-prediction] running {spec.name}")
        command = command_for_spec(args, spec)
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        raw_output_file = raw_output_dir / spec.raw_output_name
        raw_output_file.write_text(
            completed.stdout + ("\n\n# STDERR\n" + completed.stderr if completed.stderr else ""),
            encoding="utf-8",
        )
        experiment = parse_experiment_output(spec, completed.stdout)
        experiment["command"] = " ".join(command)
        experiment["raw_output_file"] = str(raw_output_file)
        experiment["returncode"] = completed.returncode
        if completed.returncode != 0:
            experiment["status"] = "failed"
            experiment["error"] = error_summary(completed.stdout, completed.stderr)
        experiments.append(experiment)
        print(f"[final-link-prediction] {spec.name} status={experiment['status']}")
    return experiments


def experiment_specs(args: argparse.Namespace) -> list[ExperimentSpec]:
    specs: list[ExperimentSpec] = []
    if not args.skip_legacy:
        specs.append(
            ExperimentSpec(
                name="experiment_1_legacy_baseline",
                label="Experiment 1: legacy baseline",
                eval_mode="legacy",
                limit_pos=100000,
                limit_neg_per_pos=100,
                methods=LEGACY_METHODS,
                raw_output_name="experiment_1_legacy_baseline.txt",
            )
        )
    specs.extend(
        [
            ExperimentSpec(
                name="experiment_2_random_candidate_pool",
                label="Experiment 2: random candidate pool",
                eval_mode="application_candidate",
                limit_pos=args.random_limit_pos,
                limit_neg_per_pos=args.limit_neg_per_pos,
                methods=DEFAULT_METHODS,
                raw_output_name="experiment_2_random_candidate_pool.txt",
                application_neg_sampling="source_fixed_random",
                tie_policy="average",
                seed=args.seed,
            ),
            ExperimentSpec(
                name="experiment_3_local_2hop_candidate_pool",
                label="Experiment 3: local 2-hop candidate pool",
                eval_mode="application_candidate",
                limit_pos=args.main_limit_pos,
                limit_neg_per_pos=args.limit_neg_per_pos,
                methods=DEFAULT_METHODS,
                raw_output_name="experiment_3_local_2hop_candidate_pool.txt",
                application_neg_sampling="source_fixed_2hop",
                tie_policy="average",
                seed=args.seed,
            ),
        ]
    )
    if args.run_full_valid:
        specs.append(
            ExperimentSpec(
                name="experiment_4_full_valid_2hop",
                label="Experiment 4: full valid 2-hop",
                eval_mode="application_candidate",
                limit_pos=100000,
                limit_neg_per_pos=args.limit_neg_per_pos,
                methods=DEFAULT_METHODS,
                raw_output_name="experiment_4_full_valid_2hop.txt",
                application_neg_sampling="source_fixed_2hop",
                tie_policy="average",
                seed=args.seed,
            )
        )
    return specs


def command_for_spec(args: argparse.Namespace, spec: ExperimentSpec) -> list[str]:
    command = [
        sys.executable,
        "scripts/smoke_link_prediction.py",
        "--raw-root",
        args.raw_root,
        "--datasets",
        args.dataset,
        "--split",
        args.split,
        "--limit-pos",
        str(spec.limit_pos),
        "--limit-neg-per-pos",
        str(spec.limit_neg_per_pos),
        "--full-train-graph",
        "--methods",
        *spec.methods,
        "--decay",
        str(args.decay),
    ]
    if spec.eval_mode == "application_candidate":
        command.extend(
            [
                "--eval-mode",
                "application_candidate",
                "--application-neg-sampling",
                str(spec.application_neg_sampling),
                "--tie-policy",
                str(spec.tie_policy),
                "--seed",
                str(spec.seed),
            ]
        )
    return command


def parse_experiment_output(spec: ExperimentSpec, output: str) -> dict[str, Any]:
    settings = {
        "eval_mode": spec.eval_mode,
        "application_neg_sampling": spec.application_neg_sampling,
        "tie_policy": spec.tie_policy,
        "limit_pos": spec.limit_pos,
        "limit_neg_per_pos": spec.limit_neg_per_pos,
        "candidate_size": spec.candidate_size,
        "full_train_graph": True,
        "seed": spec.seed,
    }
    experiment: dict[str, Any] = {
        "name": spec.name,
        "label": spec.label,
        "status": "success",
        "settings": settings,
        "results": {},
        "fallback_random_negative_ratio": None,
        "effective_pos": None,
        "skipped_pos": None,
    }
    for line in output.splitlines():
        stripped = line.strip()
        if stripped.startswith("- effective_pos:"):
            experiment["effective_pos"] = parse_int(stripped.split(":", 1)[1].strip())
        elif stripped.startswith("- skipped_pos:"):
            experiment["skipped_pos"] = parse_int(stripped.split(":", 1)[1].strip())
        elif stripped.startswith("- candidate_size ="):
            settings["candidate_size"] = parse_int(stripped.rsplit(":", 1)[-1].strip())
        elif stripped.startswith("- fallback_random_negative_ratio:"):
            experiment["fallback_random_negative_ratio"] = parse_float(
                stripped.split(":", 1)[1].strip()
            )
        elif stripped.startswith("- ERROR"):
            experiment["status"] = "failed"
            experiment["error"] = stripped.removeprefix("- ").strip()
        elif stripped.startswith("- ") and ":" in stripped:
            method, values = parse_method_line(stripped)
            if method in spec.methods:
                experiment["results"][method] = values
    return experiment


def parse_method_line(line: str) -> tuple[str | None, dict[str, float | None]]:
    match = re.match(r"^-\s+([a-zA-Z0-9_]+):\s+(.*)$", line)
    if not match:
        return None, {}
    method = match.group(1)
    values: dict[str, float | None] = {key: None for key in METRIC_LABEL_TO_KEY.values()}
    for label, key in METRIC_LABEL_TO_KEY.items():
        value_match = re.search(rf"\b{re.escape(label)}=([0-9.+\-eE]+|N/A)", line)
        if value_match:
            values[key] = parse_float(value_match.group(1))
    return method, values


def build_json_payload(
    args: argparse.Namespace,
    experiments: list[dict[str, Any]],
    generated_at: str,
) -> dict[str, Any]:
    return {
        "metadata": {
            "dataset": args.dataset,
            "split": args.split,
            "raw_root": args.raw_root,
            "seed": args.seed,
            "generated_at": generated_at,
            "note": NOTE,
        },
        "experiments": experiments,
    }


def summary_rows(experiments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for experiment in experiments:
        settings = experiment["settings"]
        fallback_ratio = experiment.get("fallback_random_negative_ratio")
        if not experiment["results"]:
            rows.append(base_summary_row(experiment, settings, "", {}, fallback_ratio))
            continue
        for method, values in experiment["results"].items():
            rows.append(base_summary_row(experiment, settings, method, values, fallback_ratio))
    return rows


def base_summary_row(
    experiment: dict[str, Any],
    settings: dict[str, Any],
    method: str,
    values: dict[str, float | None],
    fallback_ratio: float | None,
) -> dict[str, Any]:
    row = {column: "" for column in CSV_COLUMNS}
    row.update(
        {
            "experiment_name": experiment["name"],
            "status": experiment["status"],
            "eval_mode": settings["eval_mode"],
            "application_neg_sampling": settings.get("application_neg_sampling") or "",
            "tie_policy": settings.get("tie_policy") or "",
            "limit_pos": settings["limit_pos"],
            "limit_neg_per_pos": settings["limit_neg_per_pos"],
            "candidate_size": settings.get("candidate_size") or "",
            "method": method,
            "fallback_random_negative_ratio": format_csv_float(fallback_ratio),
        }
    )
    for key, value in values.items():
        if key in row:
            row[key] = format_csv_float(value)
    return row


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def markdown_report(args: argparse.Namespace, experiments: list[dict[str, Any]], generated_at: str) -> str:
    lines = [
        "# Final Link Prediction Experiment Summary",
        "",
        "## 1. Basic Setting",
        "",
        f"- Dataset: `{args.dataset}`",
        f"- Split: `{args.split}`",
        "- Train graph: `full train graph`",
        f"- Seed: `{args.seed}`",
        f"- Generated at: `{generated_at}`",
        f"- Note: {NOTE}",
        "",
        "## 2. Experiments",
        "",
        "| Experiment | Eval Mode | Negative Sampling | Tie Policy | Limit Pos | Negatives per Positive | Candidate Size |",
        "|---|---|---|---|---:|---:|---:|",
    ]
    for experiment in experiments:
        settings = experiment["settings"]
        lines.append(
            "| "
            + " | ".join(
                markdown_cell(item)
                for item in (
                    experiment["name"],
                    settings["eval_mode"],
                    settings.get("application_neg_sampling") or "",
                    settings.get("tie_policy") or "",
                    settings["limit_pos"],
                    settings["limit_neg_per_pos"],
                    format_optional(settings.get("candidate_size")),
                )
            )
            + " |"
        )
    lines.extend(["", "## 3. Main Results", ""])
    lines.append(
        "| Experiment | Method | Hits@1 | Hits@5 | Hits@10 | Hits@20 | Hits@50 | MRR | MeanRank |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for experiment in experiments:
        if not experiment["results"]:
            lines.append(f"| {markdown_cell(experiment['name'])} | failed |  |  |  |  |  |  |  |")
            continue
        for method, values in experiment["results"].items():
            lines.append(
                "| "
                + " | ".join(
                    markdown_cell(item)
                    for item in (
                        experiment["name"],
                        method,
                        format_metric(values.get("hits@1")),
                        format_metric(values.get("hits@5")),
                        format_metric(values.get("hits@10")),
                        format_metric(values.get("hits@20")),
                        format_metric(values.get("hits@50")),
                        format_metric(values.get("mrr")),
                        format_metric(values.get("mean_rank")),
                    )
                )
                + " |"
            )
    lines.extend(["", "## 4. Diagnostics", ""])
    lines.append(
        "| Experiment | Method | PosZeroRate | NegZeroRate | AvgTiesWithPos | AvgGreaterThanPos | FallbackRandomNegativeRatio |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for experiment in experiments:
        fallback_ratio = experiment.get("fallback_random_negative_ratio")
        for method, values in experiment["results"].items():
            lines.append(
                "| "
                + " | ".join(
                    markdown_cell(item)
                    for item in (
                        experiment["name"],
                        method,
                        format_metric(values.get("pos_zero_rate")),
                        format_metric(values.get("neg_zero_rate")),
                        format_metric(values.get("avg_ties_with_pos")),
                        format_metric(values.get("avg_greater_than_pos")),
                        format_metric(fallback_ratio),
                    )
                )
                + " |"
            )
    lines.extend(
        [
            "",
            "## 5. Short Notes",
            "",
            "- The local 2-hop candidate-pool experiment is the main application-oriented setting.",
            "- Hits@50 should be interpreted together with Hits@1/5/10 and MRR.",
            "- Preferential Attachment may obtain high Hits@50 but low MRR, indicating weak top-ranking ability.",
            "- Adamic-Adar and Resource Allocation are the primary recommended interpretable topology methods for system integration.",
            "",
        ]
    )
    return "\n".join(lines)


def error_summary(stdout: str, stderr: str) -> str:
    for line in stdout.splitlines():
        stripped = line.strip()
        if stripped.startswith("- ERROR"):
            return stripped.removeprefix("- ").strip()
    stripped_stderr = stderr.strip()
    return stripped_stderr.splitlines()[-1] if stripped_stderr else "Experiment failed."


def parse_float(value: str) -> float | None:
    if value in {"", "N/A", "None", "full"}:
        return None
    try:
        return float(value.replace(",", ""))
    except ValueError:
        return None


def parse_int(value: str) -> int | None:
    if value in {"", "N/A", "None", "full"}:
        return None
    try:
        return int(value.replace(",", ""))
    except ValueError:
        return None


def format_csv_float(value: Any) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return str(value)


def format_metric(value: Any) -> str:
    if value is None:
        return "N/A"
    try:
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return str(value)


def format_optional(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def markdown_cell(value: Any) -> str:
    return str(value).replace("\n", " ").replace("|", "\\|")


if __name__ == "__main__":
    raise SystemExit(main())
