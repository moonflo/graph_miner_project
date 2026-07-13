"""Small shared helpers for the four acceptance-test entry points."""
from __future__ import annotations

import json
import logging
import platform
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DEFAULT_OUTPUT = ROOT / "output" / "acceptance_test"

def setup(test_name: str, filename: str, output_dir: str) -> tuple[logging.Logger, Path, dict[str, Any]]:
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(filename); logger.handlers.clear(); logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(out / filename, encoding="utf-8")):
        handler.setFormatter(formatter); logger.addHandler(handler)
    return logger, out, {"test_name": test_name, "started_at": datetime.now(timezone.utc).isoformat(), "inputs": {}, "checks": [], "errors": []}

def finish(logger: logging.Logger, out: Path, data: dict[str, Any], json_name: str, status: str, conclusion: str) -> int:
    data.update({"status": status, "conclusion": conclusion, "ended_at": datetime.now(timezone.utc).isoformat()})
    (out / json_name).write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    logger.info("测试结论：%s", conclusion)
    return 0 if status == "PASS" else (1 if status == "FAIL" else 2)

def check(data: dict[str, Any], name: str, actual: Any, expected: Any, passed: bool) -> None:
    data["checks"].append({"name": name, "actual": actual, "expected": expected, "passed": passed})

def env_summary() -> dict[str, Any]:
    usage = shutil.disk_usage(ROOT)
    return {"python": sys.version, "executable": sys.executable, "cwd": str(Path.cwd()), "repository_root": str(ROOT), "platform": platform.platform(), "cpu_count": __import__("os").cpu_count(), "disk_total_bytes": usage.total, "disk_free_bytes": usage.free}

def add_common_args(parser: Any) -> None:
    parser.add_argument("--raw-root", default="data/raw")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
