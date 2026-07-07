from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from src.llm.client import LLMClient, MissingLLMConfigError
from src.llm.extractor import LLMExtractor
from src.llm.json_utils import parse_json_object


ROOT = Path(__file__).resolve().parents[1]


class LLMExtractionSmokeTest(unittest.TestCase):
    def test_dry_run_prints_first_prompt_without_api_key(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            documents = root / "documents.jsonl"
            _write_jsonl(
                documents,
                [
                    {
                        "doc_id": "demo:unknown:00000000",
                        "title": "Demo",
                        "text": "OpenAI released ChatGPT.",
                        "source": "demo",
                        "metadata": {},
                    }
                ],
            )

            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "run_llm_extract.py"),
                    "--input",
                    str(documents),
                    "--output-dir",
                    str(root / "out"),
                    "--dry-run",
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=True,
            )

            self.assertIn("Return only one strict JSON object", completed.stdout)
            self.assertIn("doc_id: demo:unknown:00000000", completed.stdout)
            self.assertFalse((root / "out" / "llm_extractions.jsonl").exists())

    def test_mock_mode_writes_extractions_and_resume_does_not_duplicate_raw_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            documents = root / "documents.jsonl"
            output_dir = root / "out"
            _write_jsonl(
                documents,
                [
                    {
                        "doc_id": "demo:unknown:00000000",
                        "title": "First",
                        "text": "OpenAI released ChatGPT.",
                        "source": "demo",
                        "metadata": {},
                    },
                    {
                        "doc_id": "demo:unknown:00000001",
                        "title": "Second",
                        "text": "OpenAI released ChatGPT again.",
                        "source": "demo",
                        "metadata": {},
                    },
                    {
                        "doc_id": "demo:unknown:00000002",
                        "title": "Empty",
                        "text": "",
                        "source": "demo",
                        "metadata": {},
                    },
                ],
            )

            base_command = [
                sys.executable,
                str(ROOT / "scripts" / "run_llm_extract.py"),
                "--input",
                str(documents),
                "--output-dir",
                str(output_dir),
                "--mock",
            ]
            subprocess.run(base_command, cwd=ROOT, capture_output=True, text=True, check=True)
            subprocess.run(
                [*base_command, "--resume"],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=True,
            )

            extractions = _read_jsonl(output_dir / "llm_extractions.jsonl")
            entities = _read_jsonl(output_dir / "entities.raw.jsonl")
            relations = _read_jsonl(output_dir / "relations.raw.jsonl")
            triples = _read_jsonl(output_dir / "triples.raw.jsonl")
            stats = json.loads((output_dir / "llm_extract_stats.json").read_text(encoding="utf-8"))

            self.assertEqual(len(extractions), 3)
            self.assertEqual(len(entities), 4)
            self.assertEqual(len(relations), 2)
            self.assertEqual(len(triples), 2)
            self.assertEqual(extractions[2]["error"], "empty_text")
            self.assertEqual(stats["num_documents_skipped_resume"], 3)
            self.assertEqual(stats["num_entities"], 4)

    def test_code_block_json_can_be_parsed(self) -> None:
        result = parse_json_object(
            '```json\n{"entities": [], "relations": [], "triples": []}\n```'
        )

        self.assertTrue(result.ok)
        self.assertEqual(result.data, {"entities": [], "relations": [], "triples": []})

    def test_invalid_json_is_recorded_as_extraction_error(self) -> None:
        extractor = LLMExtractor(BadJSONClient(), model="bad-json")

        result = extractor.extract(
            {
                "doc_id": "demo:unknown:00000000",
                "title": "Bad",
                "text": "This text has content.",
                "source": "demo",
                "metadata": {},
            }
        )

        self.assertIn("Could not parse JSON object", result["error"])
        self.assertEqual(result["entities"], [])

    def test_missing_api_key_has_clear_error(self) -> None:
        with self.assertRaisesRegex(MissingLLMConfigError, "LLM_API_KEY"):
            LLMClient.from_env(
                env={
                    "LLM_BASE_URL": "https://example.com/v1",
                    "LLM_MODEL": "demo-model",
                }
            )


class BadJSONClient:
    model = "bad-json"

    def generate(self, prompt: str, *, system_prompt: str | None = None) -> str:
        del prompt, system_prompt
        return "This is not JSON."


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as file_obj:
        for row in rows:
            file_obj.write(json.dumps(row, ensure_ascii=False))
            file_obj.write("\n")


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


if __name__ == "__main__":
    unittest.main()
