#!/usr/bin/env python
"""Run optional LLM extraction over a processed documents.jsonl file."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.llm.client import LLMClient, MissingLLMConfigError, MockLLMClient
from src.llm.extractor import LLMExtractor


JsonDict = dict[str, Any]


def main() -> None:
    raise SystemExit(run())


def run(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)

    try:
        documents = read_jsonl(input_path)
    except (OSError, ValueError) as exc:
        raise SystemExit(f"Error: {exc}") from exc

    if args.dry_run:
        return run_dry_run(documents, args)

    if args.mock:
        client = MockLLMClient()
        model = args.model or client.model
    else:
        try:
            client = LLMClient.from_env(model_override=args.model)
        except MissingLLMConfigError as exc:
            raise SystemExit(f"Error: {exc}") from exc
        model = client.model

    extractor = LLMExtractor(
        client,
        model=model,
        text_max_chars=args.text_max_chars,
        include_raw_response=args.include_raw_response,
    )
    stats = run_batch(
        documents,
        output_dir=output_dir,
        extractor=extractor,
        input_path=input_path,
        limit=args.limit,
        resume=args.resume,
        sleep_seconds=args.sleep,
    )
    print(
        "LLM extraction complete: "
        f"processed={stats['num_documents_processed']} "
        f"skipped_resume={stats['num_documents_skipped_resume']} "
        f"entities={stats['num_entities']} "
        f"relations={stats['num_relations']} "
        f"triples={stats['num_triples']} "
        f"failed={stats['num_failed']}"
    )
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract raw entities and relations from processed documents.jsonl."
    )
    parser.add_argument("--input", required=True, help="Input documents.jsonl path.")
    parser.add_argument("--output-dir", required=True, help="Directory for extraction outputs.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum new documents to process.")
    parser.add_argument("--resume", action="store_true", help="Skip doc_id values already extracted.")
    parser.add_argument("--sleep", type=float, default=0.0, help="Seconds to sleep between requests.")
    parser.add_argument("--model", default=None, help="Override LLM_MODEL for this run.")
    parser.add_argument("--dry-run", action="store_true", help="Print the first prompt and do not call an API.")
    parser.add_argument("--mock", action="store_true", help="Use deterministic mock output instead of an API.")
    parser.add_argument(
        "--text-max-chars",
        type=int,
        default=6000,
        help="Maximum characters from each document text sent to the LLM.",
    )
    parser.add_argument(
        "--include-raw-response",
        dest="include_raw_response",
        action="store_true",
        default=True,
        help="Store the model's raw response in llm_extractions.jsonl.",
    )
    parser.add_argument(
        "--no-raw-response",
        dest="include_raw_response",
        action="store_false",
        help="Do not store the model's raw response.",
    )
    return parser


def run_dry_run(documents: list[JsonDict], args: argparse.Namespace) -> int:
    if not documents:
        print("No documents found; nothing to dry-run.")
        return 0
    extractor = LLMExtractor(
        MockLLMClient(),
        model=args.model or "dry-run",
        text_max_chars=args.text_max_chars,
        include_raw_response=False,
    )
    print(extractor.build_prompt(documents[0]))
    return 0


def run_batch(
    documents: list[JsonDict],
    *,
    output_dir: Path,
    extractor: LLMExtractor,
    input_path: Path,
    limit: int | None,
    resume: bool,
    sleep_seconds: float,
) -> JsonDict:
    output_dir.mkdir(parents=True, exist_ok=True)
    extraction_path = output_dir / "llm_extractions.jsonl"

    existing: list[JsonDict] = []
    seen_doc_ids: set[str] = set()
    if resume and extraction_path.is_file():
        existing = read_jsonl(extraction_path)
        seen_doc_ids = {
            str(row.get("doc_id", ""))
            for row in existing
            if isinstance(row, dict) and row.get("doc_id")
        }

    new_extractions: list[JsonDict] = []
    skipped_resume = 0

    for document in documents:
        doc_id = str(document.get("doc_id", ""))
        if resume and doc_id and doc_id in seen_doc_ids:
            skipped_resume += 1
            continue
        if limit is not None and len(new_extractions) >= limit:
            break

        extraction = extractor.extract(document)
        new_extractions.append(extraction)
        if doc_id:
            seen_doc_ids.add(doc_id)
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    all_extractions = existing + new_extractions if resume else new_extractions
    write_jsonl_atomic(extraction_path, all_extractions)
    raw_counts = write_raw_outputs(output_dir, all_extractions)

    stats = build_stats(
        input_path=input_path,
        output_dir=output_dir,
        model=extractor.model,
        total_documents=len(documents),
        processed_extractions=new_extractions,
        skipped_resume=skipped_resume,
        raw_counts=raw_counts,
    )
    write_json_atomic(output_dir / "llm_extract_stats.json", stats)
    return stats


def build_stats(
    *,
    input_path: Path,
    output_dir: Path,
    model: str,
    total_documents: int,
    processed_extractions: list[JsonDict],
    skipped_resume: int,
    raw_counts: JsonDict,
) -> JsonDict:
    errors = [
        {"doc_id": row.get("doc_id", ""), "error": row.get("error", "")}
        for row in processed_extractions
        if row.get("error") not in ("", "empty_text")
    ]
    return {
        "input": str(input_path),
        "output_dir": str(output_dir),
        "model": model,
        "num_documents_total": total_documents,
        "num_documents_processed": len(processed_extractions),
        "num_documents_skipped_resume": skipped_resume,
        "num_success": sum(
            1 for row in processed_extractions if row.get("error") in ("", "empty_text")
        ),
        "num_failed": len(errors),
        "num_empty_text": sum(
            1 for row in processed_extractions if row.get("error") == "empty_text"
        ),
        "num_entities": raw_counts["num_entities"],
        "num_relations": raw_counts["num_relations"],
        "num_triples": raw_counts["num_triples"],
        "errors": errors[:100],
    }


def write_raw_outputs(output_dir: Path, extractions: Iterable[JsonDict]) -> JsonDict:
    entities: list[JsonDict] = []
    relations: list[JsonDict] = []
    triples: list[JsonDict] = []

    for extraction in extractions:
        for entity in extraction.get("entities", []):
            if isinstance(entity, dict):
                entities.append(_raw_entity(entity, extraction))
        for relation in extraction.get("relations", []):
            if isinstance(relation, dict):
                relations.append(_raw_relation(relation, extraction))
        for triple in extraction.get("triples", []):
            if isinstance(triple, dict):
                triples.append(_raw_triple(triple, extraction))

    write_jsonl_atomic(output_dir / "entities.raw.jsonl", entities)
    write_jsonl_atomic(output_dir / "relations.raw.jsonl", relations)
    write_jsonl_atomic(output_dir / "triples.raw.jsonl", triples)
    return {
        "num_entities": len(entities),
        "num_relations": len(relations),
        "num_triples": len(triples),
    }


def _raw_entity(entity: JsonDict, extraction: JsonDict) -> JsonDict:
    return {
        "name": entity.get("name", ""),
        "type": entity.get("type", "other"),
        "aliases": entity.get("aliases", []),
        "evidence": entity.get("evidence", ""),
        "source_doc_id": entity.get("source_doc_id", extraction.get("doc_id", "")),
        "source": entity.get("source", extraction.get("source", "")),
        "metadata": entity.get("metadata", {}),
    }


def _raw_relation(relation: JsonDict, extraction: JsonDict) -> JsonDict:
    return {
        "head": relation.get("head", ""),
        "head_type": relation.get("head_type", "other"),
        "tail": relation.get("tail", ""),
        "tail_type": relation.get("tail_type", "other"),
        "relation_type": relation.get("relation_type", ""),
        "evidence": relation.get("evidence", ""),
        "confidence": relation.get("confidence", 0.0),
        "source_doc_id": relation.get("source_doc_id", extraction.get("doc_id", "")),
        "source": relation.get("source", extraction.get("source", "")),
        "metadata": relation.get("metadata", {}),
    }


def _raw_triple(triple: JsonDict, extraction: JsonDict) -> JsonDict:
    return {
        "subject": triple.get("subject", ""),
        "predicate": triple.get("predicate", ""),
        "object": triple.get("object", ""),
        "evidence": triple.get("evidence", ""),
        "source_doc_id": triple.get("source_doc_id", extraction.get("doc_id", "")),
        "source": triple.get("source", extraction.get("source", "")),
    }


def read_jsonl(path: Path) -> list[JsonDict]:
    if not path.is_file():
        raise FileNotFoundError(f"JSONL file does not exist: {path}")
    rows: list[JsonDict] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL in {path} line {line_number}: {exc.msg}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid JSONL in {path} line {line_number}: row is not an object")
        rows.append(payload)
    return rows


def write_jsonl_atomic(path: Path, rows: Iterable[JsonDict]) -> None:
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as file_obj:
        for row in rows:
            file_obj.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            file_obj.write("\n")
    temp_path.replace(path)


def write_json_atomic(path: Path, payload: JsonDict) -> None:
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temp_path.replace(path)


if __name__ == "__main__":
    main()
