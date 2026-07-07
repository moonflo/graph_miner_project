"""Document-level LLM extraction orchestration."""

from __future__ import annotations

import json
import re
from typing import Any, Protocol

from .json_utils import parse_json_object
from .prompts import SYSTEM_PROMPT, build_extraction_prompt
from .schema import ENTITY_TYPES, EXTRACTION_LIST_FIELDS, JsonDict


class SupportsGenerate(Protocol):
    model: str

    def generate(self, prompt: str, *, system_prompt: str | None = None) -> str:
        ...


class LLMExtractor:
    """Extract raw entities, relations, and triples from one document."""

    def __init__(
        self,
        client: SupportsGenerate,
        *,
        model: str | None = None,
        text_max_chars: int = 6000,
        include_raw_response: bool = True,
    ) -> None:
        self.client = client
        self.model = model or getattr(client, "model", "")
        self.text_max_chars = max(1, int(text_max_chars))
        self.include_raw_response = include_raw_response

    def build_prompt(self, document: JsonDict) -> str:
        text, _ = self._prepared_text(document)
        return build_extraction_prompt(document, text)

    def extract(self, document: JsonDict) -> JsonDict:
        """Return the normalized extraction dictionary for one document."""

        text, truncated = self._prepared_text(document)
        result = self._base_result(document, truncated=truncated)
        result["metadata"]["text_chars"] = len(_normalize_text(document.get("text", "")))

        if not text:
            result["error"] = "empty_text"
            result["metadata"]["empty_text"] = True
            return result

        prompt = build_extraction_prompt(document, text)
        try:
            raw_response = self.client.generate(prompt, system_prompt=SYSTEM_PROMPT)
        except Exception as exc:
            result["error"] = f"llm_error: {type(exc).__name__}: {exc}"
            return result

        if self.include_raw_response:
            result["raw_response"] = raw_response

        parsed = parse_json_object(raw_response)
        if not parsed.ok or parsed.data is None:
            result["error"] = parsed.error
            return result

        entities, entity_errors = _normalize_entities(parsed.data.get("entities"), document)
        relations, relation_errors = _normalize_relations(parsed.data.get("relations"), document)
        triples, triple_errors = _normalize_triples(parsed.data.get("triples"), document)

        if not triples and relations:
            triples = [_triple_from_relation(relation) for relation in relations]
            result["metadata"]["triples_generated_from_relations"] = True

        result["entities"] = entities
        result["relations"] = relations
        result["triples"] = triples

        missing_errors = [
            f"{field} field is missing"
            for field in EXTRACTION_LIST_FIELDS
            if field not in parsed.data
        ]
        errors = missing_errors + entity_errors + relation_errors + triple_errors
        if errors:
            result["error"] = "; ".join(errors[:10])
        return result

    def _prepared_text(self, document: JsonDict) -> tuple[str, bool]:
        text = _normalize_text(document.get("text", ""))
        if len(text) <= self.text_max_chars:
            return text, False
        return text[: self.text_max_chars], True

    def _base_result(self, document: JsonDict, *, truncated: bool) -> JsonDict:
        return {
            "doc_id": _normalize_text(document.get("doc_id", "")),
            "source": _normalize_text(document.get("source", "")),
            "title": _normalize_text(document.get("title", "")),
            "entities": [],
            "relations": [],
            "triples": [],
            "raw_response": "",
            "error": "",
            "metadata": {
                "model": self.model,
                "truncated": truncated,
                "text_max_chars": self.text_max_chars,
            },
        }


def _normalize_entities(value: Any, document: JsonDict) -> tuple[list[JsonDict], list[str]]:
    items, errors = _as_list(value, "entities")
    rows: list[JsonDict] = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            errors.append(f"entities[{index}] is not an object")
            continue
        name = _normalize_text(item.get("name", ""))
        if not name:
            errors.append(f"entities[{index}].name is empty")
            continue
        aliases = item.get("aliases", [])
        if aliases is None:
            aliases = []
        if not isinstance(aliases, list):
            aliases = [aliases]
        rows.append(
            {
                "name": name,
                "type": _entity_type(item.get("type")),
                "aliases": [_normalize_text(alias) for alias in aliases if _normalize_text(alias)],
                "evidence": _normalize_text(item.get("evidence", "")),
                "source_doc_id": _normalize_text(document.get("doc_id", "")),
                "source": _normalize_text(document.get("source", "")),
                "metadata": _metadata(item),
            }
        )
    return rows, errors


def _normalize_relations(value: Any, document: JsonDict) -> tuple[list[JsonDict], list[str]]:
    items, errors = _as_list(value, "relations")
    rows: list[JsonDict] = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            errors.append(f"relations[{index}] is not an object")
            continue
        head = _normalize_text(item.get("head", ""))
        tail = _normalize_text(item.get("tail", ""))
        relation_type = _relation_type(item.get("relation_type", ""))
        if not head or not tail or not relation_type:
            errors.append(f"relations[{index}] is missing head, tail, or relation_type")
            continue
        rows.append(
            {
                "head": head,
                "head_type": _entity_type(item.get("head_type")),
                "tail": tail,
                "tail_type": _entity_type(item.get("tail_type")),
                "relation_type": relation_type,
                "evidence": _normalize_text(item.get("evidence", "")),
                "confidence": _confidence(item.get("confidence", 0.0)),
                "source_doc_id": _normalize_text(document.get("doc_id", "")),
                "source": _normalize_text(document.get("source", "")),
                "metadata": _metadata(item),
            }
        )
    return rows, errors


def _normalize_triples(value: Any, document: JsonDict) -> tuple[list[JsonDict], list[str]]:
    items, errors = _as_list(value, "triples")
    rows: list[JsonDict] = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            errors.append(f"triples[{index}] is not an object")
            continue
        subject = _normalize_text(item.get("subject", ""))
        predicate = _relation_type(item.get("predicate", ""))
        object_value = _normalize_text(item.get("object", ""))
        if not subject or not predicate or not object_value:
            errors.append(f"triples[{index}] is missing subject, predicate, or object")
            continue
        rows.append(
            {
                "subject": subject,
                "predicate": predicate,
                "object": object_value,
                "evidence": _normalize_text(item.get("evidence", "")),
                "source_doc_id": _normalize_text(document.get("doc_id", "")),
                "source": _normalize_text(document.get("source", "")),
            }
        )
    return rows, errors


def _as_list(value: Any, field: str) -> tuple[list[Any], list[str]]:
    if value is None:
        return [], []
    if not isinstance(value, list):
        return [], [f"{field} field is not a list"]
    return value, []


def _triple_from_relation(relation: JsonDict) -> JsonDict:
    return {
        "subject": relation["head"],
        "predicate": relation["relation_type"],
        "object": relation["tail"],
        "evidence": relation.get("evidence", ""),
        "source_doc_id": relation.get("source_doc_id", ""),
        "source": relation.get("source", ""),
    }


def _entity_type(value: Any) -> str:
    entity_type = _normalize_text(value).lower()
    return entity_type if entity_type in ENTITY_TYPES else "other"


def _relation_type(value: Any) -> str:
    relation_type = _normalize_text(value).lower()
    relation_type = re.sub(r"[^a-z0-9_]+", "_", relation_type).strip("_")
    return relation_type


def _confidence(value: Any) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        confidence = 0.0
    return min(1.0, max(0.0, confidence))


def _metadata(item: dict[str, Any]) -> JsonDict:
    metadata = item.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {"raw_metadata": metadata}
    return dict(metadata)


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        value = " ".join(_normalize_text(item) for item in value)
    elif isinstance(value, dict):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True)
    else:
        value = str(value)
    return re.sub(r"\s+", " ", value).strip()
