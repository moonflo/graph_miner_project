"""Prompts for document-level entity and relation extraction."""

from __future__ import annotations

import json
from typing import Any


SYSTEM_PROMPT = (
    "You are an information extraction engine. Return only strict JSON. "
    "Do not return Markdown, comments, explanations, or prose outside JSON."
)


def build_extraction_prompt(document: dict[str, Any], text: str) -> str:
    """Build the extraction prompt for one normalized document."""

    schema = {
        "entities": [
            {
                "name": "entity name",
                "type": "person | organization | location | event | concept | object | time | other",
                "aliases": [],
                "evidence": "verbatim source text span",
            }
        ],
        "relations": [
            {
                "head": "head entity name",
                "head_type": "entity type",
                "tail": "tail entity name",
                "tail_type": "entity type",
                "relation_type": "stable_edge_type",
                "evidence": "verbatim source text span",
                "confidence": 0.0,
            }
        ],
        "triples": [
            {
                "subject": "head entity name",
                "predicate": "stable_edge_type",
                "object": "tail entity name",
                "evidence": "verbatim source text span",
            }
        ],
    }

    return (
        "Extract entities, relations, and triples from the document text.\n\n"
        "Rules:\n"
        "1. Extract only information explicitly supported by the text.\n"
        "2. Do not add facts from common knowledge or outside context.\n"
        "3. Do not invent entities, relations, aliases, or evidence.\n"
        "4. Every evidence value must be copied from the document text.\n"
        "5. Return only one strict JSON object that can be parsed by json.loads.\n"
        "6. Do not return Markdown fences or explanatory text.\n"
        "7. If nothing can be extracted, return empty arrays.\n"
        "8. Keep relation_type short and stable for graph edge labels.\n"
        "9. Avoid relation_type values such as related_to unless no better explicit type exists.\n"
        "10. confidence must be a number from 0 to 1. Lower it when uncertain.\n\n"
        "Required JSON shape:\n"
        f"{json.dumps(schema, ensure_ascii=False, indent=2)}\n\n"
        "Document metadata:\n"
        f"doc_id: {document.get('doc_id', '')}\n"
        f"title: {document.get('title', '')}\n"
        f"source: {document.get('source', '')}\n\n"
        "Document text:\n"
        f"{text}\n"
    )
