"""Shared types and constants for the optional LLM extraction layer."""

from __future__ import annotations

from typing import Any


JsonDict = dict[str, Any]

ENTITY_TYPES = {
    "person",
    "organization",
    "location",
    "event",
    "concept",
    "object",
    "time",
    "other",
}

EXTRACTION_LIST_FIELDS = ("entities", "relations", "triples")


def empty_payload() -> JsonDict:
    """Return the minimal extraction payload expected from an LLM."""

    return {"entities": [], "relations": [], "triples": []}
