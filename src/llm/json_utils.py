"""Conservative JSON parsing helpers for model output."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class JsonParseResult:
    data: dict[str, Any] | None
    error: str

    @property
    def ok(self) -> bool:
        return self.data is not None and not self.error


def parse_json_object(text: str) -> JsonParseResult:
    """Parse a JSON object from direct output, fenced output, or surrounding prose."""

    if not text or not text.strip():
        return JsonParseResult(None, "LLM response is empty.")

    candidates = _candidate_json_strings(text)
    last_error = "No JSON object candidate found."
    seen: set[str] = set()

    for candidate in candidates:
        candidate = candidate.strip()
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError as exc:
            last_error = f"JSON decode error at char {exc.pos}: {exc.msg}"
            continue
        if not isinstance(payload, dict):
            last_error = "Parsed JSON is not an object."
            continue
        return JsonParseResult(payload, "")

    return JsonParseResult(None, f"Could not parse JSON object: {last_error}")


def _candidate_json_strings(text: str) -> list[str]:
    stripped = text.strip()
    candidates = [stripped]

    for match in re.finditer(r"```(?:json|JSON)?\s*(.*?)\s*```", stripped, re.DOTALL):
        candidates.append(match.group(1))

    extracted = _extract_first_json_object(stripped)
    if extracted:
        candidates.append(extracted)

    return candidates


def _extract_first_json_object(text: str) -> str | None:
    start = text.find("{")
    if start < 0:
        return None

    depth = 0
    in_string = False
    escaped = False

    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]

    return None
