"""Lightweight optional LLM API client for graph interpretation."""

from __future__ import annotations

import time
from typing import Any

import requests


class LLMClient:
    """Small HTTP client for natural-language explanations of graph results."""

    def __init__(
        self,
        api_key: str | None = None,
        api_url: str | None = None,
        *,
        model: str | None = None,
        provider: str = "openai-compatible",
        timeout: int = 30,
        max_retries: int = 3,
        retry_backoff: float = 1.0,
        temperature: float = 0.2,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        self.api_key = api_key
        self.api_url = api_url
        self.model = model
        self.provider = provider.lower()
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        self.temperature = temperature
        self.extra_headers = extra_headers or {}

    def generate(self, prompt: str) -> str:
        """Generate a natural-language explanation from a prompt."""

        if not self.api_url:
            return self._fallback_response(prompt)

        payload = self._build_payload(prompt)
        headers = self._build_headers()
        last_error: Exception | None = None

        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    self.api_url,
                    json=payload,
                    headers=headers,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                return _parse_generation_response(response.json())
            except (requests.RequestException, KeyError, TypeError, ValueError) as exc:
                last_error = exc
                if attempt == self.max_retries - 1:
                    break
                time.sleep(self.retry_backoff * (2**attempt))

        raise RuntimeError("LLM API request failed") from last_error

    def explain_graph(
        self,
        *,
        graph_description: str,
        nodes: list[str] | None = None,
        edges: list[tuple[str, str, float | None]] | None = None,
        task: str = "Explain the graph structure and possible latent relations.",
    ) -> str:
        """Build a graph explanation prompt and call ``generate``."""

        prompt = build_graph_prompt(
            graph_description=graph_description,
            nodes=nodes or [],
            edges=edges or [],
            task=task,
        )
        return self.generate(prompt)

    def _build_payload(self, prompt: str) -> dict[str, Any]:
        if self.provider in {"openai", "qwen", "openai-compatible"}:
            payload: dict[str, Any] = {
                "messages": [{"role": "user", "content": prompt}],
                "temperature": self.temperature,
            }
            if self.model:
                payload["model"] = self.model
            return payload

        payload = {"prompt": prompt, "temperature": self.temperature}
        if self.model:
            payload["model"] = self.model
        return payload

    def _build_headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json", **self.extra_headers}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _fallback_response(self, prompt: str) -> str:
        first_line = prompt.strip().splitlines()[0] if prompt.strip() else "No prompt provided."
        return f"LLM explanation is disabled. Prompt summary: {first_line}"


def build_graph_prompt(
    *,
    graph_description: str,
    nodes: list[str],
    edges: list[tuple[str, str, float | None]],
    task: str,
) -> str:
    """Create a compact prompt for community, path, or link explanations."""

    node_text = ", ".join(map(str, nodes)) if nodes else "No node list provided."
    edge_lines = []
    for source, target, weight in edges:
        if weight is None:
            edge_lines.append(f"- {source} -- {target}")
        else:
            edge_lines.append(f"- {source} -- {target} (weight={weight:.4f})")
    edge_text = "\n".join(edge_lines) if edge_lines else "No edge list provided."

    return (
        "You explain graph-derived results without inventing unsupported facts.\n"
        f"Task: {task}\n"
        f"Graph description: {graph_description}\n"
        f"Nodes: {node_text}\n"
        f"Edges:\n{edge_text}\n"
        "Focus on community structure, paths, and topology-based link reasoning."
    )


def _parse_generation_response(payload: dict[str, Any]) -> str:
    if "choices" in payload:
        choice = payload["choices"][0]
        if "message" in choice:
            return str(choice["message"]["content"])
        if "text" in choice:
            return str(choice["text"])
    if "output" in payload:
        return str(payload["output"])
    if "text" in payload:
        return str(payload["text"])
    if "content" in payload:
        return str(payload["content"])
    raise KeyError("LLM response must include choices, output, text, or content")
