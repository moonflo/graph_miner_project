"""Unified embedding API client with mock and HTTP backends."""

from __future__ import annotations

import hashlib
import re
import time
from collections.abc import Sequence
from typing import Any

import numpy as np
import requests


_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+")


class Embedder:
    """Embedding client that returns numpy arrays with shape ``[N, D]``."""

    def __init__(
        self,
        api_key: str | None = None,
        api_url: str | None = None,
        *,
        model: str | None = None,
        provider: str = "mock",
        batch_size: int = 32,
        timeout: int = 30,
        max_retries: int = 3,
        retry_backoff: float = 1.0,
        mock_dimension: int = 8,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        self.api_key = api_key
        self.api_url = api_url
        self.model = model
        self.provider = provider.lower()
        self.batch_size = batch_size
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        self.mock_dimension = mock_dimension
        self.extra_headers = extra_headers or {}

        if self.provider != "mock" and not self.api_url:
            raise ValueError("api_url is required when provider is not 'mock'")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

    def embed(self, texts: str | Sequence[str]) -> np.ndarray:
        """Embed one string or a list of strings.

        The returned array is always 2-dimensional. A single input string
        returns shape ``[1, D]``.
        """

        normalized_texts = _normalize_texts(texts)
        if not normalized_texts:
            return np.empty((0, 0), dtype=float)

        if self.provider == "mock":
            return _mock_embed_texts(normalized_texts, dimension=self.mock_dimension)

        batches = []
        for start in range(0, len(normalized_texts), self.batch_size):
            batch = normalized_texts[start : start + self.batch_size]
            batches.append(self._embed_http_batch(batch))

        return np.vstack(batches)

    def _embed_http_batch(self, texts: list[str]) -> np.ndarray:
        payload = self._build_payload(texts)
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
                return _parse_embedding_response(response.json())
            except (requests.RequestException, KeyError, ValueError, TypeError) as exc:
                last_error = exc
                if attempt == self.max_retries - 1:
                    break
                time.sleep(self.retry_backoff * (2**attempt))

        raise RuntimeError("Embedding API request failed") from last_error

    def _build_payload(self, texts: list[str]) -> dict[str, Any]:
        if self.provider in {"openai", "qwen", "openai-compatible"}:
            payload: dict[str, Any] = {"input": texts}
            if self.model:
                payload["model"] = self.model
            return payload

        payload = {"texts": texts}
        if self.model:
            payload["model"] = self.model
        return payload

    def _build_headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json", **self.extra_headers}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers


def _normalize_texts(texts: str | Sequence[str]) -> list[str]:
    if isinstance(texts, str):
        return [texts]
    return [str(text) for text in texts]


def _mock_embed_texts(texts: Sequence[str], dimension: int = 8) -> np.ndarray:
    vectors = []
    for text in texts:
        tokens = _TOKEN_PATTERN.findall(text.lower()) or [text.lower()]
        token_vectors = [_hash_token_vector(token, dimension) for token in tokens]
        vector = np.mean(token_vectors, axis=0)
        norm = np.linalg.norm(vector)
        vectors.append(vector / norm if norm else vector)
    return np.vstack(vectors)


def _hash_token_vector(token: str, dimension: int) -> np.ndarray:
    chunks = bytearray()
    counter = 0
    while len(chunks) < dimension:
        digest = hashlib.sha256(f"{token}:{counter}".encode("utf-8")).digest()
        chunks.extend(digest)
        counter += 1

    raw = np.frombuffer(bytes(chunks[:dimension]), dtype=np.uint8).astype(float)
    vector = raw / 127.5 - 1.0
    norm = np.linalg.norm(vector)
    return vector / norm if norm else vector


def _parse_embedding_response(payload: dict[str, Any]) -> np.ndarray:
    if "data" in payload:
        records = payload["data"]
        if records and all(isinstance(item, dict) and "index" in item for item in records):
            records = sorted(records, key=lambda item: item["index"])
        embeddings = [item["embedding"] for item in records]
    elif "embeddings" in payload:
        embeddings = payload["embeddings"]
    elif "embedding" in payload:
        embeddings = [payload["embedding"]]
    elif "vectors" in payload:
        embeddings = payload["vectors"]
    else:
        raise KeyError("Embedding response must include data, embeddings, embedding, or vectors")

    array = np.asarray(embeddings, dtype=float)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2:
        raise ValueError(f"Embedding response must be 2-dimensional, got shape {array.shape}")
    return array
