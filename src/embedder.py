"""Embedding helpers.

The demo uses deterministic mock embeddings. Real embedding APIs can be wired
through EmbeddingAPIClient without changing the graph algorithm layer.
"""

from __future__ import annotations

import hashlib
from typing import Iterable

import numpy as np
import requests


def mock_embed_texts(texts: Iterable[str], dimension: int = 8) -> np.ndarray:
    """Create deterministic mock embeddings for demos and tests."""

    vectors = []
    for text in texts:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        raw = np.frombuffer(digest, dtype=np.uint8)[:dimension].astype(float)
        vector = raw - raw.mean()
        norm = np.linalg.norm(vector)
        vectors.append(vector / norm if norm else vector)
    return np.vstack(vectors)


class EmbeddingAPIClient:
    """Minimal embedding API client placeholder."""

    def __init__(self, endpoint: str, api_key: str | None = None, timeout: int = 30):
        self.endpoint = endpoint
        self.api_key = api_key
        self.timeout = timeout

    def embed_texts(self, texts: Iterable[str]) -> np.ndarray:
        """Call an embedding API that returns {'embeddings': [[...], ...]}."""

        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        response = requests.post(
            self.endpoint,
            json={"texts": list(texts)},
            headers=headers,
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        return np.asarray(payload["embeddings"], dtype=float)
