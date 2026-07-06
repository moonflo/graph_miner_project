"""Graph construction from entity embeddings."""

from __future__ import annotations

import networkx as nx
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def build_cosine_similarity_graph(
    entities: list[dict],
    embeddings: np.ndarray,
    threshold: float = 0.8,
    top_k: int | None = None,
) -> tuple[nx.Graph, np.ndarray]:
    """Build an undirected weighted graph from cosine similarity scores."""

    if len(entities) != len(embeddings):
        raise ValueError("entities and embeddings must have the same length")

    similarity_matrix = cosine_similarity(embeddings)
    graph = nx.Graph()

    for entity in entities:
        graph.add_node(str(entity["id"]), text=entity.get("text", ""))

    for i, source in enumerate(entities):
        candidates = [
            (j, float(similarity_matrix[i, j]))
            for j in range(len(entities))
            if j != i and similarity_matrix[i, j] >= threshold
        ]
        if top_k is not None:
            candidates = sorted(candidates, key=lambda item: item[1], reverse=True)[:top_k]

        for j, score in candidates:
            if i < j:
                graph.add_edge(str(source["id"]), str(entities[j]["id"]), weight=score)

    return graph, similarity_matrix
