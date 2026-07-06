"""Optional explanation helpers for graph-derived results."""

from __future__ import annotations

import networkx as nx


def explain_shortest_path(graph: nx.Graph, path: list[str] | None) -> str:
    """Create a concise deterministic explanation for a shortest path."""

    if not path:
        return "No relationship chain was found between the selected entities."

    if len(path) == 1:
        return f"{path[0]} is the selected entity."

    parts = []
    for source, target in zip(path, path[1:]):
        weight = graph.edges[source, target].get("weight")
        if weight is None:
            parts.append(f"{source} connects to {target}")
        else:
            parts.append(f"{source} connects to {target} with similarity {weight:.3f}")

    return " -> ".join(parts)


def explain_link_prediction(source: str, target: str, score: float, method: str) -> str:
    """Explain a heuristic link prediction without calling an LLM."""

    return (
        f"{method} suggests a latent relation between {source} and {target} "
        f"with score {score:.4f}. The score is derived from graph topology."
    )
