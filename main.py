"""Minimal demo for graph-based latent relation mining."""

from __future__ import annotations

from math import log, sqrt


def mock_embedding_placeholder() -> list[list[float]]:
    """Return small fixed vectors that mimic an embedding API response."""

    return [
        [1.00, 0.00, 0.00],
        [0.96, 0.28, 0.00],
        [0.86, 0.51, 0.00],
        [0.00, 1.00, 0.00],
        [0.25, 0.97, 0.00],
        [0.10, 0.99, 0.00],
    ]


def main() -> None:
    entities = [
        {"id": "Alice", "text": "Graph mining and entity relation analysis"},
        {"id": "Bob", "text": "Network science and link prediction"},
        {"id": "Carol", "text": "Community detection in citation graphs"},
        {"id": "Diana", "text": "Protein interaction graph analysis"},
        {"id": "Evan", "text": "Biological network topology"},
        {"id": "Fiona", "text": "Pathway relationship mining"},
    ]

    embeddings = mock_embedding_placeholder()

    try:
        from src.graph_algorithms import adamic_adar_predictions, detect_louvain_communities
        from src.graph_builder import build_cosine_similarity_graph

        graph, _ = build_cosine_similarity_graph(entities, embeddings, threshold=0.94)
        predictions = adamic_adar_predictions(graph, top_n=5)
        communities = detect_louvain_communities(graph)

        print("Graph nodes:")
        print(list(graph.nodes(data=True)))

        print("\nGraph edges:")
        for source, target, data in graph.edges(data=True):
            print(f"{source} -- {target} | similarity={data['weight']:.3f}")

    except ModuleNotFoundError:
        nodes, edges, adjacency = build_fallback_graph(entities, embeddings, threshold=0.94)
        predictions = fallback_adamic_adar(adjacency, top_n=5)
        communities = fallback_communities(adjacency)

        print("Graph nodes:")
        print(nodes)

        print("\nGraph edges:")
        for source, target, weight in edges:
            print(f"{source} -- {target} | similarity={weight:.3f}")

    print("\nAdamic-Adar link prediction:")
    for source, target, score in predictions:
        print(f"{source} <-> {target} | score={score:.4f}")

    print("\nCommunities:")
    for index, community in enumerate(communities, start=1):
        print(f"Community {index}: {sorted(community)}")


def build_fallback_graph(
    entities: list[dict],
    embeddings: list[list[float]],
    threshold: float,
) -> tuple[list[tuple[str, dict]], list[tuple[str, str, float]], dict[str, set[str]]]:
    """Small no-dependency graph builder used only when dependencies are absent."""

    nodes = [(entity["id"], {"text": entity["text"]}) for entity in entities]
    edges = []
    adjacency = {entity["id"]: set() for entity in entities}

    for i, source in enumerate(entities):
        for j in range(i + 1, len(entities)):
            score = cosine_similarity(embeddings[i], embeddings[j])
            if score >= threshold:
                target = entities[j]
                edges.append((source["id"], target["id"], score))
                adjacency[source["id"]].add(target["id"])
                adjacency[target["id"]].add(source["id"])

    return nodes, edges, adjacency


def cosine_similarity(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = sqrt(sum(a * a for a in left))
    right_norm = sqrt(sum(b * b for b in right))
    return dot / (left_norm * right_norm)


def fallback_adamic_adar(
    adjacency: dict[str, set[str]],
    top_n: int,
) -> list[tuple[str, str, float]]:
    nodes = list(adjacency)
    predictions = []

    for i, source in enumerate(nodes):
        for target in nodes[i + 1 :]:
            if target in adjacency[source]:
                continue

            common_neighbors = adjacency[source] & adjacency[target]
            score = sum(
                1 / log(len(adjacency[neighbor]))
                for neighbor in common_neighbors
                if len(adjacency[neighbor]) > 1
            )
            predictions.append((source, target, score))

    return sorted(predictions, key=lambda item: item[2], reverse=True)[:top_n]


def fallback_communities(adjacency: dict[str, set[str]]) -> list[set[str]]:
    visited = set()
    communities = []

    for start in adjacency:
        if start in visited:
            continue

        stack = [start]
        community = set()
        while stack:
            node = stack.pop()
            if node in visited:
                continue

            visited.add(node)
            community.add(node)
            stack.extend(adjacency[node] - visited)

        communities.append(community)

    return communities


if __name__ == "__main__":
    main()
