# NetworkX Basics for This Project

## Purpose in this project

NetworkX is the core graph container and algorithm library used by this project. The project should construct a `networkx.Graph` from entity embeddings or OGB edges, then run classical graph algorithms over that graph.

Use NetworkX for:

- Node and edge storage
- Weighted graph construction
- Link prediction algorithms
- Shortest-path analysis
- Community-related graph utilities
- Lightweight graph statistics

Do not use NetworkX as a wrapper around a deep learning pipeline.

---

## Official source

- NetworkX documentation: `https://networkx.org/documentation/stable/`
- `Graph.add_edge`: `https://networkx.org/documentation/stable/reference/classes/generated/networkx.Graph.add_edge.html`
- Shortest paths: `https://networkx.org/documentation/stable/reference/algorithms/shortest_paths.html`

---

## Key API facts from official docs

Short official excerpts, preserved for implementation alignment:

> `Graph.add_edge(u_of_edge, v_of_edge, **attr)`

> `Add an edge between u and v.`

> `Adding an edge that already exists updates the edge data.`

> `Many NetworkX algorithms designed for weighted graphs use an edge attribute (by default weight)`

These facts imply that our graph construction code should consistently store relation strength in the edge attribute named `weight`.

---

## Recommended graph representation

Use an undirected weighted graph for most project workflows:

```python
import networkx as nx

G = nx.Graph()
G.add_node(0, label="entity_0")
G.add_node(1, label="entity_1")
G.add_edge(0, 1, weight=0.82, relation_type="cosine")
```

Recommended edge attributes:

```python
{
    "weight": float,
    "relation_type": "cosine" | "knn" | "ogb" | "llm" | "hybrid",
    "source": str | None,
}
```

Use `relation_type`, not `type`, because `type` may be confused with Python built-ins and is less explicit.

---

## Shortest-path use

Official NetworkX docs state that shortest path can be computed on weighted or unweighted graphs, and that weighted paths use the edge attribute specified by the `weight` parameter.

For this project:

- For topological relation chains, use unweighted shortest path:

```python
path = nx.shortest_path(G, source=node_a, target=node_b)
```

- For weighted relation chains, beware that similarity is high when the relation is strong. Shortest-path algorithms minimize edge cost, so do not directly use similarity as path cost. Convert similarity to distance first:

```python
for u, v, data in G.edges(data=True):
    sim = data.get("weight", 1.0)
    data["distance"] = 1.0 - sim

path = nx.shortest_path(G, source=node_a, target=node_b, weight="distance")
```

This avoids the common bug where high-similarity edges are treated as expensive.

---

## Engineering recommendations

1. Keep graph construction deterministic.
2. Store node labels separately from node ids.
3. Keep `weight` as the canonical numeric edge strength.
4. Avoid self-loops unless a later algorithm explicitly requires them.
5. Prefer `nx.Graph` for Adamic-Adar, Jaccard, Resource Allocation, and Louvain.
6. Convert directed OGB datasets to undirected only if the experiment explicitly allows it.

---

## Minimal sanity check

```python
import networkx as nx

G = nx.Graph()
G.add_edge("A", "B", weight=0.8)
G.add_edge("B", "C", weight=0.7)

assert G.number_of_nodes() == 3
assert G.number_of_edges() == 2
assert G["A"]["B"]["weight"] == 0.8
```
