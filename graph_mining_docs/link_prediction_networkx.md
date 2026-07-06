# NetworkX Link Prediction Notes

## Purpose in this project

This project uses classical graph topology scores to discover candidate hidden relations. These methods are lightweight, deterministic, CPU-friendly, and require no neural training.

Supported methods:

- Jaccard coefficient
- Adamic-Adar index
- Resource Allocation index
- Preferential Attachment, optional

These are suitable as simple hidden-link discovery baselines or as the core relation-mining module.

---

## Official sources

- NetworkX Link Prediction overview: `https://networkx.org/documentation/stable/reference/algorithms/link_prediction.html`
- Jaccard coefficient: `https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.link_prediction.jaccard_coefficient.html`
- Adamic-Adar index: `https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.link_prediction.adamic_adar_index.html`
- Resource Allocation index: `https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.link_prediction.resource_allocation_index.html`

---

## Official API facts

The NetworkX Link Prediction page lists these functions:

> `resource_allocation_index(G[, ebunch])`

> `jaccard_coefficient(G[, ebunch])`

> `adamic_adar_index(G[, ebunch])`

NetworkX returns an iterator of 3-tuples:

```python
(u, v, p)
```

where `p` is the predicted score for the candidate pair.

Important official constraint: these functions are for NetworkX undirected graphs. Do not pass `DiGraph`, `MultiGraph`, or `MultiDiGraph` directly.

---

## Formulas

Let \( \Gamma(u) \) denote the neighbors of node \(u\).

### Jaccard coefficient

\[
J(u,v)=\frac{|\Gamma(u)\cap\Gamma(v)|}{|\Gamma(u)\cup\Gamma(v)|}
\]

Interpretation:

- High if two nodes share a large fraction of their neighborhoods.
- Good for simple structural similarity.

### Adamic-Adar index

\[
AA(u,v)=\sum_{w\in\Gamma(u)\cap\Gamma(v)}\frac{1}{\log |\Gamma(w)|}
\]

Interpretation:

- Shared neighbors with lower degree contribute more.
- Good when rare/common neighbors should be weighted differently.
- Avoid self-loops because NetworkX documentation notes zero-division risk in self-loop-only cases.

### Resource Allocation index

\[
RA(u,v)=\sum_{w\in\Gamma(u)\cap\Gamma(v)}\frac{1}{|\Gamma(w)|}
\]

Interpretation:

- Similar to Adamic-Adar but penalizes high-degree common neighbors more directly.

---

## Recommended wrapper implementation

```python
import networkx as nx
from typing import Iterable, Optional, Tuple, List

NodePair = Tuple[object, object]
ScoredPair = Tuple[object, object, float]


def _ensure_simple_undirected_graph(G: nx.Graph) -> nx.Graph:
    if G.is_directed():
        G = G.to_undirected()
    if isinstance(G, (nx.MultiGraph, nx.MultiDiGraph)):
        H = nx.Graph()
        H.add_nodes_from(G.nodes(data=True))
        for u, v, data in G.edges(data=True):
            weight = data.get("weight", 1.0)
            if H.has_edge(u, v):
                H[u][v]["weight"] += weight
            else:
                H.add_edge(u, v, weight=weight)
        return H
    return G


def predict_jaccard(G: nx.Graph, ebunch: Optional[Iterable[NodePair]] = None) -> List[ScoredPair]:
    G = _ensure_simple_undirected_graph(G)
    return list(nx.jaccard_coefficient(G, ebunch))


def predict_adamic_adar(G: nx.Graph, ebunch: Optional[Iterable[NodePair]] = None) -> List[ScoredPair]:
    G = _ensure_simple_undirected_graph(G)
    G.remove_edges_from(nx.selfloop_edges(G))
    return list(nx.adamic_adar_index(G, ebunch))


def predict_resource_allocation(G: nx.Graph, ebunch: Optional[Iterable[NodePair]] = None) -> List[ScoredPair]:
    G = _ensure_simple_undirected_graph(G)
    return list(nx.resource_allocation_index(G, ebunch))
```

---

## Candidate-pair policy

Do not run link prediction over all missing edges on large graphs unless the graph is tiny. If `ebunch=None`, NetworkX may score all nonexistent edges, which is expensive.

Recommended approaches:

1. For small demos: `ebunch=None` is acceptable.
2. For real OGB-sized graphs: pass candidate pairs from validation/test negatives or sampled pairs.
3. For entity text demos: restrict to top-N embedding candidates before topology scoring.

---

## Output format for this project

Normalize all method outputs into dictionaries:

```python
{
    "source": u,
    "target": v,
    "score": float(p),
    "method": "adamic_adar" | "jaccard" | "resource_allocation"
}
```

Sort descending by `score`.

---

## Engineering guardrails

- Use `nx.Graph` unless a dataset-specific reason requires directionality.
- Remove self-loops before Adamic-Adar.
- Keep link prediction methods separate from graph construction.
- Do not train GNNs here.
- Do not use PyTorch for these classical topology scores.
