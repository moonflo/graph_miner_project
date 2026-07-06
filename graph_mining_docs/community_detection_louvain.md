# Louvain Community Detection Notes

## Purpose in this project

Louvain community detection is used to identify hidden group structure in the constructed graph. This supports the project requirement of discovering deep structural relations among entities.

It should be treated as a classical graph-mining module, not as a neural-network model.

---

## Official sources

- python-louvain documentation: `https://python-louvain.readthedocs.io/en/latest/`
- python-louvain API: `https://python-louvain.readthedocs.io/en/latest/api.html`
- GitHub repository: `https://github.com/taynaud/python-louvain`
- Original Louvain method paper: Blondel et al., *Fast unfolding of communities in large networks*, 2008.

---

## Key official facts

Short official excerpts, preserved for API alignment:

> `This package implements community detection.`

> `Package name is community but refer to python-louvain on pypi`

> `community.best_partition(graph, partition=None, weight='weight', resolution=1.0, randomize=None, random_state=None)`

> `Compute the partition of the graph nodes which maximises the modularity`

> `If the graph is not undirected.`

The practical implication is: call `community_louvain.best_partition(G, weight="weight")` on an undirected NetworkX graph.

---

## Installation note

The package installed by pip is commonly named:

```bash
pip install python-louvain
```

The import commonly used in examples is:

```python
import community as community_louvain
```

Some environments may also support:

```python
import community.community_louvain as community_louvain
```

Use the first import if it works.

---

## Recommended wrapper implementation

```python
import networkx as nx
import community as community_louvain
from collections import defaultdict
from typing import Dict, Any


def detect_louvain_communities(
    G: nx.Graph,
    weight: str = "weight",
    resolution: float = 1.0,
    random_state: int = 42,
) -> Dict[str, Any]:
    if G.is_directed():
        G = G.to_undirected()

    if G.number_of_edges() == 0:
        raise ValueError("Louvain requires a graph with at least one edge.")

    partition = community_louvain.best_partition(
        G,
        weight=weight,
        resolution=resolution,
        random_state=random_state,
    )

    communities = defaultdict(list)
    for node, cid in partition.items():
        communities[cid].append(node)

    modularity = community_louvain.modularity(partition, G, weight=weight)

    return {
        "partition": partition,
        "communities": dict(communities),
        "modularity": float(modularity),
        "num_communities": len(communities),
    }
```

---

## Output format for this project

For downstream visualization and LLM explanation, normalize the output as:

```python
{
    "node_to_community": {node_id: community_id},
    "communities": {
        community_id: [node_id_1, node_id_2, ...]
    },
    "modularity": 0.0,
    "method": "louvain"
}
```

---

## Interpretation guidance

Community detection can be described as:

- hidden group structure discovery
- latent relation cluster identification
- graph topology-based grouping

But do not claim it proves causal relations.

---

## Engineering guardrails

1. Use `nx.Graph`, not `nx.DiGraph`.
2. Keep the edge weight key as `weight`.
3. Fix `random_state` for reproducibility.
4. If the graph has no edges, return a controlled error.
5. Do not introduce GNN libraries for community detection.
6. Keep Louvain as an optional dependency if installation becomes problematic.

---

## Possible fallback

If `python-louvain` is unavailable, use NetworkX greedy modularity communities:

```python
from networkx.algorithms.community import greedy_modularity_communities
communities = list(greedy_modularity_communities(G, weight="weight"))
```

This is not Louvain, but it is a valid CPU-only community detection fallback.
