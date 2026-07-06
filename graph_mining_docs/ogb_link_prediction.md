# OGB Link Prediction Notes

## Purpose in this project

OGB is used as an evaluation dataset source for hidden relation discovery. In this project, OGB should primarily provide ground-truth graph edges and standardized link prediction splits.

The project should not become an OGB-GNN training pipeline.

---

## Official sources

- OGB Get Started: `https://ogb.stanford.edu/docs/home/`
- OGB Link Property Prediction: `https://ogb.stanford.edu/docs/linkprop/`
- OGB Python package: `https://pypi.org/project/ogb/`
- OGB paper: Hu et al., *Open Graph Benchmark: Datasets for Machine Learning on Graphs*, 2020.

---

## Key official facts

Short official excerpts, preserved for implementation alignment:

> `OGB contains graph datasets that are managed by data loaders.`

> `The loaders handle downloading and pre-processing of the datasets.`

> `OGB has standardized evaluators and leaderboards`

The OGB link property prediction page lists link datasets and metrics. Important rows for this project:

- `ogbl-ppa`: link prediction, `Hits@100`
- `ogbl-collab`: link prediction, `Hits@50`
- `ogbl-citation2`: link prediction, `MRR`

---

## Dataset recommendations

### 1. `ogbl-collab` — recommended first

Why:

- Small relative to other OGB link datasets.
- Collaboration network is intuitive and easy to explain.
- Task is future collaboration prediction.
- It naturally matches “hidden relation discovery”.

Official facts summarized from OGB docs:

- Undirected collaboration graph.
- Nodes represent authors.
- Edges represent author collaboration.
- Node features are 128-dimensional paper-based embeddings.
- Split is time-based: train until 2017, validation in 2018, test in 2019.
- Metric: Hits@50.

### 2. `ogbl-ppa` — biological relation option

Why:

- Protein association prediction is a strong hidden-relation story.
- However, it is much larger than `ogbl-collab`.

Official facts summarized from OGB docs:

- Undirected, unweighted graph.
- Nodes are proteins.
- Edges are biologically meaningful associations.
- Metric: Hits@100.

### 3. `ogbl-citation2` — citation relation option

Why:

- Citation prediction is a clear graph relation task.
- However, it is large and directed.

Official facts summarized from OGB docs:

- Directed citation graph.
- Nodes are papers.
- Edges indicate citation.
- Node features are 128-dimensional word2vec title/abstract features.
- Metric: MRR.

---

## Recommended implementation approach

Use the library-agnostic OGB loader for link prediction:

```python
from ogb.linkproppred import LinkPropPredDataset, Evaluator


def load_ogb_link_dataset(name: str, root: str = "data/raw"):
    dataset = LinkPropPredDataset(name=name, root=root)
    graph = dataset[0]
    split_edge = dataset.get_edge_split()
    evaluator = Evaluator(name=name)

    return {
        "name": name,
        "graph": graph,
        "split_edge": split_edge,
        "evaluator": evaluator,
        "num_nodes": graph.get("num_nodes"),
        "edge_index": graph.get("edge_index"),
        "node_feat": graph.get("node_feat"),
        "edge_feat": graph.get("edge_feat"),
    }
```

---

## Important data-shape notes

The exact `split_edge` structure differs by dataset. Always inspect:

```python
print(split_edge.keys())
print(split_edge["train"].keys())
print(split_edge["valid"].keys())
print(split_edge["test"].keys())
```

Do not hard-code negative edge names before checking. Typical keys may include positive edges and negative edges, but dataset details can vary.

---

## How OGB fits this project

### Basic experiment

1. Load OGB graph and split.
2. Build a NetworkX graph from training edges only.
3. Score validation/test candidate edges with classical topology scores.
4. Compare positive edge scores against negative edge scores where available.
5. Report OGB metric when evaluator format is satisfied, otherwise report Hits@K manually.

### Project interpretation

- Training edges = visible relations.
- Validation/test edges = hidden relations to discover.
- Link prediction score = latent relation strength.

This is exactly aligned with hidden relation mining.

---

## Implementation guardrails

1. OGB is for evaluation, not model training.
2. Avoid PyTorch/PyG/DGL unless OGB installation forces lightweight dependencies.
3. Prefer `ogbl-collab` for first implementation.
4. Convert large datasets into sampled subsets for quick local testing.
5. Do not run all-pairs link prediction on OGB-sized graphs.
6. Use train graph only for prediction; do not leak validation/test edges into graph construction.

---

## Manual Hits@K helper

Use this when OGB evaluator input formatting is inconvenient for a classical NetworkX baseline:

```python
import numpy as np


def hits_at_k(pos_scores, neg_scores, k: int) -> float:
    pos_scores = np.asarray(pos_scores)
    neg_scores = np.asarray(neg_scores)

    hits = []
    for s in pos_scores:
        rank = 1 + np.sum(neg_scores >= s)
        hits.append(rank <= k)
    return float(np.mean(hits))
```

This is only a simplified local metric. Use the official OGB evaluator when possible.
