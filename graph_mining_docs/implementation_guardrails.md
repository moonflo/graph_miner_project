# Implementation Guardrails for Codex

## Project identity

This repository implements a lightweight graph topology relation-mining system.

The core chain is:

```text
Entity/Text/Feature -> Embedding -> Graph Construction -> Classical Graph Algorithms -> Evaluation/Explanation
```

## Must not do

Do not introduce these unless explicitly requested:

- PyTorch model training
- GNN training pipeline
- DGL or PyG model code
- Transformer fine-tuning
- RL/agent framework
- Multi-modal model pipeline
- Long-running GPU jobs

## Allowed dependencies

Preferred:

- `networkx`
- `numpy`
- `pandas`
- `scikit-learn`
- `ogb`
- `python-louvain`
- `matplotlib`
- `pyvis`
- `requests`
- `tqdm`

## Graph construction rules

1. Use `networkx.Graph` for classical undirected topology algorithms.
2. Store edge strength in `weight`.
3. Store edge source/type in `relation_type`.
4. Avoid self-loops by default.
5. Keep directed datasets directed only in data loading; convert to undirected only for algorithms that require it, and document the conversion.

## Link prediction rules

1. Use candidate pairs, not all missing edges, on large graphs.
2. Normalize outputs as `source`, `target`, `score`, `method`.
3. Remove self-loops before Adamic-Adar.
4. Sort predicted hidden relations by descending score.

## Community detection rules

1. Use `python-louvain` if available.
2. Import as `import community as community_louvain`.
3. Use `best_partition(G, weight="weight", random_state=42)`.
4. Use NetworkX greedy modularity as fallback only.

## OGB rules

1. Use OGB only for data loading and evaluation.
2. Start with `ogbl-collab`.
3. Build the prediction graph from training edges only.
4. Never include validation/test positive edges in the graph before scoring.
5. Sample candidate pairs for quick tests.

## Reporting language

Use accurate wording:

- “latent relation candidate”
- “topology-based link prediction”
- “community-level hidden structure”
- “shortest-path relation chain”
- “graph-based structural inference”

Avoid unsupported wording:

- “causal discovery” unless causal evidence exists
- “deep learning model” unless training is actually implemented
- “guaranteed hidden relation”
- “ground-truth explanation”

## Minimal success condition

A correct implementation should be able to:

1. Load dummy entities or OGB train edges.
2. Build a NetworkX graph.
3. Score candidate hidden links with Jaccard / Adamic-Adar / Resource Allocation.
4. Detect communities with Louvain or fallback.
5. Find a shortest relation chain between two nodes.
6. Save outputs as JSON/CSV for visualization and reporting.
