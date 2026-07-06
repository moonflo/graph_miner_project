# Graph Mining Docs for Codex

This folder contains curated implementation notes for a lightweight graph-based latent relation mining system.

Important: these notes are not full copies of official documentation. They preserve the key API facts, formulas, constraints, and source links needed to implement the project correctly, while avoiding large verbatim copying.

Recommended reading order:

1. `networkx_basics.md`
2. `link_prediction_networkx.md`
3. `community_detection_louvain.md`
4. `ogb_link_prediction.md`
5. `implementation_guardrails.md`

Project-level rule:

```text
Entity/Text/Node Feature -> Embedding -> Graph Construction -> NetworkX Graph Algorithms -> Evaluation/Explanation
```

Do not convert this project into a GNN training system unless explicitly requested.
