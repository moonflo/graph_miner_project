# Classical Link Prediction Layer

This layer scores OGB link-prediction candidates with traditional NetworkX
topology heuristics. It is a classical baseline layer, not a GNN, neural model,
GPU pipeline, or training workflow.

## Scope

Implemented package:

```text
src/algorithms/
  __init__.py
  link_prediction.py
  scoring.py
  evaluation.py
```

The layer reuses the graph construction package in `src/graph/`. It does not
rewrite split loading, candidate loading, or NetworkX graph construction.

Supported scoring methods:

- Jaccard coefficient
- Adamic-Adar index
- Resource Allocation index
- Preferential Attachment
- Common neighbors
- Weighted common neighbors
- Weighted Resource Allocation
- Weighted Adamic-Adar
- Time-decay common neighbors
- Time-decay Resource Allocation

All methods require explicit `candidate_pairs`. The implementation does not
call NetworkX link-prediction APIs with `ebunch=None`, because that would
enumerate graph-wide non-edges and is not viable for OGB-scale graphs.

The weighted/time-aware methods are intended first for `ogbl_collab`. They use
official split edge `weight` and `year` when present, and they degrade to
unweighted or weighted topology scores when those attributes are missing.

## Visible Graph Boundary

Formal OGB-style evaluation builds the visible graph from official
`train_edges` only:

```python
build_visible_graph(
    "ogbl_collab",
    source="ogb_train",
    raw_root="data/raw",
    include_isolated_nodes=True,
)
```

Valid/test positive edges are never added to the visible graph. Candidate
endpoints that have no train edge are still valid evaluation nodes, so formal
scoring uses `include_isolated_nodes=True` and the scoring layer safely assigns
zero topology signal to endpoints with no neighborhood.

## Candidate-Limited Scoring

Valid/test scoring uses only official candidates:

- `ogbl_collab`: positive edges plus negative edges
- `ogbl_ppa`: positive edges plus large negative edge arrays
- `ogbl_citation2`: `source_node`, `target_node`, and row-wise
  `target_node_neg`

For `ogbl_citation2`, the negative target matrix is preserved for MRR. It is
not flattened into a global negative edge list for ordinary Hits calculation.
The scoring implementation may flatten the matrix temporarily only to batch
NetworkX scoring, then reshapes scores back to `[num_positive, num_negative]`.

## Directed Citation2 Baseline

`ogbl_citation2` is a directed citation prediction task. NetworkX's classical
link-prediction functions are undirected topology heuristics, so the scoring
layer projects directed graphs to a simple undirected graph before scoring:

- directed graphs become undirected;
- multigraphs become simple graphs;
- parallel or reversed edge weights are summed;
- self-loops are removed before scoring, especially before Adamic-Adar.

Citation2 scores should therefore be read as an undirected-projection classical
baseline, not as a directed model.

## Metrics

Manual numpy metrics are implemented in `src/algorithms/evaluation.py` so the
legacy smoke path can run without depending on an exact OGB `Evaluator` input
format.

Dataset metric routing:

- `ogbl_collab`: Hits@50
- `ogbl_ppa`: Hits@100
- `ogbl_citation2`: MRR

Hits@K compares each positive score against candidate negatives. A 2D negative
array is row-wise; a 1D negative array is treated as a shared global negative
pool. MRR for citation2 uses:

```text
rank = 1 + number of negatives with score >= positive score
MRR = mean(1 / rank)
```

## Legacy Smoke vs Official Evaluation

For `ogbl_collab`, formal leaderboard-aligned evaluation should use
`scripts/eval_ogb_official.py`. It calls
`ogb.linkproppred.Evaluator(name="ogbl-collab")`, preserves raw `edge_neg`
shape, reports the negative layout, and writes Markdown plus CSV reports. See
`graph_mining_docs/ogb_official_evaluation.md`.

The smoke script is intentionally lighter. It limits positive candidates,
negative candidates per positive, and by default limits train graph
construction to a prefix of official `train_edges`. This checks wiring,
candidate shapes, metric routing, and no-crash behavior without pretending to
be the final benchmark.

The default `--limit-pos 100` is a smoke-test value, not the full valid/test
split. Use `--full-positive-split` when you intentionally want all positives
from the current split; it ignores `--limit-pos`. For `ogbl-collab` and other
splits with a global official negative pool, `--limit-neg-per-pos` is a budget
multiplier, and the actual negative count is capped by the available official
negative pool. The smoke output prints `requested_neg`, `available_neg`,
`used_neg`, and `neg_truncated`; report actual `pos/neg` counts when comparing
runs.

Run the smoke script:

```bash
python scripts/smoke_link_prediction.py \
  --raw-root data/raw \
  --split valid \
  --limit-pos 100 \
  --limit-neg-per-pos 100
```

The default smoke methods remain the original four NetworkX baselines. To run
all supported methods for the current collab-focused experiment:

```bash
python scripts/smoke_link_prediction.py \
  --raw-root data/raw \
  --datasets ogbl_collab \
  --split valid \
  --limit-pos 1000 \
  --limit-neg-per-pos 100 \
  --full-train-graph \
  --methods all \
  --decay 0.8
```

To use the full official train graph for a local formal-style run, add:

```bash
--full-train-graph
```

Do this intentionally; full NetworkX graph construction for large OGB splits
can require substantial memory.

To score all positives in the selected valid/test split, add:

```bash
--full-positive-split
```

For repeatable `ogbl-collab` experiments, use:

```bash
python scripts/run_collab_experiments.py \
  --raw-root data/raw \
  --split valid \
  --output-dir reports/collab_experiments \
  --full-train-graph
```

The runner writes `collab_experiments.csv`, `collab_experiments.md`, and
`collab_best_summary.md` under the chosen output directory.
