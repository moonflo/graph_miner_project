# Weighted and Time-aware Link Prediction for ogbl-collab

This note describes the current `ogbl-collab` focused classical link prediction
extension. It keeps the project inside the lightweight graph-mining boundary:
no neural network, no GNN, no node2vec or DeepWalk training, and no learned
embedding re-ranker.

## Why stay classical

The current goal is to make the visible train graph more faithful before adding
any learned layer. `ogbl-collab` already carries useful edge evidence:

- `weight`: collaboration strength for an author pair;
- `year`: when that collaboration edge belongs in the time split.

Using these fields in transparent heuristics keeps the pipeline interpretable,
cheap to run on a 16 GB machine, and easy to compare against the existing
NetworkX baselines.

## Current focus

`ogbl-collab` is the main research dataset for this stage. The task is future
collaboration prediction, and the official split is time based, so edge
strength and recency are directly meaningful. `ogbl-ddi` and additional `ppa`
research are intentionally out of scope for this round.

## Added methods

The original methods remain available:

- `jaccard`
- `adamic_adar`
- `resource_allocation`
- `preferential_attachment`

The collab-friendly methods are:

- `common_neighbors`
- `weighted_common_neighbors`
- `weighted_resource_allocation`
- `weighted_adamic_adar`
- `time_decay_common_neighbors`
- `time_decay_resource_allocation`

Weighted common neighbors uses the average of the two incident edge weights for
each shared neighbor. Time-aware methods multiply an edge weight by
`decay ** (max_train_year - edge_max_year)`. If year metadata is missing, the
time-aware methods fall back to their weighted counterpart instead of failing.

## Business interpretation

These scores map cleanly to real relation mining ideas:

- common neighbors represent shared evidence paths;
- edge weight represents relation or evidence strength;
- weighted degree dampens overly broad hubs;
- edge year distinguishes fresh evidence from stale evidence;
- decay controls how quickly older relations lose influence.

## Recommended smoke run

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

The smoke report prints whether weight/year metadata was loaded, the
`max_train_year`, the decay value, and the number of topology edges used for
candidate-limited scoring. It also prints `requested_neg`, `available_neg`,
`used_neg`, and `neg_truncated`, because `limit-neg-per-pos` is a budget and
the official `ogbl-collab` negative pool can cap the actual number of negatives
used.

## Repeatable collab experiment runner

```bash
python scripts/run_collab_experiments.py \
  --raw-root data/raw \
  --split valid \
  --output-dir reports/collab_experiments \
  --full-train-graph
```

The runner performs a decay sweep and a scale sweep for the current classical
methods, then writes:

- `collab_experiments.csv`
- `collab_experiments.md`
- `collab_best_summary.md`

Use `--include-full-positive` only when you intentionally want to append one
full-positive split run to the scale sweep.
