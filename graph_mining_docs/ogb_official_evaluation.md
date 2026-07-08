# OGB Official Evaluation for ogbl-collab

This note explains the difference between the legacy candidate-limited smoke
test and the OGB Evaluator-backed official mode for `ogbl-collab`.

The project remains a lightweight graph-mining system: no neural network, no
GNN training, no node2vec training, and no learned re-ranker. The goal of this
mode is evaluation protocol alignment before comparing the classical
time-decay heuristics against OGB leaderboard baselines.

## Two Evaluation Modes

`scripts/smoke_link_prediction.py` is the legacy candidate-limited smoke test.
It is useful for fast development, trend checks, and small method sweeps. It
can limit positive edges, cap negatives, and optionally limit the train graph.
Its output starts with:

```text
Evaluation mode: candidate-limited legacy smoke test
Not directly comparable with OGB leaderboard.
```

`scripts/eval_ogb_official.py` is the official-style path. It uses
`ogb.linkproppred.Evaluator(name="ogbl-collab")`, builds the visible graph from
official `train_edges`, scores official valid/test positive edges and
`edge_neg`, and writes Markdown plus CSV reports.

## Why the Legacy Result Is Not a Leaderboard Number

A legacy smoke line such as `pos=46329 neg=100000` only reports how many flat
candidate scores were produced. It does not by itself prove that the negative
scores were organized exactly as the OGB Evaluator expects for the installed
`ogbl-collab` protocol.

Official evaluation must keep the OGB split layout visible:

- `y_pred_pos`: one score per positive edge.
- `y_pred_neg`: either the OGB shared negative pool accepted by the installed
  Hits@50 evaluator, or a row-wise matrix when a split exposes per-positive
  negatives.

The code now preserves raw `edge_neg` in `OGBSplitData.valid_edge_neg` and
`OGBSplitData.test_edge_neg`. The legacy fields `valid_neg_edges` and
`test_neg_edges` remain flattened for candidate-limited smoke tests.

## Negative Layouts

Some split files expose:

```text
edge_neg shape = [num_pos, num_neg, 2]
```

For that layout, official scoring keeps corresponding negatives aligned with
their positive edge and produces:

```text
y_pred_pos shape = [num_pos]
y_pred_neg shape = [num_pos, num_neg]
```

Other `ogbl-collab` caches, including the local OGB cache used during this
change, expose:

```text
edge_neg shape = [num_neg, 2]
```

That is a shared negative pool. The installed OGB Hits@50 evaluator accepts
`y_pred_neg` as a 1D shared pool for `ogbl-collab`. The official script reports
this as `negative_layout=shared_pool` so it is not confused with per-positive
row-wise negatives. Use `--strict-per-positive-negatives` when you want the
script to reject 2D `edge_neg` instead of using the shared-pool evaluator path.

## Debug Run

```bash
python scripts/eval_ogb_official.py \
  --raw-root data/raw \
  --dataset ogbl_collab \
  --split valid \
  --methods adamic_adar time_decay_common_neighbors \
  --limit-pos 100 \
  --limit-neg-per-pos 100 \
  --decay 0.8 \
  --full-train-graph
```

For 3D per-positive negatives, `--limit-pos` takes the first N positive edges
and the first N corresponding negative groups. `--limit-neg-per-pos` takes the
first K negatives inside each group.

For a 2D shared negative pool, `--limit-pos` still limits positives, and
`--limit-neg-per-pos` caps the shared negative pool for debugging.

## Full Valid

```bash
python scripts/eval_ogb_official.py \
  --raw-root data/raw \
  --dataset ogbl_collab \
  --split valid \
  --methods adamic_adar resource_allocation time_decay_common_neighbors time_decay_resource_allocation \
  --decay 0.8 \
  --full-train-graph \
  --output reports/ogb_official_valid.md \
  --csv-output reports/ogb_official_valid.csv
```

## Full Test

```bash
python scripts/eval_ogb_official.py \
  --raw-root data/raw \
  --dataset ogbl_collab \
  --split test \
  --methods adamic_adar resource_allocation time_decay_common_neighbors time_decay_resource_allocation \
  --decay 0.8 \
  --full-train-graph \
  --output reports/ogb_official_test.md \
  --csv-output reports/ogb_official_test.csv
```

## Output Fields

The Markdown and CSV reports include:

```text
dataset
split
method
decay
pos_used
neg_per_pos_used
total_neg_used
hits@50
nodes
edges
has_weight
has_year
max_train_year
runtime_seconds
```

They also include shape diagnostics such as `edge_neg_shape`,
`negative_layout`, `y_pred_pos_shape`, and `y_pred_neg_shape`.

## Sanity Check

OGB leaderboard Adamic-Adar for `ogbl-collab` is roughly:

```text
valid Hits@50 ~= 0.6349
test Hits@50 ~= 0.6417
```

Do not hard-code those values into tests. Use them as a human sanity check. If
official-mode Adamic-Adar is far from that range, inspect:

1. whether `edge_neg` was interpreted as shared-pool or per-positive data;
2. whether negatives were accidentally flattened into the wrong protocol;
3. whether the visible graph uses only official train edges;
4. whether test evaluation should include train+valid edges for a specific
   comparison target;
5. whether the undirected projection matches the baseline being compared.

After official mode is aligned, compare `time_decay_common_neighbors` and
`time_decay_resource_allocation` against leaderboard baselines such as
Adamic-Adar, Common Neighbor, Jaccard, GCN, GraphSAGE, and Node2vec with clear
labels that this project uses transparent classical graph heuristics.
