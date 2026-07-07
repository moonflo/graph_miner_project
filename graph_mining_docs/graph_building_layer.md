# Graph Building Layer

The graph building layer provides the stable boundary between prepared data and
traditional graph algorithms. It does not train models, does not download data,
and does not create train/valid/test splits from processed graph files.

## Manual Dataset Registry

Supported datasets are declared manually in `src/graph/dataset_registry.py`.
The current registry contains exactly:

- `ogbl_citation2`
- `ogbl_collab`
- `ogbl_ppa`

Names such as `ogbl-citation2` are normalized to `ogbl_citation2`, but unknown
datasets are rejected. The graph layer intentionally does not scan
`data/processed/` to discover new datasets. If a new OGB subset or custom graph
format is added later, it must first receive an explicit `DatasetConfig`.

Each registry entry records `canonical_name`, `ogb_name`, `processed_dir_name`,
`raw_dir_name`, `task_type`, `directed`, `edge_relation_name`,
`processed_graph_is_aggregated`, `use_official_split_for_metrics`, and `notes`.

## Processed Graph vs Official OGB Split

`data/processed/<dataset>/graph_nodes.jsonl` and `graph_edges.jsonl` are useful
for graph-construction smoke tests, topology inspection, and reusable graph
loading. They are not a replacement for official OGB split semantics.

Formal OGB evaluation must build the visible training graph from official
`train_edges`, then score against official `valid` and `test` positive and
negative candidates. Do not randomly split `processed/graph_edges.jsonl`, and
do not add valid/test positive edges to the training graph.

The graph layer therefore exposes two separate construction paths:

- `build_networkx_graph_from_processed(...)`
- `build_networkx_graph_from_train_split(...)`

Use the train-split path for metric work.

## Dataset-Specific Notes

`ogbl_citation2` is a directed citation prediction task. Its default graph type
is `nx.DiGraph`, and formal metrics should preserve direction. The candidate
layer keeps citation-style `target_node_neg` matrices instead of pretending
they are ordinary full non-edge enumerations.

`ogbl_collab` is undirected, but its processed `graph_edges.jsonl` is
aggregated and can contain fewer records than official `train_edges`. Use
processed graph edges for topology smoke tests and official split data for
metrics or algorithms that need edge multiplicity, year, or weight.

`ogbl_ppa` is undirected and should also use official split data for metrics.

## Candidate-Limited Evaluation Inputs

Full non-edge enumeration is not scalable for OGB-sized graphs. The
`src/graph/candidates.py` module provides candidate-limited helpers:

- `get_eval_candidates(dataset_name, split="valid" | "test")`
- `candidates_from_split(split_data, split)`
- `sample_non_edges(graph, num_samples, seed=42)`
- `normalize_edge_array(...)`
- `edges_to_node_pairs(...)`

`sample_non_edges` uses bounded random attempts. It does not enumerate the full
complement graph.

## Smoke Test

Run the graph building smoke test against local real data:

```bash
python scripts/smoke_graph_build.py \
  --processed-root data/processed \
  --raw-root data/raw \
  --limit-edges 5000
```

You can restrict the manually registered dataset list:

```bash
python scripts/smoke_graph_build.py \
  --processed-root data/processed \
  --raw-root data/raw \
  --limit-edges 5000 \
  --datasets ogbl_collab ogbl_ppa
```

The script builds small processed graphs and small train-visible graphs, prints
basic stats, checks registry-directed defaults, and reports valid/test candidate
shapes. It does not run full Adamic-Adar/Jaccard, enumerate all non-edges, write
to `data/`, or download OGB data.
