# Graph-based Latent Relation Mining System

This project is a lightweight graph topology relation mining system. It is designed to discover hidden or latent relations between entities through embedding-driven graph construction, graph algorithms, and optional natural-language explanation.

The system does not train deep learning models. It focuses on:

- Building graph structures from entity text or OGB nodes.
- Mining latent relations with graph topology and heuristic link prediction.
- Analyzing community structure and relationship paths.
- Optionally using an LLM only to explain graph-derived results.

## Project Goal

The goal of this project is to provide a graph-based latent relation mining system for discovering implicit relations between entities.

Inputs can be:

- Custom entity text, such as names, descriptions, documents, organizations, products, or concepts.
- OGB graph data, especially for ground truth link prediction evaluation.

Outputs can include:

- Constructed graph structures.
- Link prediction results.
- Community structures.
- Shortest path relationship chains.
- Optional LLM explanations for discovered paths or predicted links.

## Core Pipeline

The system follows a simple and interpretable pipeline:

1. Entity input
   - Text entities.
   - OGB nodes.

2. Embedding API
   - Entity text is converted into vectors.
   - The current demo uses mock embeddings.
   - A real embedding API can be connected later through `src/embedder.py`.

3. Cosine similarity graph construction
   - Pairwise cosine similarity is computed between entity embeddings.
   - Nodes represent entities.
   - Edges represent similarity-based inferred relations.
   - Edge weights store cosine similarity scores.

4. NetworkX graph analytics
   - Adamic-Adar link prediction.
   - Jaccard similarity.
   - Louvain community detection.
   - Shortest path reasoning.

5. Optional LLM explanation layer
   - LLM calls are optional.
   - The LLM is not used for training or inference over neural weights.
   - It can explain graph-derived paths, communities, or predicted links in readable language.

## Supported Datasets

The project supports custom text entity data and OGB datasets.

Supported OGB datasets:

- `ogbl-collab`
- `ogbl-ppa`
- `ogbl-citation2`

OGB is used for ground truth link prediction evaluation. It is not used to train deep learning models in this project.

## Design Philosophy

This project intentionally avoids deep learning training pipelines.

It does not:

- Train neural networks.
- Train GNNs.
- Require GPU training.
- Fine-tune large language models.

It only uses:

- Embedding API outputs.
- Graph algorithms.
- Heuristic and statistical methods.
- Optional LLM explanation after graph analysis.

At its core, this is an:

> embedding-driven graph topology inference system

The system favors interpretability, inspectable graph structure, and simple topology-driven reasoning over complex model training.

## Project Structure

```text
graph_mining_project/
├── src/
│   ├── data_loader.py
│   ├── preprocess/
│   │   ├── data_loader.py
│   │   ├── normalizer.py
│   │   ├── schema.py
│   │   ├── preprocess.py
│   ├── llm/
│   │   ├── client.py
│   │   ├── extractor.py
│   │   ├── json_utils.py
│   │   ├── prompts.py
│   │   ├── schema.py
│   ├── graph/
│   │   ├── candidates.py
│   │   ├── dataset_registry.py
│   │   ├── graph_factory.py
│   │   ├── graph_stats.py
│   │   ├── ogb_split_loader.py
│   │   ├── processed_loader.py
│   │   ├── schemas.py
│   ├── algorithms/
│   │   ├── link_prediction.py
│   │   ├── scoring.py
│   │   ├── evaluation.py
│   ├── embedder.py
│   ├── graph_builder.py
│   ├── graph_algorithms.py
│   ├── llm_explainer.py
│   ├── evaluator.py
├── scripts/
│   ├── run_llm_extract.py
│   ├── smoke_graph_build.py
│   ├── smoke_link_prediction.py
│   ├── validate_processed_data.py
├── utils/
│   ├── data_utils.py
│   ├── download_datasets.py
│   ├── embedder.py
│   ├── llm_client.py
├── data/
│   ├── raw/
│   ├── processed/
├── processed/
├── configs/
│   ├── config.yaml
├── tests/
│   ├── test_llm_extraction_smoke.py
│   ├── test_preprocess_smoke.py
├── main.py
├── requirements.txt
├── README.md
├── agent.md
```

## Quick Start

Install dependencies:

```bash
pip install -r requirements.txt
```

If you use the provided conda environment locally, activate it first:

```bash
conda activate graph-miner
```

Run the demo:

```bash
python main.py
```

The demo will:

- Create dummy entity nodes.
- Use a mock embedding placeholder.
- Build a cosine similarity graph.
- Run Adamic-Adar link prediction.
- Print detected communities.

Load an OGB evaluation dataset:

```bash
python -m utils.download_datasets ogbl-collab --negative-samples 100
```

The OGB loader is evaluation-only. It downloads the dataset through OGB,
standardizes edge splits into numpy arrays, and does not introduce a GNN
training pipeline.

Download all supported OGB evaluation datasets manually:

```bash
conda activate graph-miner
python -m utils.download_datasets ogbl-collab --negative-samples 5
python -m utils.download_datasets ogbl-ppa --negative-samples 5
python -m utils.download_datasets ogbl-citation2 --negative-samples 5
```

You can also request every supported dataset in one command:

```bash
python -m utils.download_datasets --all --negative-samples 5
```

The datasets are cached under `data/raw/` and are ignored by Git. Approximate
download sizes:

- `ogbl-collab`: about 0.11 GB
- `ogbl-ppa`: about 0.38 GB
- `ogbl-citation2`: about 2.14 GB

To verify local cached datasets without generating extra negatives, run:

```bash
python -m utils.download_datasets ogbl-collab
python -m utils.download_datasets ogbl-ppa
python -m utils.download_datasets ogbl-citation2
```

Preprocess raw datasets into graph-construction inputs:

```bash
python -m src.preprocess.preprocess --input data/raw/ogbl_collab --output data/processed
```

The preprocessor intentionally handles one dataset per run. To process another
dataset, run the command again with that dataset directory, for example
`data/raw/ogbl_collab` or `data/raw/ogbl_ppa`. Passing a parent directory such
as `data/` or `data/raw/` is rejected to keep preprocessing explicit.

Smoke-test the graph building layer on local real data:

```bash
python scripts/smoke_graph_build.py \
  --processed-root data/processed \
  --raw-root data/raw \
  --limit-edges 5000
```

The graph building layer uses a manual registry in
`src/graph/dataset_registry.py`. It currently supports only `ogbl_citation2`,
`ogbl_collab`, and `ogbl_ppa`; it does not scan `data/processed/` to auto-add
new datasets. For formal OGB metrics, build the visible graph from official
`train_edges` and evaluate with official `valid/test` positive and negative
candidate edges. Do not randomly split `processed/graph_edges.jsonl`.

See `graph_mining_docs/graph_building_layer.md` for registry rules, processed
vs official split boundaries, `ogbl_citation2` directed handling, `ogbl_collab`
aggregation notes, and candidate-limited evaluation guidance.

Smoke-test the classical link prediction layer:

```bash
python scripts/smoke_link_prediction.py \
  --raw-root data/raw \
  --split valid \
  --limit-pos 100 \
  --limit-neg-per-pos 100
```

The smoke default is intentionally not the full valid/test split. If
`--full-positive-split` is not passed, `--limit-pos` controls how many positive
edges are scored, and its default `100` is only a small smoke-test value.
`--limit-neg-per-pos` is a negative-sample budget multiplier; for OGB splits
with a global official negative pool, the actual negative count is capped by
the available official negatives. For example, `--limit-pos 20000
--limit-neg-per-pos 100` requests 2,000,000 negatives, but `ogbl-collab`
valid/test may expose fewer official negatives. The smoke output reports
`requested_neg`, `available_neg`, `used_neg`, and `neg_truncated`; report
actual `pos/neg` counts in experiment notes instead of only the CLI flags. It
is a candidate-limited legacy smoke test and is not directly comparable with
the OGB leaderboard.

Quick `ogbl-collab` smoke for the current weighted/time-aware run:

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

Use the full positive valid/test split intentionally:

```bash
python scripts/smoke_link_prediction.py \
  --raw-root data/raw \
  --datasets ogbl_collab \
  --split valid \
  --full-positive-split \
  --limit-neg-per-pos 100 \
  --full-train-graph \
  --methods all \
  --decay 0.8
```

Run OGB Evaluator-backed official mode for leaderboard-aligned `ogbl-collab`
Hits@50:

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

Use official mode for formal reports. It preserves raw `edge_neg` shape,
reports whether negatives are per-positive or a shared OGB pool, calls
`ogb.linkproppred.Evaluator(name="ogbl-collab")`, and writes Markdown plus CSV.
Use `scripts/smoke_link_prediction.py` only for quick development and method
trend checks. See `graph_mining_docs/ogb_official_evaluation.md` for the full
valid/test commands, shape expectations, and Adamic-Adar sanity check.

Run the repeatable `ogbl-collab` experiment grid and save CSV/Markdown reports:

```bash
python scripts/run_collab_experiments.py \
  --raw-root data/raw \
  --split valid \
  --output-dir reports/collab_experiments \
  --full-train-graph
```

Run only the scale sweep:

```bash
python scripts/run_collab_experiments.py \
  --raw-root data/raw \
  --split valid \
  --output-dir reports/collab_experiments_scale \
  --skip-decay-sweep \
  --positive-limits 1000 5000 10000 20000 \
  --neg-per-pos 100 \
  --full-train-graph
```

Append one full-positive split run to the scale sweep:

```bash
python scripts/run_collab_experiments.py \
  --raw-root data/raw \
  --split valid \
  --output-dir reports/collab_experiments_full \
  --include-full-positive \
  --full-train-graph
```

The experiment runner writes `collab_experiments.csv`,
`collab_experiments.md`, and `collab_best_summary.md` under `--output-dir`.
It directly reuses the classical scoring API, does not shell out to the smoke
script, and does not introduce neural networks, GNNs, or re-rankers.

This layer lives in `src/algorithms/` and uses candidate-limited classical
topology baselines. The original Jaccard, Adamic-Adar, Resource Allocation, and
Preferential Attachment methods remain the default smoke methods. For
`ogbl-collab`, explicit `--methods all` also runs common-neighbor,
weighted, and time-decay variants that use official split `weight` and `year`
when present. The visible graph is still built from official train edges only,
and valid/test candidates are scored without enumerating all non-edges. See
`graph_mining_docs/link_prediction_layer.md` for the OGB metric mapping and
`graph_mining_docs/collab_weighted_time_aware_lp.md` for the collab-focused
weighted/time-aware methods.

This writes:

```text
processed/<dataset_name>/
  documents.jsonl
  entities.jsonl
  relations.jsonl
  triples.jsonl
  graph_nodes.jsonl
  graph_edges.jsonl
  stats.json
```

The preprocessing layer supports one JSON, JSONL, CSV, TXT file, one generic
dataset directory, or one OGB link-prediction cache directory such as
`data/raw/ogbl_citation2`. It normalizes raw samples into documents, explicit
entity/relation tables, simplified triples, and graph-ready node/edge files. If
a raw dataset has only text and no entity or relation annotations, the entity,
relation, triple, node, and edge files can be empty; that is expected until a
rule-based extractor or LLM extractor is connected. See
`graph_mining_docs/preprocessing_layer.md` for the file contracts.

Optionally run LLM extraction after preprocessing a custom text dataset:

```bash
python scripts/run_llm_extract.py \
  --input data/processed/<dataset_name>/documents.jsonl \
  --output-dir data/processed/<dataset_name> \
  --mock \
  --limit 10
```

This writes raw extraction files:

```text
data/processed/<dataset_name>/
  llm_extractions.jsonl
  entities.raw.jsonl
  relations.raw.jsonl
  triples.raw.jsonl
  llm_extract_stats.json
```

The LLM extraction layer is optional and is intended for future custom
unstructured text data. It is not part of the OGB structure-graph preprocessing
flow, and it does not generate `graph_nodes.jsonl` or `graph_edges.jsonl`
directly. See `graph_mining_docs/llm_extraction_layer.md` for configuration,
dry-run, mock, resume, and small real-call examples.

## Main Components

`src/data_loader.py`

- Loads custom text entity data.
- Provides a lightweight OGB link prediction dataset loader.

`src/embedder.py`

- Provides deterministic mock embeddings for demos.
- Provides a minimal embedding API client placeholder.

`src/graph_builder.py`

- Builds a cosine similarity graph from embeddings.
- Stores entity text as node attributes.
- Stores cosine similarity as edge weight.

`src/graph/`

- Defines the explicit dataset registry for `ogbl_citation2`, `ogbl_collab`,
  and `ogbl_ppa`.
- Streams processed `graph_nodes.jsonl`, `graph_edges.jsonl`, and `stats.json`.
- Loads official OGB train/valid/test splits without triggering downloads.
- Builds NetworkX graphs from processed samples or official train splits.
- Provides candidate-limited evaluation inputs and lightweight graph stats.
- Preserves `ogbl-collab` split edge `weight` and `year` when present and
  summarizes duplicate train edges with weight sums and min/max years.

`src/graph_algorithms.py`

- Runs graph topology algorithms:
  - Adamic-Adar link prediction.
  - Jaccard coefficient.
  - Resource allocation.
  - Louvain community detection.
  - Shortest path analysis.

`src/llm_explainer.py`

- Provides optional text explanations for graph-derived results.
- Keeps explanation separate from graph computation.

`src/evaluator.py`

- Provides simple link prediction metrics for ground truth evaluation.
- Intended for OGB evaluation splits or custom labeled edge sets.

`utils/data_utils.py`

- Loads supported OGB link prediction datasets:
  - `ogbl-collab`
  - `ogbl-ppa`
  - `ogbl-citation2`
- Returns standardized numpy outputs:
  - `edge_index`
  - `node_features`
  - `num_nodes`
  - `train_edges`
  - `valid_edges`
  - `test_edges`
- Provides edge masking and negative sampling for evaluation.

`src/preprocess/preprocess.py`

- Converts raw JSON, JSONL, CSV, TXT, or OGB cache files into stable
  graph-construction inputs.
- Writes `documents.jsonl`, `entities.jsonl`, `relations.jsonl`,
  `triples.jsonl`, `graph_nodes.jsonl`, `graph_edges.jsonl`, and `stats.json`.
- Does not run graph algorithms or call LLM APIs.

`src/llm/`

- Provides an optional LLM extraction scaffold for future custom text datasets.
- Consumes processed `documents.jsonl` and writes raw extraction outputs.
- Uses an OpenAI-compatible Chat Completions API through configurable
  `LLM_BASE_URL`, `LLM_API_KEY`, and `LLM_MODEL`.
- Includes conservative JSON parsing and a mock client for local smoke tests.

`scripts/run_llm_extract.py`

- Runs optional LLM extraction from `documents.jsonl`.
- Supports `--dry-run`, `--mock`, `--limit`, `--resume`, `--sleep`, `--model`,
  `--text-max-chars`, and raw response controls.
- Keeps LLM extraction separate from OGB preprocessing and graph algorithms.

`scripts/smoke_graph_build.py`

- Builds small processed and train-visible graphs from the manual registry.
- Prints basic stats and valid/test candidate shapes.
- Does not run full graph algorithms or enumerate all non-edges.

`utils/download_datasets.py`

- Provides a dedicated dataset download and verification CLI.
- Keeps OGB data preparation out of `main.py`.

`utils/embedder.py`

- Provides a unified `Embedder` interface:
  - `embed(text: str | list[str]) -> np.ndarray`
- Supports deterministic mock embeddings for local demos.
- Supports HTTP embedding APIs with batch requests and simple retry.

`utils/llm_client.py`

- Provides an optional `LLMClient` interface:
  - `generate(prompt: str) -> str`
- Intended only for natural-language interpretation of graph-derived results,
  such as community, path, or link reasoning explanations.

## Future Work

- Multi-view graph fusion.
- Dynamic graph refinement.
- LLM-guided edge weighting.
- Pseudo-GNN using random walk approximation.

These directions should preserve the core principle of this project: graph + embedding + heuristic inference, without turning it into a deep learning or GNN training system.
