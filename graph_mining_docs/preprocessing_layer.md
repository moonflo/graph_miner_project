# Preprocessing Layer

The preprocessing layer converts raw datasets into stable intermediate files
that graph construction and relation mining code can consume later. It does not
run NetworkX algorithms, link prediction, Louvain community detection, or LLM
extraction.

## Input Layout

The command accepts a single file or one dataset directory:

```bash
python -m src.preprocess.preprocess --input data/raw/ogbl_citation2 --output processed
```

The input must be fine-grained. Pass `data/raw/ogbl_citation2`,
`data/raw/ogbl_collab`, `data/raw/ogbl_ppa`, or a custom dataset directory
directly. Do not pass `data/` or `data/raw/`; parent directories are rejected so
the command always processes exactly one dataset per run.

This matches the current OGB cache layout:

```text
data/raw/
  ogbl_collab/
  ogbl_ppa/
  ogbl_citation2/
```

Example commands:

```bash
python -m src.preprocess.preprocess --input data/raw/ogbl_citation2 --output processed
python -m src.preprocess.preprocess --input data/raw/ogbl_collab --output processed
python -m src.preprocess.preprocess --input data/raw/ogbl_ppa --output processed
```

Generic datasets can use JSON, JSONL, CSV, or TXT files. JSONL invalid lines
are skipped with line-number warnings. Text encoding is read as `utf-8` first
and retried as `utf-8-sig`.

## Output Layout

Each dataset is written under:

```text
processed/
  <dataset_name>/
    documents.jsonl
    entities.jsonl
    relations.jsonl
    triples.jsonl
    graph_nodes.jsonl
    graph_edges.jsonl
    stats.json
```

`documents.jsonl` stores normalized document samples with stable `doc_id`,
title, text, source, original id, split, and loader metadata.

`entities.jsonl` stores explicit entities from fields such as `entities`,
`entity`, `nodes`, or `mentions`. Entity names are whitespace-normalized and
merged by name.

`relations.jsonl` stores explicit relations from fields such as `relations`,
`edges`, `triples`, `spo_list`, or top-level `subject/predicate/object` style
records.

`triples.jsonl` is a simplified human-checkable view generated from
`relations.jsonl`.

`graph_nodes.jsonl` is generated from `entities.jsonl`. The node weight is the
document frequency for generic datasets, or degree-like edge weight for OGB
graph caches.

`graph_edges.jsonl` is generated from `relations.jsonl`. Duplicate
`source/target/relation` rows are merged and their weights are accumulated.

`stats.json` records counts, detected entity/relation fields, skipped samples,
skipped edges, and warnings.

## Empty Entity or Relation Files

The preprocessor only normalizes fields that are already present. If a raw text
dataset has no entity, relation, edge, triple, or SPO fields, it will still
produce `documents.jsonl`, while `entities.jsonl`, `relations.jsonl`,
`triples.jsonl`, `graph_nodes.jsonl`, and `graph_edges.jsonl` may be empty.

This is expected. A later rule-based extractor or LLM-based extractor can add
entities and relations without changing the graph algorithm layer.

OGB link-prediction caches are a special case: they usually have no document
text, but they do include node mappings and edge CSV files. The preprocessor
therefore emits empty `documents.jsonl` and graph-ready entities, relations,
nodes, and edges from the OGB cache.

Optional LLM extraction is handled after this layer by
`scripts/run_llm_extract.py`. It consumes `documents.jsonl` for future custom
text datasets and writes raw files such as `llm_extractions.jsonl`,
`entities.raw.jsonl`, `relations.raw.jsonl`, and `triples.raw.jsonl`. It is not
called by `python -m src.preprocess.preprocess`, and it is not needed for OGB
structure-graph datasets. See `graph_mining_docs/llm_extraction_layer.md`.

## Downstream Graph Construction

The graph construction layer should read:

```text
graph_nodes.jsonl
graph_edges.jsonl
```

and build a NetworkX graph from those files. Future Jaccard, Adamic-Adar,
Louvain, shortest-path, and OGB-style evaluation code should consume these
graph-ready files instead of reading raw JSON/CSV/TXT/OGB files directly.
