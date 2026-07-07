# LLM Extraction Layer

The LLM extraction layer is an optional scaffold for future custom text
datasets. It consumes normalized `documents.jsonl` files and writes raw entity,
relation, and triple extraction files that can be normalized later.

It is not used for the current OGB datasets. OGB link-prediction caches are
already structure graphs, so they do not need text-based LLM extraction.

## Position In The Pipeline

The intended flow for future unstructured text data is:

```text
raw custom text data
  -> preprocess documents
  -> documents.jsonl
  -> LLM extraction
  -> llm_extractions.jsonl
  -> entities.raw.jsonl
  -> relations.raw.jsonl
  -> triples.raw.jsonl
  -> later normalization
  -> graph_nodes.jsonl / graph_edges.jsonl
```

This layer starts after `documents.jsonl` because the preprocessing layer owns
raw file discovery, text field detection, document ids, source names, and basic
metadata. Keeping those responsibilities there gives the LLM extractor one
stable input contract:

```json
{
  "doc_id": "...",
  "title": "...",
  "text": "...",
  "source": "...",
  "metadata": {}
}
```

The extractor does not read raw dataset directories directly. It also does not
write `graph_nodes.jsonl` or `graph_edges.jsonl`, because LLM output is still
raw evidence-bearing data. A later normalization step should merge aliases,
deduplicate entities, resolve ids, choose canonical relation labels, and only
then produce graph-ready node and edge files.

## Relationship To Preprocessing

`src.preprocess.preprocess` remains the only layer that converts raw files into
the stable processed dataset layout. It is intentionally not modified to call
LLMs. To use LLM extraction, first run preprocessing for a custom text dataset,
then run:

```bash
python scripts/run_llm_extract.py \
  --input data/processed/<dataset_name>/documents.jsonl \
  --output-dir data/processed/<dataset_name> \
  --mock
```

For OGB datasets such as `ogbl-collab`, `ogbl-ppa`, and `ogbl-citation2`, the
current preprocessing layer already emits graph-ready files from structure data.
Their `documents.jsonl` files can be empty, and LLM extraction is not needed.

## Configuration

Real LLM calls use an OpenAI-compatible Chat Completions API. Configure the
client with environment variables or a local `.env` file:

```bash
LLM_API_KEY=xxx
LLM_BASE_URL=https://example.com/v1
LLM_MODEL=your-model-name
LLM_TEMPERATURE=0
LLM_MAX_TOKENS=4096
LLM_TIMEOUT=60
```

`LLM_BASE_URL` is required so the code is not tied to one fixed commercial
platform. API keys are read from configuration only and are not printed.

## Dry Run

Dry-run mode prints the first prompt and does not call an API or write outputs:

```bash
python scripts/run_llm_extract.py \
  --input data/processed/<dataset_name>/documents.jsonl \
  --output-dir data/processed/<dataset_name> \
  --dry-run
```

## Mock Test

Mock mode does not need an API key. It verifies the full file-writing path:

```bash
python scripts/run_llm_extract.py \
  --input data/processed/<dataset_name>/documents.jsonl \
  --output-dir data/processed/<dataset_name> \
  --mock \
  --limit 10 \
  --resume
```

## Small Real Run

Use a small limit before any larger run:

```bash
python scripts/run_llm_extract.py \
  --input data/processed/<dataset_name>/documents.jsonl \
  --output-dir data/processed/<dataset_name> \
  --limit 10 \
  --resume \
  --sleep 0.5 \
  --model your-model-name
```

The `--model` flag overrides `LLM_MODEL`. `--text-max-chars` controls per
document truncation and defaults to `6000`.

## Outputs

The extractor writes these files under `--output-dir`:

- `llm_extractions.jsonl`: one complete extraction result per document,
  including entities, relations, triples, raw response, error, and metadata.
- `entities.raw.jsonl`: flattened entity rows with `source_doc_id`, `source`,
  evidence, aliases, type, and metadata.
- `relations.raw.jsonl`: flattened relation rows with head, tail,
  `relation_type`, confidence, evidence, source document, source, and metadata.
- `triples.raw.jsonl`: flattened subject-predicate-object rows with evidence
  and source document metadata.
- `llm_extract_stats.json`: run statistics, counts, and parse or request
  errors.

All JSONL files are written with `ensure_ascii=False`.

## Resume Behavior

With `--resume`, the script reads existing `llm_extractions.jsonl`, skips
documents whose `doc_id` has already been processed, and then rewrites
`entities.raw.jsonl`, `relations.raw.jsonl`, and `triples.raw.jsonl` from the
complete `llm_extractions.jsonl` record. This avoids duplicate raw rows when a
batch is resumed.

## Future Normalization

This scaffold intentionally stops at raw extraction. The next layer should read
`entities.raw.jsonl`, `relations.raw.jsonl`, and `triples.raw.jsonl`, then
produce canonical `entities.jsonl`, `relations.jsonl`, `graph_nodes.jsonl`, and
`graph_edges.jsonl` after deterministic normalization. That future step should
stay separate from graph algorithms and from OGB structure preprocessing.
