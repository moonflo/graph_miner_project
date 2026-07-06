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
│   ├── embedder.py
│   ├── graph_builder.py
│   ├── graph_algorithms.py
│   ├── llm_explainer.py
│   ├── evaluator.py
├── data/
│   ├── raw/
│   ├── processed/
├── configs/
│   ├── config.yaml
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

## Future Work

- Multi-view graph fusion.
- Dynamic graph refinement.
- LLM-guided edge weighting.
- Pseudo-GNN using random walk approximation.

These directions should preserve the core principle of this project: graph + embedding + heuristic inference, without turning it into a deep learning or GNN training system.
