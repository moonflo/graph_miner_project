"""Manual dataset registry for the graph construction layer.

The graph layer intentionally does not auto-discover directories under
data/processed. New datasets must be declared here before loaders or factories
will accept them.
"""

from __future__ import annotations

from .schemas import DatasetConfig


_DATASETS: dict[str, DatasetConfig] = {
    "ogbl_citation2": DatasetConfig(
        canonical_name="ogbl_citation2",
        ogb_name="ogbl-citation2",
        processed_dir_name="ogbl_citation2",
        raw_dir_name="ogbl_citation2",
        task_type="link_prediction",
        directed=True,
        edge_relation_name="cites",
        processed_graph_is_aggregated=False,
        use_official_split_for_metrics=True,
        notes="Directed citation link prediction task; preserve edge direction for formal metrics.",
    ),
    "ogbl_collab": DatasetConfig(
        canonical_name="ogbl_collab",
        ogb_name="ogbl-collab",
        processed_dir_name="ogbl_collab",
        raw_dir_name="ogbl_collab",
        task_type="link_prediction",
        directed=False,
        edge_relation_name="collaborates_with",
        processed_graph_is_aggregated=True,
        use_official_split_for_metrics=True,
        notes="Processed graph_edges are aggregated; use official train split for metrics.",
    ),
    "ogbl_ppa": DatasetConfig(
        canonical_name="ogbl_ppa",
        ogb_name="ogbl-ppa",
        processed_dir_name="ogbl_ppa",
        raw_dir_name="ogbl_ppa",
        task_type="link_prediction",
        directed=False,
        edge_relation_name="associated_with",
        processed_graph_is_aggregated=False,
        use_official_split_for_metrics=True,
        notes="Undirected protein association link prediction task.",
    ),
}


def list_supported_datasets() -> list[str]:
    """Return the manually declared dataset names."""

    return list(_DATASETS)


def normalize_dataset_name(dataset_name: str) -> str:
    """Normalize OGB dash names to this project's underscore names."""

    normalized = dataset_name.strip().replace("-", "_")
    if normalized.startswith("ogbl_"):
        return normalized
    return normalized


def get_dataset_config(dataset_name: str) -> DatasetConfig:
    """Return the registry config for a supported dataset."""

    canonical_name = normalize_dataset_name(dataset_name)
    try:
        return _DATASETS[canonical_name]
    except KeyError as exc:
        supported = ", ".join(list_supported_datasets())
        raise ValueError(
            f"Unsupported dataset '{dataset_name}'. Supported datasets: {supported}. "
            "Add a DatasetConfig entry before using new data."
        ) from exc


def require_supported_dataset(dataset_name: str) -> str:
    """Validate and return the normalized supported dataset name."""

    return get_dataset_config(dataset_name).canonical_name
