"""Graph construction layer for OGB-backed topology workflows."""

from .dataset_registry import (
    get_dataset_config,
    list_supported_datasets,
    normalize_dataset_name,
    require_supported_dataset,
)
from .graph_factory import (
    build_networkx_graph_from_processed,
    build_networkx_graph_from_train_split,
    build_visible_graph,
    infer_directed,
)

__all__ = [
    "build_networkx_graph_from_processed",
    "build_networkx_graph_from_train_split",
    "build_visible_graph",
    "get_dataset_config",
    "infer_directed",
    "list_supported_datasets",
    "normalize_dataset_name",
    "require_supported_dataset",
]
