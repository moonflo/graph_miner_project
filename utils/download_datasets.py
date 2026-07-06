"""Command-line helper for downloading and verifying OGB datasets."""

from __future__ import annotations

import argparse
from collections.abc import Iterable

from utils.data_utils import SUPPORTED_OGB_LINK_DATASETS, load_dataset


def main() -> None:
    args = parse_args()
    names = sorted(SUPPORTED_OGB_LINK_DATASETS) if args.all else args.datasets

    for name in names:
        print_dataset_summary(
            name=name,
            negative_samples=args.negative_samples,
            root=args.root,
            confirm_download=not args.no_confirm_download,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and verify OGB link prediction datasets."
    )
    parser.add_argument(
        "datasets",
        nargs="*",
        choices=sorted(SUPPORTED_OGB_LINK_DATASETS),
        help="Dataset names to download or verify.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download or verify all supported OGB datasets.",
    )
    parser.add_argument(
        "--root",
        default="data/raw",
        help="Dataset cache directory used by OGB.",
    )
    parser.add_argument(
        "--negative-samples",
        type=int,
        default=0,
        help="Generate this many graph-wide negative samples for a smoke check.",
    )
    parser.add_argument(
        "--no-confirm-download",
        action="store_true",
        help="Do not auto-confirm OGB's large-download prompt.",
    )
    args = parser.parse_args()

    if not args.all and not args.datasets:
        parser.error("Provide at least one dataset name or pass --all.")

    return args


def print_dataset_summary(
    *,
    name: str,
    negative_samples: int = 0,
    root: str = "data/raw",
    confirm_download: bool = True,
) -> None:
    dataset = load_dataset(
        name,
        root=root,
        num_negative_samples=negative_samples if negative_samples > 0 else None,
        confirm_download=confirm_download,
    )

    print(f"\nDataset: {dataset['name']}")
    print(f"num_nodes: {dataset['num_nodes']}")
    print(f"edge_index shape: {dataset['edge_index'].shape}")
    if dataset["node_features"] is None:
        print("node_features: None")
    else:
        print(f"node_features shape: {dataset['node_features'].shape}")

    for key in _edge_summary_keys(dataset):
        value = dataset[key]
        if value is not None:
            print(f"{key} shape: {value.shape}")


def _edge_summary_keys(dataset: dict) -> Iterable[str]:
    keys = (
        "train_edges",
        "valid_edges",
        "test_edges",
        "valid_negative_edges",
        "test_negative_edges",
        "negative_edges",
    )
    return (key for key in keys if key in dataset)


if __name__ == "__main__":
    main()
