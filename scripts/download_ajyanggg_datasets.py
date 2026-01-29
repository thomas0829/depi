"""Download all datasets from the ajyanggg HuggingFace organization.

This script uses the HuggingFace Hub API to:
1. List all datasets owned by the ajyanggg organization
2. Download each dataset to the local datasets folder

Usage:
    PYTHONPATH=. uv run python3 -m opengvl.scripts.download_ajyanggg_datasets

Options:
    --output-dir: Directory to save datasets (default: ./datasets)
    --token: HuggingFace token (default: from HUGGING_FACE_HUB_TOKEN env var)
"""

import argparse
import os
import shutil
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi, snapshot_download
from loguru import logger


def list_org_datasets(org: str, token: str | None = None) -> list[str]:
    """List all datasets owned by an organization or user."""
    api = HfApi(token=token)
    datasets = api.list_datasets(author=org)
    return [ds.id for ds in datasets]


def download_dataset(
    dataset_id: str,
    output_dir: Path,
    token: str | None = None,
) -> Path:
    """Download a dataset using snapshot_download.

    Args:
        dataset_id: Full dataset ID (e.g., ajyanggg/dataset_name)
        output_dir: Base directory to save datasets
        token: HuggingFace API token

    Returns:
        Path to the downloaded dataset
    """
    dataset_name = dataset_id.split("/")[-1]
    local_dir = output_dir / dataset_name

    logger.info(f"Downloading {dataset_id} to {local_dir}")

    snapshot_download(
        repo_id=dataset_id,
        repo_type="dataset",
        local_dir=str(local_dir),
        token=token,
    )

    return local_dir


def main() -> None:
    """Main entry point for downloading ajyanggg datasets."""
    load_dotenv(override=True)

    parser = argparse.ArgumentParser(
        description="Download all datasets from the ajyanggg HuggingFace organization"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/home/prior/thomas/depi/data"),
        help="Directory to save datasets (default: /home/prior/thomas/depi/data)",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="HuggingFace token (default: from HUGGING_FACE_HUB_TOKEN env var)",
    )
    parser.add_argument(
        "--org",
        type=str,
        default="ajyanggg",
        help="HuggingFace organization/user to download from (default: ajyanggg)",
    )
    args = parser.parse_args()

    token = args.token or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        logger.warning(
            "No HuggingFace token provided. Some datasets may not be accessible. "
            "Set HUGGING_FACE_HUB_TOKEN env var or use --token flag."
        )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    logger.info(f"Fetching dataset list from {args.org}...")
    datasets = list_org_datasets(args.org, token=token)

    if not datasets:
        logger.warning(f"No datasets found for organization: {args.org}")
        return

    logger.info(f"Found {len(datasets)} datasets: {datasets}")

    successful = []
    failed = []

    for i, dataset_id in enumerate(datasets, 1):
        logger.info(f"[{i}/{len(datasets)}] Processing {dataset_id}")
        try:
            local_path = download_dataset(dataset_id, output_dir, token=token)
            successful.append((dataset_id, local_path))
            logger.success(f"Downloaded {dataset_id} to {local_path}")
        except Exception as e:
            logger.error(f"Failed to download {dataset_id}: {e}")
            failed.append((dataset_id, str(e)))

    # Clean up corrupted/unwanted data
    corrupted_path = output_dir / "franka_data/gello_teleop_video/fold_the_towel/20251120_170708"
    if corrupted_path.exists():
        logger.info(f"Removing corrupted data: {corrupted_path}")
        shutil.rmtree(corrupted_path)

    logger.info("=" * 60)
    logger.info("Download Summary")
    logger.info("=" * 60)
    logger.info(f"Total datasets: {len(datasets)}")
    logger.info(f"Successful: {len(successful)}")
    logger.info(f"Failed: {len(failed)}")

    if successful:
        logger.info("\nSuccessfully downloaded:")
        for dataset_id, path in successful:
            logger.info(f"  - {dataset_id} -> {path}")

    if failed:
        logger.warning("\nFailed downloads:")
        for dataset_id, error in failed:
            logger.warning(f"  - {dataset_id}: {error}")


if __name__ == "__main__":
    main()
