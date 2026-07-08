from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from soccer_predictor.data_pipeline import (
    DatasetSplitConfig,
    build_all_datasets,
    summarize_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build soccer predictor datasets.")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Project root folder.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Download the raw source file even if it already exists.",
    )
    parser.add_argument("--validation-start", default="2019-01-01")
    parser.add_argument("--test-start", default="2023-01-01")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split_config = DatasetSplitConfig(
        validation_start=args.validation_start,
        test_start=args.test_start,
    )

    outputs = build_all_datasets(
        args.project_root,
        force_download=args.force_download,
        split_config=split_config,
    )

    dataset = pd.read_csv(outputs["feature_dataset"], parse_dates=["date"], low_memory=False)
    train = pd.read_csv(outputs["train"], parse_dates=["date"], low_memory=False)
    validation = pd.read_csv(outputs["validation"], parse_dates=["date"], low_memory=False)
    test = pd.read_csv(outputs["test"], parse_dates=["date"], low_memory=False)

    summary = summarize_dataset(dataset)
    print("Built international match datasets")
    print(f"  rows: {summary['rows']}")
    print(f"  teams: {summary['teams']}")
    print(f"  tournaments: {summary['tournaments']}")
    print(f"  date range: {summary['first_match']} to {summary['last_match']}")
    print("  splits:")
    print(f"    train: {len(train)} rows")
    print(f"    validation: {len(validation)} rows")
    print(f"    test: {len(test)} rows")
    print("  tournament holdouts:")
    for name, path in outputs.items():
        if name.endswith("_test") and "world_cup" in name:
            holdout = pd.read_csv(path, parse_dates=["date"])
            print(
                f"    {name.removesuffix('_test')}: "
                f"{len(holdout)} test rows, "
                f"{holdout['date'].min().date()} to {holdout['date'].max().date()}"
            )
    print("  outputs:")
    for name, path in outputs.items():
        print(f"    {name}: {path}")


if __name__ == "__main__":
    main()
