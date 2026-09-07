"""Command-line entry point for distributed model training."""

from __future__ import annotations

import argparse
import json

from src.model_training import train_all
from src.spark_session import create_spark_session


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Creator Revenue models with pure PySpark.")
    parser.add_argument("--data", default="data/creator_campaign.csv")
    parser.add_argument("--artifact-dir", default="artifacts")
    parser.add_argument("--master", default="local[*]")
    parser.add_argument("--driver-memory", default="4g")
    parser.add_argument("--format", choices=["csv", "json", "jsonl", "parquet"])
    parser.add_argument("--hash-features", type=int, default=512)
    args = parser.parse_args()
    spark = create_spark_session(args.master, "CreatorRevenueTraining", driver_memory=args.driver_memory)
    try:
        metadata = train_all(
            spark,
            data_path=args.data,
            artifact_dir=args.artifact_dir,
            data_format=args.format,
            hash_features=args.hash_features,
        )
        print(json.dumps(metadata, ensure_ascii=False, indent=2))
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
