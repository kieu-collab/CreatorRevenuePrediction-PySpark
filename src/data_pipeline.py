"""Distributed ingestion and data cleaning implemented only with Spark SQL."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from .schema import RAW_NUMERIC_FEATURES, TARGET

ALIASES = {
    "handle": "creator_id",
    "nickname": "creator_name",
    "shop name": "brand_name",
    "category": "product_category",
    "date range": "campaign_date",
    "views": "total_views",
    "price(₫)": "product_price",
    "shop_avgprice": "product_price",
    "avg. unit price(₫)": "product_price",
    "productcount": "products_promoted",
    "videonum": "planned_posts",
    "livenum": "planned_live_sessions",
    "revenue(₫)": "revenue",
    "item sold": "orders",
    "itemsold": "orders",
}

DEFAULTS: dict[str, Any] = {
    "creator_name": "Unknown creator",
    "brand_name": "Unknown brand",
    "creator_niche": "Beauty & Personal Care",
    "product_category": "Beauty & Personal Care",
    "engagement_rate": 0.04,
    "product_price": 250_000.0,
    "discount_rate": 0.0,
    "campaign_duration_days": 30.0,
    "planned_posts": 1.0,
    "planned_live_sessions": 0.0,
    "products_promoted": 1.0,
    "historical_conversion_rate": 0.02,
    "brand_fit_score": 70.0,
}


def normalize_header(name: object) -> str:
    return re.sub(r"\s+", " ", str(name).replace("\n", " ").strip()).lower()


def read_distributed(spark: SparkSession, source: str, data_format: str | None = None) -> DataFrame:
    detected = (data_format or Path(source).suffix.lstrip(".")).lower()
    if detected in {"csv", "txt"}:
        return spark.read.option("header", True).option("inferSchema", True).csv(source)
    if detected in {"parquet", "pq"}:
        return spark.read.parquet(source)
    if detected in {"json", "jsonl"}:
        return spark.read.json(source)
    raise ValueError("Supported Spark sources: CSV, JSON/JSONL and Parquet. Convert XLSX before ingestion.")


def _compact_number(column_name: str):
    original = F.lower(F.trim(F.col(column_name).cast("string")))
    text = F.regexp_replace(original, "[,₫\\s]", "")
    text = F.regexp_replace(text, "vnd", "")
    is_percent = text.endswith("%")
    text = F.regexp_replace(text, "%$", "")
    suffix = F.substring(text, -1, 1)
    base = F.regexp_replace(text, "[kmb]$", "").cast("double")
    multiplier = (
        F.when(suffix == "k", F.lit(1_000.0))
        .when(suffix == "m", F.lit(1_000_000.0))
        .when(suffix == "b", F.lit(1_000_000_000.0))
        .otherwise(F.lit(1.0))
    )
    parsed = base * multiplier
    return F.when(is_percent, parsed / 100.0).otherwise(parsed)


def clean_data(df: DataFrame, require_target: bool = True) -> DataFrame:
    renamed: dict[str, str] = {}
    for original in df.columns:
        normalized = normalize_header(original)
        renamed[original] = ALIASES.get(normalized, normalized.replace(" ", "_"))
    if len(set(renamed.values())) != len(renamed):
        raise ValueError("Two source columns normalize to the same canonical name.")
    for original, canonical in renamed.items():
        if original != canonical:
            df = df.withColumnRenamed(original, canonical)

    if "avg_views" not in df.columns and "total_views" in df.columns:
        posts = F.col("planned_posts").cast("double") if "planned_posts" in df.columns else F.lit(0.0)
        lives = (
            F.col("planned_live_sessions").cast("double")
            if "planned_live_sessions" in df.columns
            else F.lit(0.0)
        )
        activities = F.greatest(F.coalesce(posts, F.lit(0.0)) + F.coalesce(lives, F.lit(0.0)), F.lit(1.0))
        df = df.withColumn("avg_views", F.col("total_views").cast("double") / activities)

    required = {"creator_id", "followers", "avg_views"}
    if require_target:
        required.add(TARGET)
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    for column, default in DEFAULTS.items():
        if column not in df.columns:
            df = df.withColumn(column, F.lit(default))
    if "campaign_date" not in df.columns:
        df = df.withColumn("campaign_date", F.lit("2025-10-08"))

    numeric_columns = set(RAW_NUMERIC_FEATURES + [TARGET, "orders", "gmv"]).intersection(df.columns)
    for column in sorted(numeric_columns):
        df = df.withColumn(column, _compact_number(column))

    for column in ["engagement_rate", "discount_rate", "historical_conversion_rate"]:
        df = df.withColumn(column, F.when(F.col(column) > 1, F.col(column) / 100.0).otherwise(F.col(column)))
        df = df.withColumn(column, F.greatest(F.lit(0.0), F.least(F.lit(1.0), F.col(column))))
    df = df.withColumn(
        "brand_fit_score", F.greatest(F.lit(0.0), F.least(F.lit(100.0), F.col("brand_fit_score")))
    )
    df = df.withColumn("campaign_date", F.to_date(F.split(F.col("campaign_date").cast("string"), "~")[0]))
    df = df.withColumn("creator_id", F.trim(F.col("creator_id").cast("string")))
    for column in ["creator_name", "brand_name", "creator_niche", "product_category"]:
        df = df.withColumn(column, F.coalesce(F.col(column).cast("string"), F.lit(str(DEFAULTS[column]))))

    valid = F.col("creator_id").isNotNull() & (F.length(F.col("creator_id")) > 0)
    valid = valid & F.col("followers").isNotNull() & (F.col("followers") >= 0)
    valid = valid & F.col("avg_views").isNotNull() & (F.col("avg_views") >= 0)
    if require_target:
        valid = valid & F.col(TARGET).isNotNull() & (F.col(TARGET) >= 0)
    return df.filter(valid).dropDuplicates(["creator_id", "brand_name", "campaign_date"])
