"""Shared feature and input contracts for batch training and realtime scoring."""

from __future__ import annotations

from pyspark.sql.types import DoubleType, StringType, StructField, StructType

TARGET = "revenue"
SEED = 42

RAW_NUMERIC_FEATURES = [
    "followers",
    "avg_views",
    "engagement_rate",
    "product_price",
    "discount_rate",
    "campaign_duration_days",
    "planned_posts",
    "planned_live_sessions",
    "products_promoted",
    "historical_conversion_rate",
    "brand_fit_score",
]

ENGINEERED_NUMERIC_FEATURES = [
    "log_followers",
    "log_avg_views",
    "log_product_price",
    "views_per_follower",
    "engaged_views",
    "discounted_price",
    "planned_reach",
    "campaign_intensity",
    "creator_power_score",
    "brand_fit_interaction",
]

NUMERIC_FEATURES = RAW_NUMERIC_FEATURES + ENGINEERED_NUMERIC_FEATURES
CATEGORICAL_FEATURES = ["creator_niche", "product_category", "creator_tier", "brand_name"]

LEAKAGE_COLUMNS = {
    "revenue",
    "gmv",
    "orders",
    "live_gmv",
    "video_gmv",
    "showcase_gmv",
    "content_efficiency",
    "creator_roi",
    "rps",
}

INPUT_SCHEMA = StructType(
    [
        StructField("creator_id", StringType(), False),
        StructField("creator_name", StringType(), True),
        StructField("brand_name", StringType(), True),
        StructField("creator_niche", StringType(), True),
        StructField("product_category", StringType(), True),
        StructField("campaign_date", StringType(), True),
        *[StructField(name, DoubleType(), True) for name in RAW_NUMERIC_FEATURES],
    ]
)
