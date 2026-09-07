"""Leakage-safe marketing feature engineering with native Spark expressions."""

from __future__ import annotations

import math

from pyspark.sql import DataFrame
from pyspark.sql import functions as F


def build_features(df: DataFrame) -> DataFrame:
    followers = F.greatest(F.col("followers"), F.lit(0.0))
    avg_views = F.greatest(F.col("avg_views"), F.lit(0.0))
    product_price = F.greatest(F.col("product_price"), F.lit(0.0))
    safe_followers = F.when(followers > 0, followers)
    duration = F.greatest(F.col("campaign_duration_days"), F.lit(1.0))

    out = df.withColumn(
        "creator_tier",
        F.when(followers < 10_000, "Nano")
        .when(followers < 100_000, "Micro")
        .when(followers < 500_000, "Mid-tier")
        .when(followers < 1_000_000, "Macro")
        .otherwise("Mega"),
    )
    out = out.withColumn("log_followers", F.log1p(followers))
    out = out.withColumn("log_avg_views", F.log1p(avg_views))
    out = out.withColumn("log_product_price", F.log1p(product_price))
    out = out.withColumn("views_per_follower", F.least(avg_views / safe_followers, F.lit(10.0)))
    out = out.withColumn("engaged_views", avg_views * F.col("engagement_rate"))
    out = out.withColumn("discounted_price", product_price * (1.0 - F.col("discount_rate")))
    out = out.withColumn("planned_reach", avg_views * F.greatest(F.col("planned_posts"), F.lit(0.0)))
    out = out.withColumn(
        "campaign_intensity",
        (F.col("planned_posts") + 2.5 * F.col("planned_live_sessions")) / duration,
    )

    follower_score = F.least(F.log1p(followers) / math.log1p(10_000_000), F.lit(1.0))
    views_score = F.least(F.log1p(avg_views) / math.log1p(10_000_000), F.lit(1.0))
    engagement_score = F.greatest(F.lit(0.0), F.least(F.col("engagement_rate") / 0.15, F.lit(1.0)))
    conversion_score = F.greatest(
        F.lit(0.0), F.least(F.col("historical_conversion_rate") / 0.10, F.lit(1.0))
    )
    out = out.withColumn(
        "creator_power_score",
        100.0
        * (
            0.25 * follower_score
            + 0.30 * views_score
            + 0.25 * engagement_score
            + 0.20 * conversion_score
        ),
    )
    return out.withColumn(
        "brand_fit_interaction", F.col("brand_fit_score") / 100.0 * F.col("engaged_views")
    )
