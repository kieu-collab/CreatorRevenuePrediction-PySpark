from pyspark.sql import Row

from src.data_pipeline import clean_data
from src.feature_engineering import build_features
from src.model_training import grouped_creator_split
from src.spark_session import create_spark_session


def test_spark_clean_features_and_group_split():
    spark = create_spark_session("local[1]", "CreatorRevenueTests")
    try:
        rows = [
            Row(
                creator_id=f"c{i // 2}",
                creator_name=f"Creator {i // 2}",
                brand_name=f"Brand {i % 2}",
                creator_niche="Beauty",
                product_category="Skincare",
                campaign_date=f"2026-01-{(i % 28) + 1:02d}",
                followers=10_000.0 + i,
                avg_views=5_000.0 + i,
                engagement_rate=0.04,
                product_price=250_000.0,
                discount_rate=0.10,
                campaign_duration_days=14.0,
                planned_posts=3.0,
                planned_live_sessions=1.0,
                products_promoted=2.0,
                historical_conversion_rate=0.02,
                brand_fit_score=80.0,
                revenue=10_000_000.0 + i,
            )
            for i in range(60)
        ]
        featured = build_features(clean_data(spark.createDataFrame(rows)))
        assert "creator_power_score" in featured.columns
        train, test = grouped_creator_split(featured)
        train_creators = {row.creator_id for row in train.select("creator_id").distinct().collect()}
        test_creators = {row.creator_id for row in test.select("creator_id").distinct().collect()}
        assert train_creators.isdisjoint(test_creators)
    finally:
        spark.stop()
