"""Distributed regression evaluation utilities."""

from __future__ import annotations

from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from .schema import TARGET


def regression_metrics(predictions: DataFrame) -> dict[str, float]:
    metrics = {
        "MAE": RegressionEvaluator(labelCol=TARGET, predictionCol="prediction", metricName="mae").evaluate(
            predictions
        ),
        "RMSE": RegressionEvaluator(labelCol=TARGET, predictionCol="prediction", metricName="rmse").evaluate(
            predictions
        ),
        "R2": RegressionEvaluator(labelCol=TARGET, predictionCol="prediction", metricName="r2").evaluate(
            predictions
        ),
    }
    mape = (
        predictions.filter(F.abs(F.col(TARGET)) > 1e-9)
        .select(F.avg(F.abs((F.col(TARGET) - F.col("prediction")) / F.col(TARGET))) * 100.0)
        .first()[0]
    )
    metrics["MAPE"] = float(mape) if mape is not None else float("nan")
    return {name: float(value) for name, value in metrics.items()}


def error_analysis_by_tier(predictions: DataFrame) -> list[dict[str, float | int | str]]:
    rows = (
        predictions.withColumn("absolute_error", F.abs(F.col(TARGET) - F.col("prediction")))
        .withColumn(
            "absolute_percentage_error",
            F.when(
                F.abs(F.col(TARGET)) > 1e-9,
                F.abs((F.col(TARGET) - F.col("prediction")) / F.col(TARGET)) * 100.0,
            ),
        )
        .groupBy("creator_tier")
        .agg(
            F.count("*").alias("rows"),
            F.avg("absolute_error").alias("MAE"),
            F.expr("percentile_approx(absolute_percentage_error, 0.5)").alias("MdAPE"),
        )
        .orderBy("creator_tier")
        .collect()
    )
    return [row.asDict(recursive=True) for row in rows]
