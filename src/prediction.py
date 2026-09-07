"""Realtime single-row inference through a saved Spark PipelineModel."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from pyspark.ml import PipelineModel
from pyspark.sql import SparkSession

from .feature_engineering import build_features
from .schema import INPUT_SCHEMA


def load_pipeline_model(path: str | Path) -> PipelineModel:
    return PipelineModel.load(str(path))


def predict_one(payload: dict[str, Any], spark: SparkSession, model: PipelineModel) -> dict[str, float]:
    row = {field.name: payload.get(field.name) for field in INPUT_SCHEMA.fields}
    print("ROW:", row)

for key, value in row.items():
    print(
        key,
        value,
        type(value)
    )
    source = spark.createDataFrame([row], schema=INPUT_SCHEMA)
    prediction_log = float(model.transform(build_features(source)).select("prediction_log").first()[0])
    predicted_revenue = max(0.0, math.expm1(prediction_log))
    effective_price = max(float(payload["product_price"]) * (1.0 - float(payload["discount_rate"])), 1.0)
    expected_orders = predicted_revenue / effective_price
    creator_cost = float(payload.get("creator_cost", 0.0))
    gross_margin = float(payload.get("gross_margin_rate", 0.35))
    roi = (
        (predicted_revenue * gross_margin - creator_cost) / creator_cost * 100.0
        if creator_cost > 0
        else float("nan")
    )
    roas = predicted_revenue / creator_cost if creator_cost > 0 else float("nan")
    return {
        "predicted_revenue": predicted_revenue,
        "expected_orders": expected_orders,
        "expected_roi_pct": roi,
        "expected_roas": roas,
    }
