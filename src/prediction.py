"""Realtime single-row inference through a saved Spark PipelineModel."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from pyspark.ml import PipelineModel
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StringType,
    DoubleType,
    FloatType,
    IntegerType,
    LongType,
    ShortType,
    ByteType,
    BooleanType,
)

from .feature_engineering import build_features
from .schema import INPUT_SCHEMA


def load_pipeline_model(path: str | Path) -> PipelineModel:
    """Load saved Spark PipelineModel."""
    return PipelineModel.load(str(path))


def _convert_value(value: Any, spark_type: Any) -> Any:
    """
    Convert Streamlit / NumPy / Python values
    into native Python types accepted by PySpark.
    """

    if value is None:
        return None

    # Convert NumPy scalar -> native Python scalar
    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:
            pass

    # String
    if isinstance(spark_type, StringType):
        return str(value)

    # Float / Double
    if isinstance(spark_type, (DoubleType, FloatType)):
        try:
            value = float(value)

            # Avoid NaN / infinity values
            if not math.isfinite(value):
                return None

            return value
        except (TypeError, ValueError):
            return None

    # Integer
    if isinstance(
        spark_type,
        (
            IntegerType,
            LongType,
            ShortType,
            ByteType,
        ),
    ):
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    # Boolean
    if isinstance(spark_type, BooleanType):
        if isinstance(value, str):
            value_lower = value.strip().lower()

            if value_lower in ("true", "1", "yes", "y"):
                return True

            if value_lower in ("false", "0", "no", "n"):
                return False

        return bool(value)

    return value


def _prepare_row(payload: dict[str, Any]) -> dict[str, Any]:
    """
    Prepare one row so every value matches INPUT_SCHEMA.
    """

    row: dict[str, Any] = {}

    for field in INPUT_SCHEMA.fields:
        raw_value = payload.get(field.name)

        row[field.name] = _convert_value(
            raw_value,
            field.dataType,
        )

    return row


def predict_one(
    payload: dict[str, Any],
    spark: SparkSession,
    model: PipelineModel,
) -> dict[str, float]:
    """
    Run realtime prediction for one creator scenario.
    """

    # -----------------------------------------------------
    # 1. Prepare input row
    # -----------------------------------------------------

    row = _prepare_row(payload)

    # Debug information in Streamlit Cloud logs
    print("\n========== PREDICTION INPUT ==========")

    for field in INPUT_SCHEMA.fields:
        value = row.get(field.name)

        print(
            f"{field.name}: "
            f"value={value!r}, "
            f"python_type={type(value).__name__}, "
            f"spark_type={field.dataType.simpleString()}"
        )

    print("======================================\n")

    # -----------------------------------------------------
    # 2. Create Spark DataFrame
    # -----------------------------------------------------

    source = spark.createDataFrame(
        [row],
        schema=INPUT_SCHEMA,
    )

    # -----------------------------------------------------
    # 3. Feature engineering
    # -----------------------------------------------------

    features = build_features(source)

    # -----------------------------------------------------
    # 4. Prediction
    # -----------------------------------------------------

    result = (
        model
        .transform(features)
        .select("prediction_log")
        .first()
    )

    if result is None:
        raise RuntimeError(
            "Spark model returned no prediction."
        )

    prediction_log = float(result["prediction_log"])

    # Revenue model was trained using log1p(revenue)
    predicted_revenue = max(
        0.0,
        math.expm1(prediction_log),
    )

    # -----------------------------------------------------
    # 5. Expected orders
    # -----------------------------------------------------

    product_price = float(
        payload.get("product_price", 0.0) or 0.0
    )

    discount_rate = float(
        payload.get("discount_rate", 0.0) or 0.0
    )

    effective_price = max(
        product_price * (1.0 - discount_rate),
        1.0,
    )

    expected_orders = (
        predicted_revenue / effective_price
    )

    # -----------------------------------------------------
    # 6. ROI / ROAS
    # -----------------------------------------------------

    creator_cost = float(
        payload.get("creator_cost", 0.0) or 0.0
    )

    gross_margin = float(
        payload.get(
            "gross_margin_rate",
            0.35,
        )
        or 0.35
    )

    if creator_cost > 0:

        roi = (
            (
                predicted_revenue * gross_margin
                - creator_cost
            )
            / creator_cost
            * 100.0
        )

        roas = (
            predicted_revenue
            / creator_cost
        )

    else:

        roi = float("nan")
        roas = float("nan")

    # -----------------------------------------------------
    # 7. Return result
    # -----------------------------------------------------

    return {
        "predicted_revenue": float(
            predicted_revenue
        ),
        "expected_orders": float(
            expected_orders
        ),
        "expected_roi_pct": float(roi),
        "expected_roas": float(roas),
    }
