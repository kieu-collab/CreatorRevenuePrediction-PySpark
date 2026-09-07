"""Train and persist seven distributed Spark regression candidates."""

from __future__ import annotations

import csv
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pyspark.ml import Pipeline
from pyspark.ml.feature import FeatureHasher, Imputer, StandardScaler, VectorAssembler
from pyspark.ml.regression import (
    DecisionTreeRegressor,
    GBTRegressor,
    GeneralizedLinearRegression,
    IsotonicRegression,
    LinearRegression,
    RandomForestRegressor,
)
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from .data_pipeline import clean_data, read_distributed
from .evaluation import error_analysis_by_tier, regression_metrics
from .feature_engineering import build_features
from .schema import CATEGORICAL_FEATURES, NUMERIC_FEATURES, SEED, TARGET


def model_slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def declared_model_names() -> list[str]:
    return [
        "Spark Linear Regression",
        "Spark Generalized Linear Regression",
        "Spark Decision Tree",
        "Spark Random Forest",
        "Spark GBT Squared",
        "Spark GBT Absolute",
        "Spark Isotonic Power Score",
    ]


def grouped_creator_split(df: DataFrame, test_buckets: int = 2) -> tuple[DataFrame, DataFrame]:
    """Hash split ensures a creator never crosses train and test sets."""
    bucket = F.pmod(F.xxhash64(F.col("creator_id")), F.lit(10))
    return df.filter(bucket >= test_buckets), df.filter(bucket < test_buckets)


def preprocessing_stages(hash_features: int = 512) -> list[Any]:
    imputed = [f"{column}__imputed" for column in NUMERIC_FEATURES]
    imputer = Imputer(strategy="median", inputCols=NUMERIC_FEATURES, outputCols=imputed)
    hasher = FeatureHasher(
        inputCols=CATEGORICAL_FEATURES,
        outputCol="categorical_features",
        numFeatures=hash_features,
        categoricalCols=CATEGORICAL_FEATURES,
    )
    assembler = VectorAssembler(
        inputCols=imputed + ["categorical_features"], outputCol="unscaled_features", handleInvalid="keep"
    )
    scaler = StandardScaler(
        inputCol="unscaled_features", outputCol="features", withStd=True, withMean=False
    )
    return [imputer, hasher, assembler, scaler]


def model_candidates() -> dict[str, Any]:
    """Return seven Spark estimators; no pandas or scikit-learn estimator is used."""
    models: dict[str, Any] = {
        "Spark Linear Regression": LinearRegression(
            featuresCol="features",
            labelCol="label_log",
            predictionCol="prediction_log",
            maxIter=100,
            regParam=0.1,
            elasticNetParam=0.0,
            standardization=False,
        ),
        "Spark Generalized Linear Regression": GeneralizedLinearRegression(
            featuresCol="features",
            labelCol="label_log",
            predictionCol="prediction_log",
            family="gaussian",
            link="identity",
            maxIter=100,
            regParam=0.1,
        ),
        "Spark Decision Tree": DecisionTreeRegressor(
            featuresCol="features",
            labelCol="label_log",
            predictionCol="prediction_log",
            maxDepth=10,
            minInstancesPerNode=3,
            seed=SEED,
        ),
        "Spark Random Forest": RandomForestRegressor(
            featuresCol="features",
            labelCol="label_log",
            predictionCol="prediction_log",
            numTrees=80,
            maxDepth=8,
            minInstancesPerNode=3,
            featureSubsetStrategy="0.8",
            seed=SEED,
        ),
        "Spark GBT Squared": GBTRegressor(
            featuresCol="features",
            labelCol="label_log",
            predictionCol="prediction_log",
            maxIter=60,
            stepSize=0.05,
            maxDepth=4,
            minInstancesPerNode=3,
            subsamplingRate=0.85,
            seed=SEED,
        ),
        "Spark GBT Absolute": GBTRegressor(
            featuresCol="features",
            labelCol="label_log",
            predictionCol="prediction_log",
            lossType="absolute",
            maxIter=60,
            stepSize=0.05,
            maxDepth=4,
            minInstancesPerNode=3,
            subsamplingRate=0.85,
            seed=SEED,
        ),
        "Spark Isotonic Power Score": IsotonicRegression(
            featuresCol="features",
            labelCol="label_log",
            predictionCol="prediction_log",
            featureIndex=NUMERIC_FEATURES.index("creator_power_score"),
            isotonic=True,
        ),
    }
    return models


def _write_csv(rows: list[dict[str, Any]], path: Path, columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _defaults(df: DataFrame) -> dict[str, Any]:
    defaults: dict[str, Any] = {}
    for feature in NUMERIC_FEATURES:
        value = df.approxQuantile(feature, [0.5], 0.01)
        if value:
            defaults[feature] = float(value[0])
    for feature in ["creator_niche", "product_category", "brand_name"]:
        row = df.groupBy(feature).count().orderBy(F.desc("count"), F.asc(feature)).first()
        defaults[feature] = str(row[feature]) if row and row[feature] is not None else "Unknown"
    defaults.update({"creator_cost": 10_000_000.0, "gross_margin_rate": 0.35})
    return defaults


def _feature_importance(model, hash_features: int) -> list[dict[str, Any]]:
    estimator_model = model.stages[-1]
    values = None
    if hasattr(estimator_model, "featureImportances"):
        values = estimator_model.featureImportances.toArray().tolist()
    elif hasattr(estimator_model, "coefficients"):
        values = [abs(float(value)) for value in estimator_model.coefficients]
    if values is None:
        return []
    names = NUMERIC_FEATURES + [f"categorical_hash_{index}" for index in range(hash_features)]
    return sorted(
        ({"feature": name, "importance": float(value)} for name, value in zip(names, values)),
        key=lambda row: row["importance"],
        reverse=True,
    )


def train_all(
    spark: SparkSession,
    data_path: str,
    artifact_dir: str | Path = "artifacts",
    data_format: str | None = None,
    hash_features: int = 512,
) -> dict[str, Any]:
    output = Path(artifact_dir)
    output.mkdir(parents=True, exist_ok=True)
    clean = clean_data(read_distributed(spark, data_path, data_format), require_target=True)
    featured = build_features(clean).withColumn("label_log", F.log1p(F.col(TARGET))).cache()
    row_count = featured.count()
    creator_count = featured.select("creator_id").distinct().count()
    train, test = grouped_creator_split(featured)
    train, test = train.cache(), test.cache()
    train_rows, test_rows = train.count(), test.count()
    if min(train_rows, test_rows) == 0:
        raise RuntimeError("Creator hash holdout produced an empty train or test set.")

    fitted: dict[str, Any] = {}
    scored: dict[str, DataFrame] = {}
    metrics: list[dict[str, Any]] = []
    failures: dict[str, str] = {}
    stages = preprocessing_stages(hash_features)
    for name, estimator in model_candidates().items():
        try:
            model = Pipeline(stages=stages + [estimator]).fit(train)
            predictions = model.transform(test).withColumn(
                "prediction", F.greatest(F.exp(F.col("prediction_log")) - 1.0, F.lit(0.0))
            )
            metrics.append({"Model": name, **regression_metrics(predictions)})
            fitted[name] = model
            scored[name] = predictions
        except Exception as exc:
            failures[name] = f"{type(exc).__name__}: {exc}"

    if not metrics:
        raise RuntimeError(f"All Spark models failed: {failures}")
    metrics.sort(key=lambda row: (row["RMSE"], row["MAE"]))
    best_name = str(metrics[0]["Model"])
    available_models: dict[str, str] = {}
    for name, model in fitted.items():
        relative_path = f"models/{model_slug(name)}"
        model.write().overwrite().save(str(output / relative_path))
        available_models[name] = relative_path

    best_predictions = scored[best_name]
    best_predictions.select(
        "creator_id", "creator_name", "brand_name", "campaign_date", TARGET, "prediction"
    ).write.mode("overwrite").parquet(str(output / "test_predictions.parquet"))
    _write_csv(metrics, output / "metrics.csv", ["Model", "MAE", "RMSE", "MAPE", "R2"])
    _write_csv(
        error_analysis_by_tier(best_predictions),
        output / "error_analysis.csv",
        ["creator_tier", "rows", "MAE", "MdAPE"],
    )
    importance = _feature_importance(fitted[best_name], hash_features)
    _write_csv(importance, output / "feature_importance.csv", ["feature", "importance"])

    metadata = {
        "project": "CreatorRevenuePredictionPySpark",
        "trained_at_utc": datetime.now(UTC).isoformat(),
        "engine": "PySpark DataFrame + Spark ML Pipeline",
        "spark_version": spark.version,
        "rows": row_count,
        "unique_creators": creator_count,
        "train_rows": train_rows,
        "test_rows": test_rows,
        "split_strategy": "deterministic_hash_holdout_by_creator",
        "target": TARGET,
        "target_transform": "log1p/expm1",
        "best_model": best_name,
        "available_models": available_models,
        "successful_model_count": len(available_models),
        "failed_models": failures,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "categorical_hash_dimensions": hash_features,
        "defaults": _defaults(featured),
        "data_note": "Current 4,798-row dataset validates the Spark architecture; it is not itself big data.",
    }
    (output / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata
