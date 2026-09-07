"""SparkSession factory shared by training and Streamlit serving."""

from __future__ import annotations

import os
import sys

from pyspark.sql import SparkSession


def create_spark_session(
    master: str = "local[*]", app_name: str = "CreatorRevenuePredictionPySpark", driver_memory: str = "1g"
):
    # Force driver and workers to use the same interpreter in venv/Colab/Streamlit.
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
    return (
        SparkSession.builder.appName(app_name)
        .master(master)
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.driver.memory", driver_memory)
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
