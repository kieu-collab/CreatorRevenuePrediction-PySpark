"""Realtime Streamlit demo backed directly by saved Spark PipelineModel artifacts."""

from __future__ import annotations

import csv
import json
import math
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import streamlit as st
from pyspark.sql import functions as F

from src.data_pipeline import clean_data, read_distributed
from src.feature_engineering import build_features
from src.model_training import grouped_creator_split
from src.prediction import load_pipeline_model, predict_one
from src.spark_session import create_spark_session

ROOT = Path(__file__).resolve().parent
ARTIFACT_DIR = ROOT / "artifacts"
METADATA_PATH = ARTIFACT_DIR / "metadata.json"

st.set_page_config(page_title="Creator Revenue · PySpark", page_icon="⚡", layout="wide")
st.markdown(
    """
    <style>
    .block-container {padding-top:1.4rem; padding-bottom:3rem;}
    [data-testid="stMetric"] {background:#f7f9fc; border:1px solid #e3e8ef; padding:15px; border-radius:12px;}
    div[data-testid="stForm"] {border:1px solid #e3e8ef; padding:18px; border-radius:14px;}
    </style>
    """,
    unsafe_allow_html=True,
)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def read_csv_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key in ["MAE", "RMSE", "MAPE", "R2", "importance"]:
            if key in row and row[key] not in {None, ""}:
                row[key] = float(row[key])
    return rows


@st.cache_resource
def get_spark():
    return create_spark_session("local[2]", "CreatorRevenueRealtimeServing")


@st.cache_resource
def get_model(path: str, modified_time: float):
    del modified_time
    return load_pipeline_model(path)


def vnd(value: float) -> str:
    return f"{value:,.0f} ₫"


def bar_spec(rows: list[dict], metric: str, title: str, lower_is_better: bool = True) -> dict:
    data = [
        {"Model": row["Model"].replace("Spark ", ""), "Value": float(row[metric])}
        for row in sorted(rows, key=lambda item: float(item[metric]), reverse=not lower_is_better)
    ]
    return {
        "title": title,
        "data": {"values": data},
        "mark": {"type": "bar", "cornerRadiusTopLeft": 4, "cornerRadiusTopRight": 4},
        "encoding": {
            "x": {"field": "Model", "type": "nominal", "sort": None, "axis": {"labelAngle": -25}},
            "y": {"field": "Value", "type": "quantitative"},
            "color": {"field": "Model", "type": "nominal", "legend": None},
            "tooltip": [
                {"field": "Model", "type": "nominal"},
                {"field": "Value", "type": "quantitative", "format": ",.3f"},
            ],
        },
        "height": 320,
    }


metadata = read_json(METADATA_PATH)
metrics = read_csv_rows(ARTIFACT_DIR / "metrics.csv")
model_paths = {
    name: ARTIFACT_DIR / relative
    for name, relative in metadata.get("available_models", {}).items()
    if (ARTIFACT_DIR / relative).exists()
}

st.title("⚡ Creator Revenue Prediction — Pure PySpark")
st.caption("Spark DataFrame ETL · Spark ML training · Spark PipelineModel realtime inference")

page = st.sidebar.radio(
    "Điều hướng",
    [
        "Creator Revenue Prediction",
        "Creator Analytics Dashboard",
        "Scenario Simulation",
        "Data & Spark Runtime",
    ],
)

if model_paths:
    model_names = list(model_paths)
    best_name = str(metadata.get("best_model", model_names[0]))
    selected_name = st.sidebar.selectbox(
        "Mô hình Spark đang triển khai",
        model_names,
        index=model_names.index(best_name) if best_name in model_names else 0,
    )
    selected_path = model_paths[selected_name]
    spark = get_spark()
    model = get_model(str(selected_path), selected_path.stat().st_mtime)
    defaults = metadata.get("defaults", {})
    st.sidebar.success(f"Đang dùng: {selected_name}")
    st.sidebar.caption(f"Best theo RMSE: {best_name}")
    st.sidebar.caption(f"Spark {metadata.get('spark_version', 'N/A')} · {metadata.get('rows', 0):,} dòng")
else:
    model = None
    spark = None
    defaults = {}
    selected_name = "N/A"
    best_name = "N/A"
    st.sidebar.error("Chưa có Spark PipelineModel trong artifacts/models.")


def creator_form(key: str = "prediction") -> dict:
    with st.form(f"{key}_form"):
        st.subheader("Creator information")
        c1, c2, c3 = st.columns(3)
        followers = c1.number_input("Followers", min_value=0, value=int(defaults.get("followers", 100_000)), step=10_000)
        avg_views = c2.number_input("Average views", min_value=0, value=int(defaults.get("avg_views", 50_000)), step=5_000)
        engagement = c3.number_input("Engagement rate (%)", 0.0, 100.0, float(defaults.get("engagement_rate", 0.04)) * 100, 0.1)

        st.subheader("Product & campaign")
        c4, c5, c6 = st.columns(3)
        price = c4.number_input("Product price (VND)", min_value=1_000, value=int(defaults.get("product_price", 250_000)), step=10_000)
        category = c5.selectbox("Category", ["Skincare", "Makeup", "Fragrance", "Personal Care", "Food & Beverage", "Household", "Other"])
        brand_fit = c6.slider("Brand fit score", 0, 100, int(defaults.get("brand_fit_score", 70)))

        c7, c8, c9, c10 = st.columns(4)
        duration = c7.number_input("Duration (days)", min_value=1, value=int(defaults.get("campaign_duration_days", 30)))
        posts = c8.number_input("Planned videos", min_value=0, value=int(defaults.get("planned_posts", 3)))
        lives = c9.number_input("Live sessions", min_value=0, value=int(defaults.get("planned_live_sessions", 1)))
        discount = c10.number_input("Discount (%)", 0.0, 90.0, float(defaults.get("discount_rate", 0.0)) * 100, 1.0)

        c11, c12, c13 = st.columns(3)
        conversion = c11.number_input("Historical conversion (%)", 0.0, 100.0, float(defaults.get("historical_conversion_rate", 0.02)) * 100, 0.1)
        creator_cost = c12.number_input("Creator cost (VND)", min_value=0, value=int(defaults.get("creator_cost", 10_000_000)), step=1_000_000)
        gross_margin = c13.number_input("Gross margin (%)", 0.0, 100.0, float(defaults.get("gross_margin_rate", 0.35)) * 100, 1.0)
        submitted = st.form_submit_button("Predict Revenue", type="primary", width="stretch")
    return {
        "submitted": submitted,
        "creator_id": f"{key}_creator",
        "creator_name": "Prospective creator",
        "brand_name": "Prospective brand",
        "creator_niche": "Beauty & Personal Care",
        "product_category": category,
        "campaign_date": datetime.now(UTC).date().isoformat(),
        "followers": float(followers),
        "avg_views": float(avg_views),
        "engagement_rate": float(engagement) / 100.0,
        "product_price": float(price),
        "discount_rate": float(discount) / 100.0,
        "campaign_duration_days": float(duration),
        "planned_posts": float(posts),
        "planned_live_sessions": float(lives),
        "products_promoted": 1.0,
        "historical_conversion_rate": float(conversion) / 100.0,
        "brand_fit_score": float(brand_fit),
        "creator_cost": float(creator_cost),
        "gross_margin_rate": float(gross_margin) / 100.0,
    }


if page == "Creator Revenue Prediction":
    payload = creator_form()
    if payload.pop("submitted"):
        if model is None:
            st.error("Chưa có Spark model để dự báo.")
        else:
            with st.spinner("Spark PipelineModel đang chấm điểm..."):
                result = predict_one(payload, spark, model)
            st.session_state["last_payload"] = payload
            st.subheader("Expected campaign outcome")
            a, b, c, d = st.columns(4)
            a.metric("Expected Revenue", vnd(result["predicted_revenue"]))
            b.metric("Expected Orders", f"{result['expected_orders']:,.0f}")
            c.metric("Expected ROI", "N/A" if math.isnan(result["expected_roi_pct"]) else f"{result['expected_roi_pct']:,.1f}%")
            d.metric("Expected ROAS", "N/A" if math.isnan(result["expected_roas"]) else f"{result['expected_roas']:,.2f}x")
            st.info(f"Kết quả được tính trực tiếp bởi {selected_name} đã lưu ở định dạng Spark PipelineModel.")

elif page == "Creator Analytics Dashboard":
    st.header("Creator Analytics Dashboard")
    st.caption("Model comparison, revenue prediction chart, creator ranking and feature importance")
    if not metrics:
        st.info("Chạy train.py để tạo metrics.csv.")
    else:
        selected = next((row for row in metrics if row["Model"] == selected_name), metrics[0])
        a, b, c, d = st.columns(4)
        a.metric("MAE", vnd(selected["MAE"]))
        b.metric("RMSE", vnd(selected["RMSE"]))
        c.metric("MAPE", f"{selected['MAPE']:,.2f}%")
        d.metric("R²", f"{selected['R2']:.4f}")
        left, right = st.columns(2)
        left.vega_lite_chart(bar_spec(metrics, "RMSE", "RMSE — thấp hơn tốt hơn"), width="stretch")
        right.vega_lite_chart(bar_spec(metrics, "MAE", "MAE — thấp hơn tốt hơn"), width="stretch")
        left2, right2 = st.columns(2)
        left2.vega_lite_chart(bar_spec(metrics, "MAPE", "MAPE (%) — thấp hơn tốt hơn"), width="stretch")
        right2.vega_lite_chart(bar_spec(metrics, "R2", "R² — cao hơn tốt hơn", False), width="stretch")
        ranked = sorted(metrics, key=lambda row: (row["RMSE"], row["MAE"]))
        table = [
            {
                "No.": index,
                "Model": row["Model"],
                "MAE": f"{row['MAE']:,.0f}",
                "RMSE": f"{row['RMSE']:,.0f}",
                "MAPE (%)": f"{row['MAPE']:,.2f}",
                "R²": f"{row['R2']:.4f}",
            }
            for index, row in enumerate(ranked, 1)
        ]
        st.subheader("Bảng xếp hạng mô hình")
        st.dataframe(table, width="stretch", hide_index=True)
        importance = read_csv_rows(ARTIFACT_DIR / "feature_importance.csv")[:15]
        if importance:
            st.subheader("Top factors affecting revenue")
            st.vega_lite_chart(
                {
                    "data": {"values": importance},
                    "mark": "bar",
                    "encoding": {
                        "y": {"field": "feature", "type": "nominal", "sort": "-x"},
                        "x": {"field": "importance", "type": "quantitative"},
                        "tooltip": ["feature", "importance"],
                    },
                    "height": 420,
                },
                width="stretch",
            )

        with st.spinner("Spark đang tổng hợp Creator ranking và chấm điểm holdout..."):
            campaign_data = clean_data(
                read_distributed(spark, str(ROOT / "data" / "creator_campaign.csv")),
                require_target=True,
            ).cache()
            ranking_rows = (
                campaign_data.groupBy("creator_id", "creator_name")
                .agg(
                    F.sum("revenue").alias("actual_revenue"),
                    F.sum(F.coalesce(F.col("orders"), F.lit(0.0))).alias("actual_orders"),
                    F.avg("followers").alias("followers"),
                    F.avg("engagement_rate").alias("engagement_rate"),
                )
                .orderBy(F.desc("actual_revenue"))
                .limit(20)
                .collect()
            )
            featured = build_features(campaign_data).withColumn("label_log", F.log1p(F.col("revenue")))
            _, holdout = grouped_creator_split(featured)
            prediction_rows = (
                model.transform(holdout)
                .withColumn("prediction", F.greatest(F.exp(F.col("prediction_log")) - 1.0, F.lit(0.0)))
                .select("creator_name", "revenue", "prediction")
                .orderBy(F.desc("revenue"))
                .limit(25)
                .collect()
            )
            campaign_data.unpersist()

        st.subheader("Revenue prediction chart — actual vs predicted")
        prediction_chart_rows = [
            {
                "Creator": row["creator_name"] or "Unknown",
                "Actual revenue": float(row["revenue"]),
                "Predicted revenue": float(row["prediction"]),
            }
            for row in prediction_rows
        ]
        st.vega_lite_chart(
            {
                "data": {"values": prediction_chart_rows},
                "transform": [{"fold": ["Actual revenue", "Predicted revenue"], "as": ["Series", "Revenue"]}],
                "mark": {"type": "bar", "tooltip": True},
                "encoding": {
                    "x": {"field": "Creator", "type": "nominal", "sort": None, "axis": {"labelAngle": -35}},
                    "xOffset": {"field": "Series"},
                    "y": {"field": "Revenue", "type": "quantitative", "title": "Revenue (VND)"},
                    "color": {"field": "Series", "type": "nominal"},
                },
                "height": 380,
            },
            width="stretch",
        )

        st.subheader("Creator ranking")
        ranking_table = [
            {
                "Rank": index,
                "Creator": row["creator_name"] or row["creator_id"],
                "Creator ID": row["creator_id"],
                "Followers": f"{float(row['followers'] or 0):,.0f}",
                "Engagement": f"{100 * float(row['engagement_rate'] or 0):,.2f}%",
                "Actual revenue": vnd(float(row["actual_revenue"] or 0)),
                "Orders": f"{float(row['actual_orders'] or 0):,.0f}",
            }
            for index, row in enumerate(ranking_rows, 1)
        ]
        st.dataframe(ranking_table, width="stretch", hide_index=True)

elif page == "Scenario Simulation":
    st.header("Scenario Simulation")
    if model is None:
        st.info("Chưa có Spark model.")
    else:
        base = st.session_state.get(
            "last_payload",
            {
                "creator_id": "scenario_creator",
                "creator_name": "Scenario creator",
                "brand_name": "Prospective brand",
                "creator_niche": "Beauty & Personal Care",
                "product_category": "Skincare",
                "campaign_date": datetime.now(UTC).date().isoformat(),
                "followers": float(defaults.get("followers", 100_000)),
                "avg_views": float(defaults.get("avg_views", 50_000)),
                "engagement_rate": float(defaults.get("engagement_rate", 0.04)),
                "product_price": float(defaults.get("product_price", 250_000)),
                "discount_rate": float(defaults.get("discount_rate", 0.0)),
                "campaign_duration_days": float(defaults.get("campaign_duration_days", 30)),
                "planned_posts": float(defaults.get("planned_posts", 3)),
                "planned_live_sessions": float(defaults.get("planned_live_sessions", 1)),
                "products_promoted": 1.0,
                "historical_conversion_rate": float(defaults.get("historical_conversion_rate", 0.02)),
                "brand_fit_score": float(defaults.get("brand_fit_score", 70)),
                "creator_cost": float(defaults.get("creator_cost", 10_000_000)),
                "gross_margin_rate": float(defaults.get("gross_margin_rate", 0.35)),
            },
        )
        c1, c2, c3 = st.columns(3)
        follower_change = c1.slider("Follower change", -50, 100, 20, format="%d%%")
        discount_change = c2.slider("Discount change", -20, 40, 10, format="%d pp")
        posting_change = c3.slider("Posting frequency change", -80, 200, 25, format="%d%%")
        scenario = dict(base)
        scenario["followers"] = max(0.0, base["followers"] * (1.0 + follower_change / 100.0))
        scenario["discount_rate"] = min(0.9, max(0.0, base["discount_rate"] + discount_change / 100.0))
        scenario["planned_posts"] = max(0.0, round(base["planned_posts"] * (1.0 + posting_change / 100.0)))
        baseline = predict_one(base, spark, model)
        changed = predict_one(scenario, spark, model)
        delta = changed["predicted_revenue"] - baseline["predicted_revenue"]
        a, b, c = st.columns(3)
        a.metric("Baseline revenue", vnd(baseline["predicted_revenue"]))
        b.metric("Scenario revenue", vnd(changed["predicted_revenue"]), delta=vnd(delta))
        c.metric("Scenario orders", f"{changed['expected_orders']:,.0f}")
        st.warning("Mô phỏng phản ánh association mô hình học được, không phải causal lift được đảm bảo.")

else:
    st.header("Data & Spark Runtime")
    if metadata:
        st.json(
            {
                "engine": metadata.get("engine"),
                "spark_version": metadata.get("spark_version"),
                "rows": metadata.get("rows"),
                "unique_creators": metadata.get("unique_creators"),
                "split": metadata.get("split_strategy"),
                "successful_models": metadata.get("successful_model_count"),
            }
        )
    st.write("Upload CSV để Spark đọc, chuẩn hóa và kiểm tra schema ngay trong demo. Training đầy đủ chạy ở Colab/cluster bằng `train.py`.")
    uploaded = st.file_uploader("Creator campaign CSV", type=["csv"])
    if uploaded and spark is not None:
        upload_path = Path(tempfile.gettempdir()) / "creator_campaign_upload.csv"
        upload_path.write_bytes(uploaded.getvalue())
        try:
            clean = clean_data(read_distributed(spark, str(upload_path)), require_target=False)
            row_count = clean.count()
            creator_count = clean.select("creator_id").distinct().count()
            st.success(f"Spark validation: {row_count:,} dòng hợp lệ · {creator_count:,} creators")
            st.dataframe([row.asDict(recursive=True) for row in clean.limit(50).collect()], width="stretch", hide_index=True)
        except Exception as exc:
            st.error(f"Spark không thể xử lý file: {exc}")
    st.caption("Demo sử dụng local[2] trên một máy chủ Streamlit; production có thể trỏ cùng code tới Dataproc, EMR, Databricks hoặc Kubernetes.")
