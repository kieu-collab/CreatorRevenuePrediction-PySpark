# Creator Revenue Prediction — Pure PySpark

Project độc lập dự báo doanh thu/GMV kỳ vọng của Creator/KOL cho chiến dịch TikTok Shop mỹ phẩm/FMCG. Toàn bộ data pipeline, feature engineering, train/test, model evaluation, model persistence và inference đều dùng Spark DataFrame + Spark ML PipelineModel.

Repository cũ `kieu-collab/CreatorRevenuePrediction` không bị thay đổi bởi project này.

## Business problem

Trước khi booking KOL, brand nhập thông tin creator, sản phẩm và kế hoạch campaign để ước tính:

- expected revenue/GMV;
- expected orders;
- expected ROI và ROAS;
- độ nhạy doanh thu khi follower, discount hoặc posting frequency thay đổi.

Kết quả là decision-support estimate, không phải cam kết doanh thu hay bằng chứng nhân quả.

## Pure PySpark architecture

```text
CSV / JSON / Parquet on data lake
        ↓ Spark distributed reader
Canonical cleaning with Spark SQL expressions
        ↓ no Python UDF
Leakage-safe Spark feature engineering
        ↓
Creator-level deterministic holdout
        ↓
Spark ML feature hashing + imputation + scaling
        ↓
7 Spark regression candidates
        ↓
MAE · RMSE · MAPE · R²
        ↓
Saved Spark PipelineModel directories
        ↓
Streamlit realtime scoring through Spark DataFrame
```

Không có import `pandas`, `sklearn` hoặc `scikit-learn` trong pipeline. Streamlit có thể kéo theo thư viện riêng của framework, nhưng model code không sử dụng chúng.

## Seven Spark models

| Model | Spark implementation | Role |
|---|---|---|
| Spark Linear Regression | `pyspark.ml.regression.LinearRegression` | Transparent baseline |
| Spark Generalized Linear Regression | `GeneralizedLinearRegression` | Regularized Gaussian/identity baseline |
| Spark Decision Tree | `DecisionTreeRegressor` | Interpretable nonlinear tree |
| Spark Random Forest | `RandomForestRegressor` | Robust bagged tree ensemble |
| Spark GBT Squared | `GBTRegressor(lossType="squared")` | Spark-native boosting focused on larger errors |
| Spark GBT Absolute | `GBTRegressor(lossType="absolute")` | Robust Spark-native boosting |
| Spark Isotonic Power Score | `IsotonicRegression` | Monotonic nonlinear baseline on Creator Power Score |

All candidates share the same Spark preprocessing contract and log-transformed target. The winner is the successful model with the lowest holdout RMSE.

## Leakage-safe features

Pre-campaign features include followers, average views, engagement, price, discount, duration, planned videos/lives, historical conversion and brand fit. Spark derives:

- log followers/views/price;
- views per follower;
- engaged views;
- discounted price;
- planned reach;
- campaign intensity;
- Creator Power Score;
- brand-fit interaction;
- creator tier.

Current-campaign revenue, GMV, orders, Content Efficiency, RPS and ROI are excluded from model inputs.

## Project structure

```text
CreatorRevenuePredictionPySpark/
├── data/creator_campaign.csv
├── artifacts/
│   ├── models/<spark_pipeline_model>/
│   ├── metadata.json
│   ├── metrics.csv
│   ├── feature_importance.csv
│   └── error_analysis.csv
├── notebooks/Creator_Revenue_PySpark.ipynb
├── src/
│   ├── data_pipeline.py
│   ├── evaluation.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   ├── prediction.py
│   ├── schema.py
│   └── spark_session.py
├── tests/
├── app.py
├── train.py
├── packages.txt
├── requirements.txt
└── runtime.txt
```

## Dataset contract

Canonical grain: one row per `creator_id × brand/product × campaign`.

Required training columns:

```text
creator_id, followers, avg_views, revenue
```

Recommended fields:

```text
creator_name, brand_name, creator_niche, product_category, campaign_date,
engagement_rate, product_price, discount_rate, campaign_duration_days,
planned_posts, planned_live_sessions, products_promoted,
historical_conversion_rate, brand_fit_score, creator_cost,
gross_margin_rate, orders, gmv, revenue
```

PySpark reads CSV, JSON/JSONL and Parquet. Convert XLSX to CSV or Parquet before distributed ingestion.

## Local installation

Prerequisites: Python 3.12 and Java 17+.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Train all models

```bash
python train.py \
  --data data/creator_campaign.csv \
  --artifact-dir artifacts \
  --master 'local[*]' \
  --driver-memory 4g \
  --hash-features 128
```

On Dataproc, EMR, Databricks, Kubernetes or Spark standalone, replace `local[*]` with the cluster master/configuration. For large production data, use partitioned Parquet/Delta instead of one CSV.

## Run Streamlit realtime demo

```bash
streamlit run app.py
```

The app loads the selected saved `PipelineModel`, creates a one-row Spark DataFrame, performs the same feature engineering and returns revenue/orders/ROI. It also includes model comparison and scenario simulation pages.

## Google Colab

Open `notebooks/Creator_Revenue_PySpark.ipynb`, run all cells, then download `CreatorRevenuePySpark_artifacts.zip`. Colab is a one-machine Spark runtime (`local[*]`), useful for reproducibility and functional testing. A real big-data benchmark requires a multi-node cluster and a materially larger dataset.

## Streamlit Community Cloud

- `packages.txt` installs OpenJDK 17.
- `requirements.txt` installs PySpark and Streamlit; every estimator comes from Spark MLlib.
- Commit `artifacts/models/` because Streamlit only serves trained immutable artifacts.
- Do not retrain seven models on every app reload.

## Honest big-data statement

The included canonical dataset has 4,798 rows, so it is not big data. This project is implemented with distributed Spark APIs and can scale its processing contract; the speed/cost advantage must be measured again on a large, partitioned campaign history.

## Public links

- GitHub: `https://github.com/kieu-collab/CreatorRevenuePrediction-PySpark`
- Colab: added after the public notebook is created.
- Streamlit: added after the new app is deployed.
