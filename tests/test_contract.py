from src.data_pipeline import normalize_header
from src.model_training import declared_model_names, model_slug
from src.schema import LEAKAGE_COLUMNS, NUMERIC_FEATURES


def test_headers_and_slugs_are_stable():
    assert normalize_header("Revenue\n(₫)") == "revenue (₫)"
    assert normalize_header("  Shop   Name ") == "shop name"
    assert model_slug("Spark Random Forest") == "spark_random_forest"


def test_feature_contract_excludes_leakage():
    assert not LEAKAGE_COLUMNS.intersection(NUMERIC_FEATURES)


def test_seven_spark_estimators_are_declared():
    names = declared_model_names()
    assert len(names) == 7
    assert all(name.startswith("Spark ") for name in names)
