import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

# Ensure project root is on sys.path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.build_features import build_preprocessor
from src.serving.api import app


def _load_data_config():
    cfg_path = PROJECT_ROOT / "configs" / "data.yml"
    if cfg_path.exists():
        with cfg_path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def _load_sample_dataframe() -> pd.DataFrame:
    candidates = [
        PROJECT_ROOT / "data" / "test" / "test.csv",
        PROJECT_ROOT / "data" / "raw" / "customer sales prediction dataset.csv",
    ]
    for path in candidates:
        if path.exists():
            return pd.read_csv(path)

    # Fallback synthetic data if files are missing
    return pd.DataFrame(
        {
            "age": [25, 40, 55],
            "annual_income": [30000, 60000, 90000],
            "website_visits": [10, 20, 30],
            "time_on_site": [25.0, 45.0, 60.0],
            "discount_rate": [0.1, 0.2, 0.15],
            "past_purchases": [2, 5, 7],
            "region": ["North", "South", "East"],
            "device_type": ["mobile", "desktop", "tablet"],
            "membership_level": ["basic", "silver", "gold"],
            "monthly_sales": [100.0, 200.0, 300.0],
        }
    )


@pytest.fixture(scope="session")
def sample_df() -> pd.DataFrame:
    return _load_sample_dataframe()


@pytest.fixture(scope="session")
def X_sample(sample_df: pd.DataFrame) -> pd.DataFrame:
    cfg = _load_data_config()
    target = cfg.get("target", "monthly_sales")
    if target in sample_df.columns:
        return sample_df.drop(columns=[target])
    return sample_df.copy()


@pytest.fixture(scope="session")
def preprocessor():
    cfg = _load_data_config()
    num_cols = cfg.get(
        "numerical_features",
        [
            "age",
            "annual_income",
            "website_visits",
            "time_on_site",
            "discount_rate",
            "past_purchases",
        ],
    )
    cat_cols = cfg.get(
        "categorical_features",
        ["region", "device_type", "membership_level"],
    )
    return build_preprocessor(num_cols, cat_cols)


@pytest.fixture(scope="session")
def model(preprocessor, sample_df: pd.DataFrame):
    cfg = _load_data_config()
    target = cfg.get("target", "monthly_sales")
    if target in sample_df.columns:
        X = sample_df.drop(columns=[target])
        y = sample_df[target]
    else:
        X = sample_df.copy()
        y = pd.Series([0] * len(sample_df))

    model = RandomForestRegressor(
        n_estimators=10,
        max_depth=5,
        random_state=42,
    )

    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("model", model),
    ])

    pipeline.fit(X, y)
    return pipeline


@pytest.fixture()
def client():
    app.config["TESTING"] = True
    with app.test_client() as test_client:
        yield test_client
