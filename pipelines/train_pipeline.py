import sys
from pathlib import Path

import yaml
from sklearn.model_selection import train_test_split

# Allow running this file directly (adds project root to sys.path)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.ingetion import load_data
from src.data.preprocessing import remove_outliers_iqr
from src.features.build_features import build_preprocessor
from src.models.train_model import train_model
from src.models.evaluate import evaluate

df = load_data("data/raw/customer sales prediction dataset.csv")

with open("configs/data.yml") as f:
    data_cfg = yaml.safe_load(f)

with open("configs/model.yml") as f:
    model_cfg = yaml.safe_load(f)["random_forest"]

with open("configs/training.yml") as f:
    train_cfg = yaml.safe_load(f)

X = df.drop(data_cfg["target"], axis=1)
y = df[data_cfg["target"]]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=train_cfg["test_size"], random_state=42
)

X_train, y_train = remove_outliers_iqr(
    X_train, y_train, train_cfg["outlier_iqr_multiplier"]
)

train_dir = Path("data/train")
test_dir = Path("data/test")
train_dir.mkdir(parents=True, exist_ok=True)
test_dir.mkdir(parents=True, exist_ok=True)

train_df = X_train.copy()
train_df[data_cfg["target"]] = y_train
train_df.to_csv(train_dir / "train.csv", index=False)

test_df = X_test.copy()
test_df[data_cfg["target"]] = y_test
test_df.to_csv(test_dir / "test.csv", index=False)

preprocessor = build_preprocessor(
    data_cfg["numerical_features"],
    data_cfg["categorical_features"]
)

model = train_model(preprocessor, model_cfg, X_train, y_train)
metrics = evaluate(model, X_test, y_test)

print(metrics)
