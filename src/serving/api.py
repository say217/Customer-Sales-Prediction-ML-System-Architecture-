import sys
from pathlib import Path

import joblib
import pandas as pd
import yaml
from flask import Flask, request, jsonify

# Allow running this file directly (adds project root to sys.path)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.serving.schemas import validate_request, validate_batch

app = Flask(__name__)

MODEL_PATH = Path("artifacts/customer_sales_prediction.joblib")
TEST_DATA_PATH = Path("data/test/test.csv")
DATA_CONFIG_PATH = Path("configs/data.yml")


def _load_target_column() -> str:
    if DATA_CONFIG_PATH.exists():
        with DATA_CONFIG_PATH.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        return str(cfg.get("target", "monthly_sales"))
    return "monthly_sales"


def _load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Train and save the model first."
        )
    return joblib.load(MODEL_PATH)

@app.route("/health", methods=["GET"])
def health():
    return {
        "status": "ok",
        "model_path": str(MODEL_PATH),
        "model_exists": MODEL_PATH.exists(),
        "test_data_path": str(TEST_DATA_PATH),
        "test_data_exists": TEST_DATA_PATH.exists(),
    }

@app.route("/predict", methods=["POST"])
def predict():
    try:
        payload = request.get_json()
        validate_request(payload)
        model = _load_model()
        df = pd.DataFrame([payload])
        prediction = model.predict(df)[0]

        return jsonify({
            "prediction": float(prediction)
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 400


@app.route("/predict-batch", methods=["POST"])
def predict_batch():
    try:
        payloads = request.get_json()
        validate_batch(payloads)

        model = _load_model()
        df = pd.DataFrame(payloads)
        preds = model.predict(df)

        return jsonify({
            "rows": int(len(df)),
            "predictions": [float(p) for p in preds]
        })
    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 400


@app.route("/predict-test", methods=["GET"])
def predict_test():
    try:
        if not TEST_DATA_PATH.exists():
            return jsonify({
                "error": f"Test data not found at {TEST_DATA_PATH}"
            }), 404

        model = _load_model()
        df = pd.read_csv(TEST_DATA_PATH)

        target = _load_target_column()
        if target in df.columns:
            X = df.drop(columns=[target])
        else:
            X = df

        preds = model.predict(X)
        output = df.copy()
        output["prediction"] = preds

        return jsonify({
            "rows": int(len(output)),
            "predictions": output.to_dict(orient="records")
        })
    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)




