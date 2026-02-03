# scripts/serve.py
import os
import sys

sys.path.append(".")  # allow imports from src/

os.environ["FLASK_ENV"] = "production"

from src.serving.api import app

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
