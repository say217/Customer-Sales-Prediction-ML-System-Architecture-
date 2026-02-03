# scripts/batch_predict.py
import pickle
import pandas as pd

model = pickle.load(open("customer_sales_prediction.joblib", "rb"))
df = pd.read_csv("data/raw/customer sales prediction dataset.csv")

preds = model.predict(df)
df["monthly_sales_prediction"] = preds

df.to_csv("data/processed/predictions.csv", index=False)
