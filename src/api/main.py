# FastAPI application
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, Request
from src.api.pydantic_models import PredictionRequest,PredictionResponse

app = FastAPI()

model_name = "CreditRiskRandomForest"
model_version = '1'
model_uri = "src/mlruns/0/models/m-9e17dfac59bf4352aab1acce8d964513/artifacts"
model = mlflow.sklearn.load_model(model_uri)



@app.get("/")
def root():
    return {"message": "Credit Risk API is alive 🚀"}




# === /predict endpoint ===
@app.post("/predict",response_model=PredictionResponse)
def predict(input_data: PredictionRequest):
    # Convert Pydantic model to DataFrame
    input_df = pd.DataFrame([input_data.dict()])

    # Get fraud probability (class 1)
    risk_proba = model.predict_proba(input_df)[0][1]

    return {"risk_probability": risk_proba}