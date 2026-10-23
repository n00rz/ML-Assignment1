"""FastAPI backend: loads both saved models and serves predictions + the HTML page."""
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

app = FastAPI(title="Insurance ML App")

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

reg_model = joblib.load("regression_model.joblib")
clf_model = joblib.load("classification_model.joblib")

Sex = Literal["male", "female"]
Region = Literal["northeast", "northwest", "southeast", "southwest"]


class RegressionInput(BaseModel):
    age: int = Field(ge=18, le=100)
    sex: Sex
    bmi: float = Field(gt=10, lt=70)
    children: int = Field(ge=0, le=10)
    smoker: Literal["yes", "no"]
    region: Region


class ClassificationInput(BaseModel):
    age: int = Field(ge=18, le=100)
    sex: Sex
    bmi: float = Field(gt=10, lt=70)
    children: int = Field(ge=0, le=10)
    region: Region
    charges: float = Field(gt=0)


@app.post("/predict/regression")
def predict_regression(data: RegressionInput):
    row = pd.DataFrame([data.model_dump()])
    price = float(reg_model.predict(row)[0])
    return {"predicted_charges": round(max(price, 0), 2)}


@app.post("/predict/classification")
def predict_classification(data: ClassificationInput):
    row = pd.DataFrame([data.model_dump()])
    label = str(clf_model.predict(row)[0])
    proba = float(clf_model.predict_proba(row)[0].max())
    return {"prediction": label, "is_smoker": label == "yes", "confidence": round(proba, 3)}


@app.get("/")
def home():
    return FileResponse("static/index.html")