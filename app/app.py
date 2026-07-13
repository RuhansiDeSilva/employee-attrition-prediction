"""
Employee Attrition Prediction App
----------------------------------
A simple FastAPI app with an HTML form UI that lets a user enter the
top 10 most important employee attributes and get an attrition risk
prediction from the trained model.

All other model inputs (that the user isn't asked about) are filled in
automatically using dataset averages/most-common values, stored in
model/defaults.json.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"

app = FastAPI(title="Employee Attrition Predictor")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
# Serve static assets (CSS/JS/images)
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

# ---------------------------------------------------------------------
# Load trained artifacts (place your .pkl files inside the model/ folder)
# ---------------------------------------------------------------------
model = joblib.load(MODEL_DIR / "attrition_model.pkl")
scaler = joblib.load(MODEL_DIR / "robust_scaler.pkl")
label_encoders = joblib.load(MODEL_DIR / "label_encoders.pkl")
model_columns = joblib.load(MODEL_DIR / "model_columns.pkl")

with open(MODEL_DIR / "defaults.json") as f:
    DEFAULTS = json.load(f)

# Columns that were log-transformed during training (Stage 4) because
# they were right-skewed. Must be transformed the same way at inference.
LOG_COLS = [
    "YearsSinceLastPromotion", "YearsAtCompany", "MonthlyIncome",
    "TotalWorkingYears", "NumCompaniesWorked", "DistanceFromHome",
    "YearsInCurrentRole", "YearsWithCurrManager", "PercentSalaryHike",
]

# Multi-category columns that were one-hot encoded during training (Stage 6)
MULTI_CAT_COLS = ["BusinessTravel", "Department", "EducationField",
                   "JobRole", "MaritalStatus"]

SATISFACTION_COLS = ["EnvironmentSatisfaction", "JobSatisfaction",
                      "RelationshipSatisfaction", "WorkLifeBalance"]

# The 10 fields the user actually fills in on the form
FORM_FIELDS = [
    "Age", "MonthlyIncome", "OverTime", "TotalWorkingYears",
    "YearsAtCompany", "JobLevel", "JobSatisfaction",
    "EnvironmentSatisfaction", "WorkLifeBalance", "DistanceFromHome",
]


def build_feature_row(user_input: dict) -> pd.DataFrame:
    """Take the 10 user-provided fields, fill the rest with dataset
    defaults, then replicate the exact preprocessing pipeline used
    during training (log transform -> feature engineering -> encoding
    -> scaling -> column alignment)."""

    row = DEFAULTS.copy()
    row.update(user_input)
    df = pd.DataFrame([row])

    # Make sure numeric fields are actually numeric (form data arrives as strings)
    for col in df.columns:
        if col not in ["BusinessTravel", "Department", "EducationField",
                        "JobRole", "MaritalStatus", "OverTime", "Gender"]:
            df[col] = pd.to_numeric(df[col])

    # --- Stage 4 equivalent: log transform skewed columns ---
    for col in LOG_COLS:
        df[col] = np.log1p(df[col])

    # --- Stage 5 equivalent: feature engineering ---
    df["TenureRatio"] = df["YearsAtCompany"] / (df["TotalWorkingYears"] + 1)
    df["PromotionGap"] = df["YearsSinceLastPromotion"] / (df["YearsAtCompany"] + 1)
    df["IncomePerJobLevel"] = df["MonthlyIncome"] / (df["JobLevel"] + 1)
    df["AvgSatisfaction"] = df[SATISFACTION_COLS].mean(axis=1)

    # --- Stage 6 equivalent: encoding ---
    for col, le in label_encoders.items():
        if col in df.columns:
            df[col] = le.transform(df[col].astype(str))

    df = pd.get_dummies(df, columns=[c for c in MULTI_CAT_COLS if c in df.columns],
                         drop_first=True)

    # Align to the exact columns/order the model was trained on
    df = df.reindex(columns=model_columns, fill_value=0)

    # --- Stage 8 equivalent: scaling (only columns the scaler was fit on) ---
    scale_cols = [c for c in scaler.feature_names_in_ if c in df.columns]
    df[scale_cols] = scaler.transform(df[scale_cols])

    return df


@app.get("/", response_class=HTMLResponse)
def root_welcome(request: Request):
    """Animated landing page shown at root (first page on app start)."""
    return templates.TemplateResponse(request, "animated.html", {})


@app.get("/form", response_class=HTMLResponse)
def form_page(request: Request):
    return templates.TemplateResponse(request, "index.html", {"result": None})


@app.get("/welcome", response_class=HTMLResponse)
def welcome_page(request: Request):
    """Also keep a /welcome route for direct access to the animated page."""
    return templates.TemplateResponse(request, "animated.html", {})


@app.post("/predict", response_class=HTMLResponse)
def predict(
    request: Request,
    Age: float = Form(...),
    MonthlyIncome: float = Form(...),
    OverTime: str = Form(...),
    TotalWorkingYears: float = Form(...),
    YearsAtCompany: float = Form(...),
    JobLevel: float = Form(...),
    JobSatisfaction: float = Form(...),
    EnvironmentSatisfaction: float = Form(...),
    WorkLifeBalance: float = Form(...),
    DistanceFromHome: float = Form(...),
):
    user_input = {
        "Age": Age, "MonthlyIncome": MonthlyIncome, "OverTime": OverTime,
        "TotalWorkingYears": TotalWorkingYears, "YearsAtCompany": YearsAtCompany,
        "JobLevel": JobLevel, "JobSatisfaction": JobSatisfaction,
        "EnvironmentSatisfaction": EnvironmentSatisfaction,
        "WorkLifeBalance": WorkLifeBalance, "DistanceFromHome": DistanceFromHome,
    }

    X = build_feature_row(user_input)
    proba = float(model.predict_proba(X)[:, 1][0])
    pred = "Yes" if proba >= 0.5 else "No"

    result = {
        "prediction": pred,
        "probability": round(proba * 100, 1),
        "risk_level": "High" if proba >= 0.6 else "Moderate" if proba >= 0.3 else "Low",
    }

    return templates.TemplateResponse(request, "index.html", {
        "result": result, "form_input": user_input
    })
