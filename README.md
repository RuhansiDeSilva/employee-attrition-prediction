
# Employee Attrition Prediction

A machine learning project that predicts whether an employee is likely to leave a company (attrition), based on IBM's HR Analytics Employee Attrition dataset. The project covers the full pipeline — exploratory data analysis, feature engineering, encoding, scaling, class-imbalance handling, model training/tuning, and evaluation — and ships a working **FastAPI web app** that serves live predictions from the trained model.

---

## 📁 Repository Structure

```
employee-attrition-prediction/
├── data/
│   └── Employee.csv                     # Raw dataset (IBM HR Analytics)
├── notebooks/
│   └── Employee_Attrition_Prediction.ipynb   # Full EDA + ML pipeline
├── app/
│   ├── app.py                           # FastAPI application
│   ├── requirements.txt                 # App dependencies
│   ├── model/
│   │   ├── attrition_model.pkl          # Trained XGBoost model
│   │   ├── robust_scaler.pkl            # Fitted RobustScaler
│   │   ├── label_encoders.pkl           # Fitted LabelEncoders (binary cols)
│   │   ├── model_columns.pkl            # Final training column order
│   │   └── defaults.json                # Dataset averages/modes for unseen fields
│   ├── templates/
│   │   ├── animated.html                # Animated landing page
│   │   └── index.html                   # Prediction form + result page
│   └── static/
│       ├── css/                         # Stylesheets
│       └── js/                          # Client-side scripts
└── README.md
```

---

## 📓 Notebook: `Employee_Attrition_Prediction.ipynb`

The notebook walks through the complete data science workflow used to build the deployed model, in the following stages:

1. **Data Loading & Overview**
   Loads `Employee.csv`, inspects shape, data types, missing values, duplicates, and summary statistics.

2. **Exploratory Data Analysis (EDA)**
   - Histograms of all numeric features.
   - Count plots of categorical features.
   - Notes the target imbalance (`Attrition`: ~84% "No" vs ~16% "Yes").
   - Correlation heatmap of numeric features, plus a bar chart of each numeric feature's correlation with attrition.
   - Cramér's V analysis to measure association strength between categorical features and attrition.

3. **Data Cleaning**
   - Detects and drops constant columns (e.g. `EmployeeCount`, `Over18`, `StandardHours`) and ID-like columns (e.g. `EmployeeNumber`).
   - Sanity checks to confirm no constant/ID columns remain.

4. **Outlier Handling & Skew Correction**
   - Visualizes outliers with boxplots.
   - Applies IQR-based capping to continuous columns (e.g. `MonthlyIncome`, `TotalWorkingYears`, `YearsAtCompany`).
   - Checks skewness and applies `log1p` transformation to right-skewed columns (`YearsSinceLastPromotion`, `YearsAtCompany`, `MonthlyIncome`, `TotalWorkingYears`, `NumCompaniesWorked`, `DistanceFromHome`, `YearsInCurrentRole`, `YearsWithCurrManager`, `PercentSalaryHike`).

5. **Feature Engineering**
   Creates four new engineered features:
   - `TenureRatio` – share of total working life spent at the current company.
   - `PromotionGap` – years since last promotion, relative to tenure.
   - `IncomePerJobLevel` – income normalized by job seniority.
   - `AvgSatisfaction` – average of all satisfaction-related scores.

6. **Encoding**
   - Label-encodes binary categorical columns (`Attrition`, `OverTime`, `Gender`).
   - One-hot encodes multi-category columns (`BusinessTravel`, `Department`, `EducationField`, `JobRole`, `MaritalStatus`).
   - Converts boolean dummy columns to integers.

7. **Train/Test Split**
   Stratified 80/20 split on the target to preserve class balance in both sets.

8. **Feature Scaling**
   Fits a `RobustScaler` on continuous numeric training columns only (excluding binary-like columns), then applies it to both train and test sets.

9. **Class Imbalance Handling**
   Applies **SMOTE** to the training set only (never the test set) to balance the minority "Yes" class, with before/after distribution checks.

10. **Model Training & Hyperparameter Tuning**
    Trains and tunes three classifiers with `RandomizedSearchCV` (5-fold CV, optimizing F1):
    - Logistic Regression
    - Random Forest
    - XGBoost

11. **Model Evaluation & Comparison**
    Compares all three models on the held-out test set using classification report, confusion matrix, ROC-AUC, and F1 score, then selects the best performer (XGBoost).

12. **Feature Importance**
    Plots the top 15 most important features from the winning XGBoost model and checks whether the engineered features (`TenureRatio`, `PromotionGap`, `IncomePerJobLevel`, `AvgSatisfaction`) rank among them.

13. **Model Serialization**
    Saves the final artifacts used by the web app: `attrition_model.pkl`, `robust_scaler.pkl`, `label_encoders.pkl`, and `model_columns.pkl`, then reloads them to verify predictions match.

> <!-- 📸 Add screenshots of the notebook outputs here (e.g. EDA plots, correlation heatmap, feature importance chart, model comparison table) -->

---

## 🚀 The Web App: `app/app.py`

A lightweight **FastAPI** application that lets a user enter the **10 most predictive employee attributes** and get an instant attrition risk prediction, without needing to know every feature the model was trained on.

**How it works:**
- The user only fills in: `Age`, `MonthlyIncome`, `OverTime`, `TotalWorkingYears`, `YearsAtCompany`, `JobLevel`, `JobSatisfaction`, `EnvironmentSatisfaction`, `WorkLifeBalance`, `DistanceFromHome`.
- All other model inputs are auto-filled from dataset averages/most-common values stored in `model/defaults.json`.
- `build_feature_row()` replicates the exact training-time preprocessing pipeline at inference time: log-transforming skewed columns → engineering the same derived features → label/one-hot encoding → aligning to `model_columns` → scaling with the saved `RobustScaler`.
- The model returns a probability, which is mapped to a **Low / Moderate / High** risk level and a Yes/No prediction.

**Routes:**

| Route      | Method | Description                                   |
|------------|--------|------------------------------------------------|
| `/`        | GET    | Animated landing page                          |
| `/welcome` | GET    | Same animated landing page (direct access)     |
| `/form`    | GET    | Prediction input form                          |
| `/predict` | POST   | Accepts form data, returns prediction result   |

> <!-- 📸 Add screenshots of the app output here (e.g. landing page, prediction form, and the result screen showing risk level) -->

---

## 🛠️ Deployment / Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/RuhansiDeSilva/employee-attrition-prediction.git
cd employee-attrition-prediction
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
cd app
pip install -r requirements.txt
```

`requirements.txt` includes: `fastapi`, `uvicorn[standard]`, `jinja2`, `python-multipart`, `pandas`, `numpy`, `scikit-learn`, `joblib`, `xgboost`.

### 4. Run the app

From the `app/` directory:

```bash
uvicorn app:app --reload
```

The app will start at **http://127.0.0.1:8000**

- Visit `/` or `/welcome` for the animated landing page.
- Visit `/form` to open the prediction form and submit employee details.
- The form posts to `/predict`, which renders the prediction result on the same page.

### 5. (Optional) Retrain the model

Open `notebooks/Employee_Attrition_Prediction.ipynb` in Jupyter, run all cells against `data/Employee.csv`, and it will regenerate `attrition_model.pkl`, `robust_scaler.pkl`, `label_encoders.pkl`, and `model_columns.pkl`. Copy the newly generated files into `app/model/` to update the deployed model.

### 6. (Optional) Deploy to a server / container

The app is a standard ASGI app, so it can be deployed anywhere `uvicorn`/`gunicorn` + ASGI is supported (e.g. Docker, Render, Railway, EC2, Azure App Service):

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

<img width="1037" height="855" alt="Screenshot 2026-08-11 125700" src="https://github.com/user-attachments/assets/e6aa1ccd-5d97-46b8-ad2a-c42661a7d7a5" />


---

## 📊 Dataset

`data/Employee.csv` is the IBM HR Analytics Employee Attrition & Performance dataset, containing 35 attributes per employee (demographics, job role, compensation, satisfaction scores, tenure, etc.) with `Attrition` (Yes/No) as the target variable.

---

## 🧰 Tech Stack

- **Data/ML:** Python, pandas, numpy, scikit-learn, XGBoost, imbalanced-learn (SMOTE), matplotlib, seaborn, scipy
- **Web app:** FastAPI, Jinja2, Uvicorn
- **Model persistence:** joblib

---

## 📌 Notes

- The scaler, encoders, and column order used in the app **must** match those produced by the notebook — if you retrain the model, make sure to re-export and replace all four `.pkl`/`.json` artifacts together to avoid a mismatch between training-time and inference-time preprocessing.
- SMOTE is applied only to the training data; the test set (and any live prediction input) always reflects the natural class distribution.
