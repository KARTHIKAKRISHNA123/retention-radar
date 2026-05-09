# RetentionRadar - Employee Attrition Prediction System For Retention

### An End-to-End Machine Learning Application with Regularized Logistic Regression

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikit-learn)](https://scikit-learn.org)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Deployed-FFD21E?logo=huggingface)](https://huggingface.co)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Project Motivation](#project-motivation)
3. [Live Demo](#live-demo)
4. [Architecture](#architecture)
5. [Dataset](#dataset)
6. [Data Preprocessing & Feature Engineering](#data-preprocessing--feature-engineering)
7. [Model Development & Regularization Strategy](#model-development--regularization-strategy)
8. [Model Serialization](#model-serialization)
9. [Web Application (Streamlit)](#web-application-streamlit)
10. [UI/UX Design](#uiux-design)
11. [Deployment — Hugging Face Spaces](#deployment--hugging-face-spaces)
12. [Project Structure](#project-structure)
13. [Installation & Local Setup](#installation--local-setup)
14. [Troubleshooting & Known Issues](#troubleshooting--known-issues)
15. [Tools & Technologies](#tools--technologies)
16. [Author](#author)

---

## Project Overview

**Employee Attrition Prediction** is a full-stack machine learning application that predicts whether an employee is at risk of leaving an organization. The project covers the complete ML lifecycle: raw data ingestion, feature engineering, model training with three regularization strategies, serialization, interactive web UI development, and cloud deployment on Hugging Face Spaces.

The final application allows HR professionals or managers to input employee attributes and receive an instant, data-driven attrition risk assessment — displayed through a clean, animated dark-mode interface.

---

## Project Motivation

Employee turnover is one of the most costly operational challenges for organizations. According to industry estimates, replacing a single employee can cost anywhere from 50% to 200% of their annual salary when accounting for recruitment, onboarding, lost productivity, and institutional knowledge loss.

Traditional HR processes rely on gut instinct or lagging indicators — exit interviews, resignation letters — which come too late to take preventive action. A machine learning–driven approach changes this: by analyzing patterns in employee data (age, department, tenure, satisfaction scores), organizations can proactively identify at-risk employees and intervene before attrition occurs.

This project demonstrates how a relatively simple yet well-regularized logistic regression model, when paired with a usable web interface, can translate raw workforce data into actionable HR intelligence.

---

## Live Demo

> Deployed on Hugging Face Spaces (Streamlit SDK)

**App URL:** `https://huggingface.co/spaces/KARTHIKAKRISHNA123/employee-attrition-prediction`

**Features of the live app:**
- Lottie-animated AI header
- Real-time attrition prediction on form submission
- Custom dark-mode color palette (blue & gray)
- Zero-setup — runs entirely in the browser via Hugging Face infrastructure

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER BROWSER                            │
│              Hugging Face Spaces (Streamlit Cloud)              │
└───────────────────────────┬─────────────────────────────────────┘
                            │  HTTP Request
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                        app.py  (Streamlit)                      │
│                                                                 │
│  1. Collect Inputs  →  Age (number), Department (dropdown)      │
│  2. Build DataFrame →  pd.DataFrame({...})                      │
│  3. Encode          →  pd.get_dummies() + .reindex()            │
│  4. Load Artifacts  →  joblib.load(l1_model.pkl)                │
│                        joblib.load(model_features.pkl)          │
│  5. Predict         →  model.predict(user_data_encoded)         │
│  6. Display Result  →  st.success() / st.error()                │
└─────────────────────────────────────────────────────────────────┘
                            │
                ┌───────────┴───────────┐
                ▼                       ▼
        l1_model.pkl           model_features.pkl
   (Trained L1 Logistic       (Expected column schema
    Regression weights)        after one-hot encoding)
```

---

## Dataset

| Attribute        | Value                           |
|------------------|---------------------------------|
| File             | `employee_turnover.csv`         |
| Features Used    | `Age`, `Department`             |
| Target Variable  | Attrition (binary: 0 = Stay, 1 = Leave) |
| Encoding         | One-Hot Encoding for `Department` |

> **Note:** The dataset used here is a representative sample for educational/academic purposes.

---

## Data Preprocessing & Feature Engineering

### Categorical Encoding — One-Hot Encoding

The `Department` column contains nominal string values (`HR`, `Sales`, `IT`). Machine learning models require numerical input, so One-Hot Encoding was applied to convert each department into a separate binary column:

```
Department_HR    Department_IT    Department_Sales
     1                0                 0           ← HR employee
     0                1                 0           ← IT employee
     0                0                 1           ← Sales employee
```

**Implementation:**
```python
import pandas as pd

X_encoded = pd.get_dummies(X, columns=["Department"])
```

### Why One-Hot Encoding (not Label Encoding)?

Label Encoding assigns ordinal integers (HR=0, IT=1, Sales=2), which introduces a false ordering relationship — the model would incorrectly interpret "Sales > IT > HR." One-Hot Encoding avoids this by treating each category as completely independent.

### Feature Blueprint Preservation

After encoding, the **exact column order** was saved to `model_features.pkl`. This ensures that at inference time, any user input — regardless of the department selected — is aligned to the same schema the model was trained on. Missing columns are safely filled with `0` via `.reindex(fill_value=0)`.

```python
model_features = list(X_encoded.columns)
joblib.dump(model_features, "model_features.pkl")
```

---

## Model Development & Regularization Strategy

Three Logistic Regression models were trained and compared to evaluate the impact of regularization:

### Models Trained

| Model      | Regularization | Parameter     | Purpose                                      |
|------------|----------------|---------------|----------------------------------------------|
| Baseline   | None           | `penalty=None`| Establishes an unregularized benchmark       |
| L2 (Ridge) | L2             | `C=1.0`       | Shrinks all coefficients — prevents overfitting |
| L1 (Lasso) | L1             | `C=1.0`       | Zeros out less important features — sparse model |

### What is Regularization?

Regularization adds a penalty term to the loss function during training to discourage overly complex models:

```
Loss = Log-Loss + λ × Penalty

L1 Penalty: λ × Σ|wᵢ|         (sum of absolute weights)
L2 Penalty: λ × Σwᵢ²          (sum of squared weights)
```

Where `C = 1/λ` in scikit-learn's API — a smaller `C` means stronger regularization.

### Why L1 Was Selected

L1 regularization was chosen as the production model for two key reasons:

1. **Feature Selection via Sparsity:** L1 drives the weights of redundant or irrelevant features exactly to zero, effectively performing automatic feature selection. This makes the model more interpretable and efficient.

2. **Suitability for High-Dimensional HR Data:** In real-world HR datasets with many correlated features (job level, salary band, job satisfaction), L1 produces a leaner model that focuses on the most predictive signals.

```python
from sklearn.linear_model import LogisticRegression

baseline = LogisticRegression(penalty=None, max_iter=1000)
l2_model = LogisticRegression(penalty='l2', C=1.0, solver='lbfgs', max_iter=1000)
l1_model = LogisticRegression(penalty='l1', C=1.0, solver='liblinear', max_iter=1000)
```

> **Solver Note:** L1 regularization requires `solver='liblinear'` or `solver='saga'` in scikit-learn. The default `lbfgs` solver does not support L1.

---

## Model Serialization

Trained artifacts were serialized using `joblib` for efficient binary storage and fast deserialization at runtime.

```python
import joblib

# Save the production model
joblib.dump(l1_model, "l1_model.pkl")

# Save the feature schema
joblib.dump(model_features, "model_features.pkl")
```

### Saved Artifacts

| File                         | Contents                                      |
|------------------------------|-----------------------------------------------|
| `l1_model.pkl`               | Trained L1 Logistic Regression model weights  |
| `l2_model.pkl`               | Trained L2 model (retained for comparison)    |
| `baseline_logistic_model.pkl`| Baseline model (retained for comparison)      |
| `model_features.pkl`         | Ordered list of feature column names          |

---

## Web Application (Streamlit)

The prediction interface was built with **Streamlit**, a Python-native framework for rapid data application development.

### Inference Pipeline (`app.py`)

```python
import streamlit as st
import pandas as pd
import joblib

# Step 1: Load artifacts
model = joblib.load("l1_model.pkl")
model_features = joblib.load("model_features.pkl")

# Step 2: Collect user inputs
age = st.number_input("Age", min_value=18, max_value=100, value=30)
department = st.selectbox("Department", ["HR", "Sales", "IT"])

# Step 3: Predict on button click
if st.button("Predict Attrition"):
    user_input = pd.DataFrame({"Age": [age], "Department": [department]})

    # Step 4: Encode and align schema
    user_data_encoded = pd.get_dummies(user_input).reindex(
        columns=model_features, fill_value=0
    )

    # Step 5: Predict and display
    prediction = model.predict(user_data_encoded)
    if prediction[0] == 1:
        st.error("Warning: This employee is at high risk of leaving.")
    else:
        st.success("This employee is likely to stay with the company.")
```

### Key Design Decisions

- **`.reindex(fill_value=0)`** — Prevents `KeyError` or shape mismatch when the user selects a department that, after encoding, produces fewer columns than the model expects. All missing columns are safely set to `0`.
- **Button-gated prediction** — The prediction block is wrapped inside `if st.button(...)` to prevent premature execution on page load, which would cause `NameError` if any variable is not yet defined.

---

## UI/UX Design

### Lottie Animation Integration

A Lottie JSON animation (AI/tech themed) was loaded from a remote URL and rendered using `streamlit-lottie`:

```python
import requests
from streamlit_lottie import st_lottie

def load_lottie_url(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_anim = load_lottie_url("https://lottie.host/...")
if lottie_anim:
    st_lottie(lottie_anim, height=200, key="coding")
```

### Custom Dark Mode Theme

A `.streamlit/config.toml` file was created to enforce a professional dark theme with a blue-gray palette:

```toml
[theme]
base = "dark"
primaryColor = "#1E90FF"
backgroundColor = "#0E1117"
secondaryBackgroundColor = "#1C1C2E"
textColor = "#FAFAFA"
```

---

## Deployment — Hugging Face Spaces

The application is hosted on **Hugging Face Spaces** using the Streamlit SDK.

### Hugging Face `README.md` YAML Metadata

```yaml
---
title: Employee Attrition Prediction
emoji: 🧑‍💼
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.x
app_file: app.py
pinned: false
---
```

> **Color Constraint:** Hugging Face Spaces only accepts colors from its predefined palette (`blue`, `green`, `red`, `indigo`, etc.). Custom hex codes in the `colorFrom`/`colorTo` fields are not supported and will cause a YAML validation error.

### `requirements.txt`

```
pandas
scikit-learn
joblib
streamlit
requests
streamlit-lottie
```

All dependencies are pinned at the package level. The Hugging Face environment installs these automatically on Space creation or re-deployment.

---

## Project Structure

```
Mine/
│
├── app.py                                          # Main Streamlit application
├── employee_turnover.csv                           # Source dataset
│
├── employee_attrition_prediction_using_           
│   logistic_regression.ipynb                       # Full training notebook
│
├── l1_model.pkl                                    # Production model (L1 Logistic Regression)
├── l2_model.pkl                                    # L2 model (comparison artifact)
├── baseline_logistic_model.pkl                     # Baseline model (comparison artifact)
├── model_features.pkl                              # Saved feature column schema
│
├── requirements.txt                                # Python dependencies
├── config.toml                                     # Streamlit theme configuration
│
└── .streamlit/
    └── config.toml                                 # (Streamlit reads from here at runtime)
```

---

## Installation & Local Setup

### Prerequisites

- Python 3.10 or higher
- pip

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/KARTHIKAKRISHNA123/employee-attrition-prediction.git
cd employee-attrition-prediction

# 2. (Recommended) Create a virtual environment
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the Streamlit app
streamlit run app.py
```

The app will open automatically at `http://localhost:8501`.

---

## Troubleshooting & Known Issues

### Issue 1 — `NameError` on Page Load

**Symptom:** `NameError: name 'prediction' is not defined` when the app loads.

**Root Cause:** The prediction logic was placed outside the `if st.button(...)` block, causing it to execute before the user clicked Predict.

**Fix:** Wrap all prediction-related code inside the button's conditional block:
```python
if st.button("Predict Attrition"):
    # All inference logic goes here
```

---

### Issue 2 — `ValueError` at Inference (Feature Schema Mismatch)

**Symptom:** `ValueError: X has N features, but model expects M features.`

**Root Cause:** `pd.get_dummies()` on a single-row input only generates columns for the selected department. For example, selecting `"IT"` produces `Department_IT=1` but omits `Department_HR` and `Department_Sales` entirely.

**Fix:** Apply `.reindex()` with `fill_value=0` after encoding to restore all expected columns:
```python
user_data_encoded = pd.get_dummies(user_input).reindex(
    columns=model_features, fill_value=0
)
```

---

### Issue 3 — Hugging Face YAML Color Validation Error

**Symptom:** Space fails to build with a YAML metadata error on `colorFrom` or `colorTo`.

**Root Cause:** Custom hex color codes (e.g., `#1E90FF`) are not accepted in the Hugging Face Space metadata schema.

**Fix:** Use only platform-supported named colors:
```yaml
colorFrom: blue
colorTo: indigo
```

---

## Tools & Technologies

| Category             | Tool / Library           | Purpose                                       |
|----------------------|--------------------------|-----------------------------------------------|
| Language             | Python 3.10+             | Core development language                     |
| ML Framework         | scikit-learn             | Logistic Regression, train/test split         |
| Data Manipulation    | pandas                   | DataFrame operations, One-Hot Encoding        |
| Model Serialization  | joblib                   | Save and load `.pkl` model artifacts          |
| Web Framework        | Streamlit                | Interactive prediction UI                     |
| Animation            | streamlit-lottie         | Lottie JSON animation rendering               |
| HTTP Client          | requests                 | Fetching Lottie animation from remote URL     |
| Notebook             | Jupyter Notebook         | Model development and experimentation         |
| Version Control      | Git + GitHub             | Source code management                        |
| Deployment           | Hugging Face Spaces      | Cloud hosting (Streamlit SDK)                 |
| Theme Configuration  | TOML                     | Streamlit dark-mode custom palette            |

---

## Author

**Karthika Krishna**
Student — Artificial Intelligence & Machine Learning
Prime Institute

- GitHub: [@KARTHIKAKRISHNA123](https://github.com/KARTHIKAKRISHNA123)
- Project Space: [Hugging Face](https://huggingface.co/spaces/KARTHIKAKRISHNA123)

---

*Built as part of the AI/ML coursework. This project demonstrates end-to-end ML application development — from raw data to a deployed, interactive prediction interface.*
