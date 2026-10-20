# ⚡ SUTM Predictive Maintenance Dashboard

**Academic project · Telkom University · Data courtesy of PT PLN UP3 Bandung**

An interactive Streamlit dashboard that analyzes visual and thermal
inspection data of medium-voltage overhead distribution lines (SUTM),
scores asset health and risk, and ranks poles for maintenance
priority using machine learning.

![Dashboard Overview](docs/dashboard-overview.png)

## 🎯 Problem & Approach

Inspection data for overhead lines is collected in spreadsheets, which
makes it hard to see which feeders and components need attention first.
This project turns 16,419 visual and 1,305 thermal inspection records
(10,789 unique poles) into an interactive dashboard that supports
condition-based maintenance decisions.

**Pipeline:** component condition text → health index (0–3) →
rule-based risk class (Critical / High / Medium / Low) → risk score →
ML model comparison → maintenance priority list.

## ✨ Key Features

| Tab | Contents |
|-----|----------|
| 📊 Overview | KPI cards, risk distribution, risk by ULP, monthly inspection volume |
| ⚠️ Risk Analysis | Risk treemap, top feeders, per-ULP summary |
| 🔬 Health Index | 12-component health index, heatmap, worst-condition ranking |
| 🌡️ Thermal | ΔT distribution and thermal health index |
| 🤖 ML Results | Model comparison (F1-score), cross-validation stability |
| 🎯 Priority List | Sortable maintenance priority table with CSV export |

- Filters by ULP area, risk class and feeder
- Auto-refresh every 30 seconds, or upload new Excel files from the sidebar
- Dark-theme interface

## 🛠 Tech Stack

Python · Streamlit · Pandas · NumPy · Plotly

## 📈 Results

Five classifiers were compared using 5-fold stratified cross-validation
on 10,789 unique poles with 12 visual health-index features.
Risk classes were derived from a rule-based classification of
component conditions.

| Model | Accuracy | F1-Score |
|-------|----------|----------|
| **XGBoost** | **0.9342** | **0.9329** |
| Random Forest | 0.8544 | 0.8578 |
| Decision Tree | 0.8274 | 0.8396 |
| SVM | 0.7886 | 0.7937 |
| KNN | 0.7767 | 0.7685 |

XGBoost achieved the best performance, with a fold-to-fold standard
deviation of ±0.0136.

> **Note:** The dataset is highly imbalanced (about 95.7% Low risk and
> a single Critical pole), so weighted F1 is reported alongside accuracy.
> Because labels come from a rule-based classification, the scores
> reflect how well each model learns that rule from the health-index features.

## 🚀 Getting Started

    git clone https://github.com/Muat01/sutm-predictive-maintenance-dashboard.git
    cd sutm-predictive-maintenance-dashboard
    pip install -r requirements.txt
    streamlit run app.py

The app opens at `http://localhost:8501`.

## 📂 Data Availability

The inspection data was provided by PT PLN UP3 Bandung for an academic
case study and is **not included** in this repository. To request access,
please contact me via [LinkedIn](https://www.linkedin.com/in/muatasem-hadi).

To run the dashboard, place the two inspection Excel files next to
`app.py` (file names must contain `SUTM` and `T1` / `T2`), or upload
them from the dashboard sidebar.

## 📁 Project Structure

    ├── app.py                 # Main Streamlit app
    ├── requirements.txt       # Dependencies
    ├── .streamlit/config.toml # Theme configuration
    └── docs/                  # Screenshots

## Author

**Muatasem Hadi** · MEng Candidate in Industrial Engineering,
Telkom University · [LinkedIn](https://www.linkedin.com/in/muatasem-hadi)
