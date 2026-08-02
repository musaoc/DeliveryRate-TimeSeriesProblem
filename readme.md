# Spotter Freight Rate ML Challenge

Production-ready machine learning solution for predicting freight spot rates across US freight corridors, handling data anomalies, eliminating target leakage, implementing rolling time-series backtesting, and generating validated submission files.

---

## 📌 Solution Highlights

- **Data Cleaning & Anomaly Resolution**:
  - Identified **292 corrupted negative-weight loads** in `train-test.csv` (e.g., `-47,500 lb`) and dropped them from training to ensure physical validity.
  - Repaired **145 negative-weight loads** in `validation.csv` via absolute value transformations to maintain full 12,000-load test coverage for scoring.
  - Fixed missing macroeconomic signals (`market_index`, `quote_signal`) with **temporal linear interpolation** by date instead of static global medians.
- **Zero-Leakage Feature Engineering**:
  - **Spatial Geometry**: Haversine distance, route circuity ratio, coordinate deltas ($\Delta \text{lat}, \Delta \text{lon}$), bearing angle, and route midpoints (centroids).
  - **Operational Interactions**: Weight-per-mile, market index $\times$ distance, market index $\times$ quote signal, ton-miles.
  - **Calendar Seasonality**: Day of week, day of year, month-end/month-start flags, and cyclical sine/cosine transforms.
  - **Out-of-Fold (OOF) Smoothed Target Encodings**: $m$-estimate smoothed target encodings for origin, destination, and lane computed strictly out-of-fold with zero target leakage.
- **Expanding-Window Rolling Time-Series Cross-Validation (Backtesting)**:
  - Validated models across 4 sequential forward horizons (Jul, Aug, Sep, Oct 2025) simulating true out-of-sample forward deployment.
  - Models evaluated: **Ridge Regression**, **Random Forest**, **LightGBM**, **XGBoost**, and a **Blended Ensemble**.
- **Model Performance**:
  - LightGBM / XGBoost achieved **MAE ~ $205**, **WAPE ~ 8.6%**, and **$R^2 \approx 0.965$**.
  - Blended ensemble outperforms individual models with lower variance and balanced predictions.

---

## 📂 Repository Structure

```text
├── data_quality_insights.txt    # Detailed log of data quality anomalies, findings, & rationale
├── build_model.py               # Standalone end-to-end Python pipeline script
├── main.ipynb                   # Fully executed Jupyter Notebook with visual storytelling & EDA
├── requirements.txt             # Project dependencies
├── score.py                     # Official validation and scoring script
├── validation_predictions.csv   # Final 12,000 load predictions (load_id, predicted_rate)
├── december-chart-inputs.csv    # 31 December predictions for fixed lane trajectory
├── model_comparison.png         # Cross-validation performance chart across models
├── scorer_results/
│   └── candidate_december.png   # Scorer-generated December 2025 rate trajectory chart
└── readme.md                    # Project documentation
```

---

## 🚀 Quickstart & Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the End-to-End Pipeline
Execute the full data cleaning, rolling cross-validation, retraining, and export pipeline:
```bash
python build_model.py
```
*Or open and run all cells in [main.ipynb](main.ipynb).*

### 3. Run the Official Scorer
Validate both output files and generate the December prediction chart:
```bash
python score.py --predictions validation_predictions.csv --december-predictions december-chart-inputs.csv
```

Expected Output:
```text
Validated 12,000 final predictions.
Validated 31 fixed December predictions.
Created chart: scorer_results\candidate_december.png
Final validation metrics are calculated by Spotter after submission.
```

---

## 📊 Cross-Validation Benchmark Results

| Model | Rolling MAE ($) | Rolling RMSE ($) | $R^2$ Score | WAPE (%) |
| :--- | :---: | :---: | :---: | :---: |
| **Ensemble (Proposed)** | **$199.05** | **$271.31** | **0.9670** | **8.37%** |
| **LightGBM Regressor** | $205.33 | $278.99 | 0.9651 | 8.63% |
| **XGBoost Regressor** | $205.91 | $279.08 | 0.9650 | 8.65% |
| **Random Forest** | $240.08 | $317.58 | 0.9548 | 10.09% |
| **Ridge Regression** | $329.33 | $419.65 | 0.9210 | 13.86% |

---

## 📝 Submission Deliverables
- **Validation Predictions**: `validation_predictions.csv` (12,000 rows, `load_id,predicted_rate`)
- **December Forecast Chart**: `scorer_results/candidate_december.png`
- **Data Quality Insights Documentation**: `data_quality_insights.txt`
- **Jupyter Notebook**: `main.ipynb`
