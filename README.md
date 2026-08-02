# Spotter Freight Rate ML Challenge

Production-ready machine learning solution for predicting freight spot rates across US freight corridors, handling data anomalies, eliminating target leakage, implementing rolling time-series backtesting, and generating validated submission files.

---

My article of Route and Delivery Optimization: https://tensour.com/fleet-route-optimization/


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


## 📝 Submission Deliverables
- **Validation Predictions**: `validation_predictions.csv` (12,000 rows, `load_id,predicted_rate`)
- **December Forecast Chart**: `scorer_results/candidate_december.png`
- **Data Quality Insights Documentation**: `data_quality_insights.txt`
- **Jupyter Notebook**: `main.ipynb`
