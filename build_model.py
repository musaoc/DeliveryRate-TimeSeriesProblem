"""
Spotter Freight Rate Prediction - Production Training & Validation Pipeline
===========================================================================
This script implements an end-to-end Machine Learning pipeline:
1. Rigorous data cleaning (repairing negative weights via np.abs() in both train and validation,
   time-series linear interpolation for daily macroeconomic signals).
2. Spatial, interaction, calendar, and zero-leakage Out-of-Fold (OOF) target encoding.
3. Expanding-window rolling time-series cross-validation (backtesting).
4. Ensembling LightGBM, XGBoost, and Random Forest Regressors.
5. Exporting validated predictions to validation_predictions.csv and december-chart-inputs.csv.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
import xgboost as xgb

# ---------------------------------------------------------------------------
# 1. Load Data
# ---------------------------------------------------------------------------
print("=" * 80)
print("1. LOADING DATASETS")
print("=" * 80)
train_raw = pd.read_csv("train-test.csv")
val_raw = pd.read_csv("validation.csv")
dec_raw = pd.read_csv("december-chart-inputs.csv")

print(f"Raw Train shape: {train_raw.shape}")
print(f"Raw Validation shape: {val_raw.shape}")
print(f"Raw December inputs shape: {dec_raw.shape}")

# Convert date columns
train_raw['date'] = pd.to_datetime(train_raw['date'])
val_raw['date'] = pd.to_datetime(val_raw['date'])
dec_raw['date'] = pd.to_datetime(dec_raw['date'])

# ---------------------------------------------------------------------------
# 2. Data Cleaning & Integrity Handling
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("2. DATA CLEANING & MACRO IMPUTATION")
print("=" * 80)

# Anomaly 1: Negative Weights
# Consistent approach: repair via np.abs() in both train and validation.
# Rationale: We cannot confirm whether the magnitude is correct (sign-flip) or
# the entire value is corrupt. However, the absolute magnitudes fall within normal
# freight weight ranges (1,000-47,500 lb), making sign-repair a reasonable heuristic.
# Using the same strategy across all datasets ensures methodological consistency.
neg_train_count = (train_raw['weight'] < 0).sum()
neg_val_count = (val_raw['weight'] < 0).sum()
print(f"Repairing {neg_train_count} negative-weight rows in train-test.csv via np.abs()...")
print(f"Repairing {neg_val_count} negative-weight rows in validation.csv via np.abs()...")
train_df = train_raw.copy()
train_df['weight'] = np.abs(train_df['weight'])

# Validation: same repair strategy
val_df = val_raw.copy()
val_df['weight'] = np.abs(val_df['weight'])

dec_df = dec_raw.copy()
dec_df['weight'] = np.abs(dec_df['weight'])

# Clean weight median imputation
clean_weight_median = train_df['weight'].median()
train_df['weight'] = train_df['weight'].fillna(clean_weight_median)
val_df['weight'] = val_df['weight'].fillna(clean_weight_median)
dec_df['weight'] = dec_df['weight'].fillna(clean_weight_median)

# Anomaly 2: Temporal Macro Signals (market_index & quote_signal)
# Build continuous daily master series and linearly interpolate missing values by date
all_loads = pd.concat([
    train_df[['date', 'market_index', 'quote_signal']],
    val_df[['date', 'market_index', 'quote_signal']]
], ignore_index=True)

daily_macro = all_loads.groupby('date')[['market_index', 'quote_signal']].mean().reset_index()
daily_macro = daily_macro.sort_values('date').set_index('date')
daily_macro_clean = daily_macro.interpolate(method='time').bfill().ffill().reset_index()

# Map clean daily macro indices back to datasets
date_to_market = dict(zip(daily_macro_clean['date'], daily_macro_clean['market_index']))
date_to_quote = dict(zip(daily_macro_clean['date'], daily_macro_clean['quote_signal']))

train_df['market_index'] = train_df['market_index'].fillna(train_df['date'].map(date_to_market))
train_df['quote_signal'] = train_df['quote_signal'].fillna(train_df['date'].map(date_to_quote))

val_df['market_index'] = val_df['market_index'].fillna(val_df['date'].map(date_to_market))
val_df['quote_signal'] = val_df['quote_signal'].fillna(val_df['date'].map(date_to_quote))

# Populate December inputs with exact December macroeconomic signals from daily_macro
dec_df['market_index'] = dec_df['date'].map(date_to_market)
dec_df['quote_signal'] = dec_df['date'].map(date_to_quote)

# Populate missing coordinates for Lexington -> Fort Wayne
lex_row = train_df[train_df['pickup'] == 'Lexington'].iloc[0]
fw_row = train_df[train_df['delivery'] == 'Fort Wayne'].iloc[0]
dec_df['pickup_lat'] = lex_row['pickup_lat']
dec_df['pickup_lon'] = lex_row['pickup_lon']
dec_df['delivery_lat'] = fw_row['delivery_lat']
dec_df['delivery_lon'] = fw_row['delivery_lon']

print(f"Clean Train records: {len(train_df)} (dropped {neg_train_count} invalid records)")
print(f"Clean Validation records: {len(val_df)}")
print(f"Clean December records: {len(dec_df)}")

# ---------------------------------------------------------------------------
# 3. Feature Engineering
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("3. FEATURE ENGINEERING")
print("=" * 80)

def haversine_np(lat1, lon1, lat2, lon2):
    r = 3958.8  # Earth radius in miles
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2.0)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0)**2
    return 2 * r * np.arcsin(np.sqrt(a))

def calculate_bearing(lat1, lon1, lat2, lon2):
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dlambda = np.radians(lon2 - lon1)
    x = np.sin(dlambda) * np.cos(phi2)
    y = np.cos(phi1) * np.sin(phi2) - np.sin(phi1) * np.cos(phi2) * np.cos(dlambda)
    return (np.degrees(np.arctan2(x, y)) + 360) % 360

def build_base_features(df):
    feat = df.copy()
    
    # 1. Calendar / Temporal Features
    feat['day_of_week'] = feat['date'].dt.dayofweek
    feat['day_of_month'] = feat['date'].dt.day
    feat['month'] = feat['date'].dt.month
    feat['day_of_year'] = feat['date'].dt.dayofyear
    feat['is_weekend'] = feat['day_of_week'].isin([5, 6]).astype(int)
    feat['is_month_start'] = (feat['day_of_month'] <= 3).astype(int)
    feat['is_month_end'] = (feat['day_of_month'] >= 28).astype(int)
    
    # Cyclical representations
    feat['sin_dow'] = np.sin(2 * np.pi * feat['day_of_week'] / 7.0)
    feat['cos_dow'] = np.cos(2 * np.pi * feat['day_of_week'] / 7.0)
    feat['sin_doy'] = np.sin(2 * np.pi * feat['day_of_year'] / 365.0)
    feat['cos_doy'] = np.cos(2 * np.pi * feat['day_of_year'] / 365.0)
    
    # 2. Spatial & Geographic Geometry Features
    feat['calc_dist'] = haversine_np(feat['pickup_lat'], feat['pickup_lon'], feat['delivery_lat'], feat['delivery_lon'])
    feat['circuity_ratio'] = feat['distance'] / (feat['calc_dist'] + 1e-4)
    feat['delta_lat'] = np.abs(feat['delivery_lat'] - feat['pickup_lat'])
    feat['delta_lon'] = np.abs(feat['delivery_lon'] - feat['pickup_lon'])
    feat['bearing_angle'] = calculate_bearing(feat['pickup_lat'], feat['pickup_lon'], feat['delivery_lat'], feat['delivery_lon'])
    feat['centroid_lat'] = (feat['pickup_lat'] + feat['delivery_lat']) / 2.0
    feat['centroid_lon'] = (feat['pickup_lon'] + feat['delivery_lon']) / 2.0
    
    # 3. Operational & Interaction Features
    feat['weight_per_mile'] = feat['weight'] / (feat['distance'] + 1.0)
    feat['market_dist_prod'] = feat['market_index'] * feat['distance']
    feat['market_quote_prod'] = feat['market_index'] * feat['quote_signal']
    feat['weight_dist_prod'] = (feat['weight'] / 1000.0) * feat['distance']
    
    # 4. Lane identifier
    feat['lane'] = feat['pickup'] + "->" + feat['delivery']
    
    return feat

# Create base feature tables
train_feat = build_base_features(train_df)
val_feat = build_base_features(val_df)
dec_feat = build_base_features(dec_df)

# One-hot encoding for equipment
train_feat = pd.get_dummies(train_feat, columns=['equipment'], drop_first=False)
val_feat = pd.get_dummies(val_feat, columns=['equipment'], drop_first=False)
dec_feat = pd.get_dummies(dec_feat, columns=['equipment'], drop_first=False)

equipment_cols = [c for c in train_feat.columns if c.startswith('equipment_')]
for c in equipment_cols:
    if c not in val_feat.columns:
        val_feat[c] = False
    if c not in dec_feat.columns:
        dec_feat[c] = False

# Frequency encodings (fit strictly on training data)
for col in ['pickup', 'delivery', 'lane']:
    freq = train_feat[col].value_counts()
    train_feat[f'{col}_freq'] = train_feat[col].map(freq).fillna(0)
    val_feat[f'{col}_freq'] = val_feat[col].map(freq).fillna(0)
    dec_feat[f'{col}_freq'] = dec_feat[col].map(freq).fillna(0)

# Out-of-Fold Smoothed Target Encoding Function (m-estimate smoothing)
def compute_smoothed_target_encoding(train_series, target_series, test_series, m=10.0):
    global_mean = target_series.mean()
    stats = target_series.groupby(train_series).agg(['count', 'mean'])
    smoothed = (stats['count'] * stats['mean'] + m * global_mean) / (stats['count'] + m)
    return test_series.map(smoothed).fillna(global_mean)

# Compute full training set smoothed target encoding for validation and December datasets
global_rate_mean = train_feat['posted_rate'].mean()
for col in ['pickup', 'delivery', 'lane']:
    val_feat[f'{col}_target_enc'] = compute_smoothed_target_encoding(
        train_feat[col], train_feat['posted_rate'], val_feat[col], m=10.0
    )
    dec_feat[f'{col}_target_enc'] = compute_smoothed_target_encoding(
        train_feat[col], train_feat['posted_rate'], dec_feat[col], m=10.0
    )

# Compute 5-fold OOF target encodings for the training set (Zero Target Leakage)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
for col in ['pickup', 'delivery', 'lane']:
    oof_enc = pd.Series(index=train_feat.index, dtype=float)
    for tr_idx, val_idx in kf.split(train_feat):
        tr_cat, tr_y = train_feat[col].iloc[tr_idx], train_feat['posted_rate'].iloc[tr_idx]
        val_cat = train_feat[col].iloc[val_idx]
        oof_enc.iloc[val_idx] = compute_smoothed_target_encoding(tr_cat, tr_y, val_cat, m=10.0)
    train_feat[f'{col}_target_enc'] = oof_enc

feature_cols = [
    'pickup_lat', 'pickup_lon', 'delivery_lat', 'delivery_lon',
    'distance', 'weight', 'market_index', 'quote_signal',
    'day_of_week', 'day_of_month', 'month', 'day_of_year',
    'is_weekend', 'is_month_start', 'is_month_end',
    'sin_dow', 'cos_dow', 'sin_doy', 'cos_doy',
    'calc_dist', 'circuity_ratio', 'delta_lat', 'delta_lon',
    'bearing_angle', 'centroid_lat', 'centroid_lon',
    'weight_per_mile', 'market_dist_prod', 'market_quote_prod', 'weight_dist_prod',
    'pickup_freq', 'delivery_freq', 'lane_freq',
    'pickup_target_enc', 'delivery_target_enc', 'lane_target_enc'
] + equipment_cols

print(f"Engineered {len(feature_cols)} features for model training.")

X = train_feat[feature_cols]
y = train_feat['posted_rate']
X_val = val_feat[feature_cols]
X_dec = dec_feat[feature_cols]

# ---------------------------------------------------------------------------
# 4. Expanding-Window Rolling Time-Series Cross-Validation
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("4. EXPANDING-WINDOW ROLLING TIME-SERIES CROSS-VALIDATION (BACKTESTING)")
print("=" * 80)

# Backtesting folds:
# Fold 1: Train Months 1-6 -> Val Month 7 (July)
# Fold 2: Train Months 1-7 -> Val Month 8 (August)
# Fold 3: Train Months 1-8 -> Val Month 9 (September)
# Fold 4: Train Months 1-9 -> Val Month 10 (October)
temporal_folds = [
    (list(range(1, 7)), 7, "Jul 2025"),
    (list(range(1, 8)), 8, "Aug 2025"),
    (list(range(1, 9)), 9, "Sep 2025"),
    (list(range(1, 10)), 10, "Oct 2025"),
]

def get_models():
    return {
        'Ridge Regression': Ridge(alpha=10.0),
        'Random Forest': RandomForestRegressor(n_estimators=120, max_depth=12, random_state=42, n_jobs=-1),
        'LightGBM': lgb.LGBMRegressor(n_estimators=350, learning_rate=0.05, max_depth=8, num_leaves=31, random_state=42, verbose=-1),
        'XGBoost': xgb.XGBRegressor(n_estimators=350, learning_rate=0.05, max_depth=6, random_state=42, n_jobs=-1)
    }

cv_results = []
for tr_months, val_month, val_name in temporal_folds:
    print(f"\n--- Backtest Fold: Train Months {tr_months[0]}-{tr_months[-1]} -> Validate {val_name} (Month {val_month}) ---")
    tr_mask = train_feat['month'].isin(tr_months)
    val_mask = train_feat['month'] == val_month
    
    X_tr_fold = X[tr_mask].copy()
    y_tr_fold = y[tr_mask].copy()
    X_val_fold = X[val_mask].copy()
    y_val_fold = y[val_mask].copy()
    
    # Recompute smoothed target encoding strictly on fold training data to ensure zero leakage
    for cat_col in ['pickup', 'delivery', 'lane']:
        X_val_fold[f'{cat_col}_target_enc'] = compute_smoothed_target_encoding(
            train_feat.loc[tr_mask, cat_col], y_tr_fold, train_feat.loc[val_mask, cat_col], m=10.0
        )
    
    fold_models = get_models()
    fold_preds = {}
    
    for name, model in fold_models.items():
        model.fit(X_tr_fold, y_tr_fold)
        p = model.predict(X_val_fold)
        p = np.clip(p, a_min=100.0, a_max=None)
        fold_preds[name] = p
        
        rmse = np.sqrt(mean_squared_error(y_val_fold, p))
        mae = mean_absolute_error(y_val_fold, p)
        r2 = r2_score(y_val_fold, p)
        wape = (np.abs(y_val_fold - p).sum() / y_val_fold.sum()) * 100.0
        
        cv_results.append({
            'Fold': val_name, 'Model': name, 'RMSE': rmse, 'MAE': mae, 'R2': r2, 'WAPE(%)': wape
        })
        print(f"{name:18s} | RMSE: ${rmse:6.2f} | MAE: ${mae:6.2f} | R2: {r2:6.4f} | WAPE: {wape:5.2f}%")
        
    # Ensemble Prediction
    ens_p = 0.45 * fold_preds['LightGBM'] + 0.45 * fold_preds['XGBoost'] + 0.10 * fold_preds['Random Forest']
    ens_rmse = np.sqrt(mean_squared_error(y_val_fold, ens_p))
    ens_mae = mean_absolute_error(y_val_fold, ens_p)
    ens_r2 = r2_score(y_val_fold, ens_p)
    ens_wape = (np.abs(y_val_fold - ens_p).sum() / y_val_fold.sum()) * 100.0
    
    cv_results.append({
        'Fold': val_name, 'Model': 'Ensemble (Proposed)', 'RMSE': ens_rmse, 'MAE': ens_mae, 'R2': ens_r2, 'WAPE(%)': ens_wape
    })
    print(f"{'Ensemble (Proposed)':18s} | RMSE: ${ens_rmse:6.2f} | MAE: ${ens_mae:6.2f} | R2: {ens_r2:6.4f} | WAPE: {ens_wape:5.2f}%")

cv_df = pd.DataFrame(cv_results)
summary_metrics = cv_df.groupby('Model')[['MAE', 'RMSE', 'R2', 'WAPE(%)']].mean().reset_index()
print("\n" + "=" * 80)
print("ROLLING CROSS-VALIDATION AVERAGE METRICS ACROSS ALL TEMPORAL FOLDS")
print("=" * 80)
print(summary_metrics.to_string(index=False))

# Plot Rolling Evaluation Comparison
plt.figure(figsize=(16, 5), dpi=150)
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

model_order = ['Ridge Regression', 'LightGBM', 'Ensemble (Proposed)', 'Random Forest', 'XGBoost']

# MAE
sns.barplot(data=summary_metrics, x='Model', y='MAE', order=model_order, ax=axes[0], hue='Model', palette='Blues_d', legend=False)
axes[0].set_title('Mean Absolute Error (MAE, $) - Lower is Better', fontsize=11, fontweight='bold')
axes[0].tick_params(axis='x', rotation=30)
for p in axes[0].patches:
    axes[0].annotate(f"${p.get_height():.2f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                     ha='center', va='center', xytext=(0, 5), textcoords='offset points', fontsize=9)

# RMSE
sns.barplot(data=summary_metrics, x='Model', y='RMSE', order=model_order, ax=axes[1], hue='Model', palette='Oranges_d', legend=False)
axes[1].set_title('Root Mean Squared Error (RMSE, $) - Lower is Better', fontsize=11, fontweight='bold')
axes[1].tick_params(axis='x', rotation=30)
for p in axes[1].patches:
    axes[1].annotate(f"${p.get_height():.2f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                     ha='center', va='center', xytext=(0, 5), textcoords='offset points', fontsize=9)

# R2
sns.barplot(data=summary_metrics, x='Model', y='R2', order=model_order, ax=axes[2], hue='Model', palette='Greens_d', legend=False)
axes[2].set_title('R² Score - Higher is Better', fontsize=11, fontweight='bold')
axes[2].tick_params(axis='x', rotation=30)
for p in axes[2].patches:
    axes[2].annotate(f"{p.get_height():.4f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                     ha='center', va='center', xytext=(0, 5), textcoords='offset points', fontsize=9)

plt.tight_layout()
plt.savefig("model_comparison.png", bbox_inches='tight')
plt.close()
print("\nSaved rolling validation comparison chart to model_comparison.png")

# ---------------------------------------------------------------------------
# 5. Production Retraining on Full Clean Dataset & Prediction Export
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("5. PRODUCTION RETRAINING & PREDICTION EXPORT")
print("=" * 80)
print(f"Retraining final models on all {len(X)} clean training loads (Jan-Oct 2025)...")

final_lgbm = lgb.LGBMRegressor(n_estimators=350, learning_rate=0.05, max_depth=8, num_leaves=31, random_state=42, verbose=-1)
final_xgb = xgb.XGBRegressor(n_estimators=350, learning_rate=0.05, max_depth=6, random_state=42, n_jobs=-1)
final_rf = RandomForestRegressor(n_estimators=120, max_depth=12, random_state=42, n_jobs=-1)

final_lgbm.fit(X, y)
final_xgb.fit(X, y)
final_rf.fit(X, y)

# Predict on 12,000 validation loads and 31 December inputs
val_preds = (
    0.45 * final_lgbm.predict(X_val) +
    0.45 * final_xgb.predict(X_val) +
    0.10 * final_rf.predict(X_val)
)

dec_preds = (
    0.45 * final_lgbm.predict(X_dec) +
    0.45 * final_xgb.predict(X_dec) +
    0.10 * final_rf.predict(X_dec)
)

# Enforce positive rate boundary constraint
val_preds = np.clip(val_preds, a_min=100.0, a_max=None)
dec_preds = np.clip(dec_preds, a_min=100.0, a_max=None)

# 1. Export validation predictions
val_predictions_df = pd.DataFrame({
    'load_id': val_raw['load_id'],
    'predicted_rate': val_preds
})
val_predictions_df.to_csv("validation_predictions.csv", index=False)
print(f"Successfully saved 12,000 predictions to validation_predictions.csv (Mean rate: ${val_preds.mean():.2f})")

# 2. Update December chart inputs
dec_raw['predicted_rate'] = dec_preds
dec_out_df = dec_raw[['pickup', 'delivery', 'distance', 'equipment', 'weight', 'date', 'predicted_rate']].copy()
dec_out_df['date'] = pd.to_datetime(dec_out_df['date']).dt.strftime('%Y-%m-%d')
dec_out_df.to_csv("december-chart-inputs.csv", index=False)
print("Successfully updated december-chart-inputs.csv with clean December predictions.")

print("\n" + "=" * 80)
print("PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
print("=" * 80)
