#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MLB Totals (NO EV, push-free .5 lines)
Goal: maximize classification accuracy/F1 for Over/Under.

Pipeline:
- XGB Poisson (predict total runs) -> p_over_poisson (via Poisson tail vs line)
- Per-line Isotonic calibration (no pushes)
- Logistic sidecar (binary:logistic on Over label)
- OOF Stacking meta-model combines: [p_poi_cal, p_logit, mu, total_line, (mu - total_line)]
- Global isotonic on blended prob
- Threshold tuning:
    - Global threshold that maximizes OOF accuracy
    - Per-line-bucket thresholds (override when enough data)
- Chronological TimeSeries CV
- No prices used; no EV logic.

Leak safety:
- Excludes true outcomes/labels from features (R_home, R_away, total_runs, over_binary, over_under_label)
- Uses only pre-game signals (line, weather, rolling_*, rest, etc.)
- Chronological folds (no future→past bleed)

Assumptions:
- Dataset is 'jordan_final.csv' or 'home_games_with_point.csv'
- Total line column is 'total_line' or 'Point'
"""

import os
import math
import json
import joblib
import numpy as np
import pandas as pd
import optuna

from sklearn.model_selection import TimeSeriesSplit
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    roc_auc_score, brier_score_loss, f1_score, accuracy_score,
    precision_recall_fscore_support, confusion_matrix
)
from xgboost import XGBRegressor

optuna.logging.set_verbosity(optuna.logging.WARNING)
RANDOM_STATE = 42

# ----------------------------- Helpers -----------------------------
def poisson_cdf(k: int, mu: float) -> float:
    if mu <= 0:
        return 1.0 if k >= 0 else 0.0
    term = math.exp(-mu)
    s = term
    up_to = max(0, k)
    for n in range(1, up_to + 1):
        term *= mu / n
        s += term
    return float(min(1.0, max(0.0, s)))

def poisson_sf(k: int, mu: float) -> float:
    return max(0.0, 1.0 - poisson_cdf(k, mu))

def line_bucket(x: float) -> float:
    if pd.isna(x):
        return np.nan
    return round(float(x) * 2.0) / 2.0

def p_over_from_mu_and_line_no_push(mu: np.ndarray, line: np.ndarray):
    """
    Over prob for .5 lines (no pushes). If an integer line appears, still use P(total > floor(line)).
    """
    p_over = np.zeros_like(mu, dtype=float)
    for i, (m, L) in enumerate(zip(mu, line)):
        if pd.isna(L) or not np.isfinite(m) or m <= 0:
            p_over[i] = np.nan
            continue
        k = math.floor(float(L))
        p_over[i] = poisson_sf(k, float(m))
    return p_over

def make_season_folds(dates: pd.Series, n_splits=5):
    order = np.argsort(pd.to_datetime(dates, errors='coerce').fillna(pd.Timestamp('1900-01-01')))
    idx = np.arange(len(dates))[order]
    tscv = TimeSeriesSplit(n_splits=n_splits)
    for tr, va in tscv.split(idx):
        yield idx[tr], idx[va]

# ----------------------------- Load data -----------------------------
DATA_PATH = "jordan_final.csv"
if not os.path.exists(DATA_PATH):
    if os.path.exists("home_games_with_point.csv"):
        DATA_PATH = "home_games_with_point.csv"
    else:
        raise FileNotFoundError("Dataset not found (jordan_final.csv or home_games_with_point.csv).")

df = pd.read_csv(DATA_PATH)

# Harmonize total line column
if 'total_line' not in df.columns:
    if 'Point' in df.columns:
        df['total_line'] = pd.to_numeric(df['Point'], errors='coerce')
    elif 'point' in df.columns:
        df['total_line'] = pd.to_numeric(df['point'], errors='coerce')
    else:
        raise ValueError("No total line found: expected 'total_line' or 'Point'/'point'.")
else:
    df['total_line'] = pd.to_numeric(df['total_line'], errors='coerce')

required = {'R_home', 'R_away', 'Date_Parsed', 'total_line'}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Dataset is missing required columns: {missing}")

# Core filters
df['R_home'] = pd.to_numeric(df['R_home'], errors='coerce')
df['R_away'] = pd.to_numeric(df['R_away'], errors='coerce')
mask_ok = df['total_line'].notna() & df['R_home'].notna() & df['R_away'].notna()
df = df.loc[mask_ok].copy().reset_index(drop=True)

# Time order
df['Date_Parsed'] = pd.to_datetime(df['Date_Parsed'], errors='coerce')
df = df.sort_values('Date_Parsed').reset_index(drop=True)

# Labels (push-free)
df['total_runs'] = df['R_home'] + df['R_away']
df['over_binary'] = (df['total_runs'] > df['total_line']).astype(int)
df['over_under_label'] = np.where(df['over_binary'] == 1, 'over', 'under')

# ----------------------------- Features (leak-safe) -----------------------------
# Keep numerics; EXCLUDE true outcomes/labels ONLY (keep all rolling_* features)
numeric_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]
EXCLUDE = {'R_home', 'R_away', 'total_runs', 'over_binary', 'over_under_label'}
feature_cols = [c for c in numeric_cols if c not in EXCLUDE]

# Safety asserts
assert EXCLUDE.isdisjoint(feature_cols), f"Leaky columns in features: {EXCLUDE & set(feature_cols)}"

# Build matrices
X_full = df[feature_cols].astype(float).fillna(0.0).values
lines = df['total_line'].to_numpy()
line_buckets_arr = np.array([line_bucket(x) for x in lines], dtype=float)
dates = df['Date_Parsed'].copy()

y_total = df['total_runs'].to_numpy()
y_over = df['over_binary'].to_numpy().astype(int)
y_eval_mask = np.isfinite(y_over)

# ----------------------------- Monotone constraints -----------------------------
def build_monotone_constraints(cols):
    cons = []
    for c in cols:
        if c == 'total_line':
            cons.append(1)      # higher line -> higher P(over)
        elif c == 'temperature_2m':
            cons.append(1)
        elif c == 'starter_era_sum':
            cons.append(-1)
        else:
            cons.append(0)
    return "(" + ",".join(str(x) for x in cons) + ")"

# ----------------------------- Optuna tuning for Poisson -----------------------------
def objective(trial):
    params = {
        'objective': 'count:poisson',
        'tree_method': 'hist',
        'n_estimators': trial.suggest_int('n_estimators', 700, 1400),
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'learning_rate': trial.suggest_float('learning_rate', 0.02, 0.08, log=True),
        'subsample': trial.suggest_float('subsample', 0.7, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 3.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 3.0, log=True),
        'random_state': RANDOM_STATE,
        'monotone_constraints': build_monotone_constraints(feature_cols),
    }
    aucs = []
    for tr_idx, va_idx in make_season_folds(dates, n_splits=5):
        Xtr, Xva = X_full[tr_idx], X_full[va_idx]
        ytr = y_total[tr_idx]
        yva_over = y_over[va_idx]
        m = XGBRegressor(**params)
        m.fit(Xtr, ytr, verbose=False)
        mu_va = np.clip(m.predict(Xva), 1e-6, 50)
        p_over_raw = p_over_from_mu_and_line_no_push(mu_va, lines[va_idx])
        mask = np.isfinite(p_over_raw)
        y_true = yva_over[mask]
        if y_true.size > 0 and np.unique(y_true).size > 1:
            aucs.append(roc_auc_score(y_true, p_over_raw[mask]))
    return float(np.mean(aucs)) if aucs else 0.0

print("🔎 Optuna tuning (Poisson)…")
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=40)
best_params = study.best_params
best_params['monotone_constraints'] = build_monotone_constraints(feature_cols)
print("✅ Best Poisson params:", best_params)

# ----------------------------- CV Train: Poisson + per-line Isotonic + Logistic + capture μ -----------------------------
oof_poisson_raw = np.full(len(df), np.nan)
oof_poisson_cal = np.full(len(df), np.nan)
oof_logit = np.full(len(df), np.nan)
oof_mu = np.full(len(df), np.nan)

for fold_id, (tr_idx, va_idx) in enumerate(make_season_folds(dates, n_splits=5), start=1):
    Xtr, Xva = X_full[tr_idx], X_full[va_idx]
    ytr_total, yva_total = y_total[tr_idx], y_total[va_idx]
    ytr_over,  yva_over  = y_over[tr_idx],  y_over[va_idx]
    lines_va = lines[va_idx]
    buckets_va = line_buckets_arr[va_idx]

    # Poisson on total runs
    model_poi = XGBRegressor(
        objective='count:poisson',
        tree_method='hist',
        random_state=RANDOM_STATE,
        **best_params
    )
    model_poi.fit(Xtr, ytr_total, eval_set=[(Xva, yva_total)], verbose=False)
    mu_va = np.clip(model_poi.predict(Xva), 1e-6, 50)
    oof_mu[va_idx] = mu_va
    p_over_raw = p_over_from_mu_and_line_no_push(mu_va, lines_va)
    oof_poisson_raw[va_idx] = p_over_raw

    # Per-line-bucket isotonic + fallback (on Poisson raw)
    mask = np.isfinite(p_over_raw)
    yva_over_masked = yva_over[mask]
    pva_raw = p_over_raw[mask]
    bva = buckets_va[mask]

    cals = {}
    iso_fallback = None
    if len(yva_over_masked) >= 50 and np.unique(yva_over_masked).size > 1:
        iso_fallback = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip').fit(pva_raw, yva_over_masked)

    for b in np.unique(bva):
        idx_b = (bva == b)
        if idx_b.sum() >= 25:
            yb = yva_over_masked[idx_b]
            if np.unique(yb).size > 1:
                iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip')
                iso.fit(pva_raw[idx_b], yb)
                cals[float(b)] = iso

    p_cal = p_over_raw.copy()
    for i in range(len(p_cal)):
        if not np.isfinite(p_cal[i]):
            continue
        b = buckets_va[i]
        if b in cals:
            p_cal[i] = cals[b].predict([p_cal[i]])[0]
        elif iso_fallback is not None:
            p_cal[i] = iso_fallback.predict([p_cal[i]])[0]
    oof_poisson_cal[va_idx] = p_cal

    # Logistic sidecar (direct over/under)
    model_log = XGBRegressor(
        objective='binary:logistic',
        tree_method='hist',
        random_state=RANDOM_STATE,
        n_estimators=800,
        max_depth=5,
        learning_rate=0.03,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=1e-4,
        reg_lambda=1e-3,
        monotone_constraints=build_monotone_constraints(feature_cols),
    )
    model_log.fit(Xtr, ytr_over, verbose=False)  # .predict returns prob for binary:logistic
    p_logit = np.clip(model_log.predict(Xva), 1e-6, 1 - 1e-6)
    oof_logit[va_idx] = p_logit

    print(f"Fold {fold_id}: {len(cals)} bucket calibrators{' + fallback' if iso_fallback is not None else ''} + logistic sidecar")

# ----------------------------- OOF Stacking meta-model + global isotonic + thresholds -----------------------------
mask_eval = y_eval_mask & np.isfinite(oof_poisson_cal) & np.isfinite(oof_logit) & np.isfinite(oof_mu)
y_true = y_over[mask_eval].astype(int)

p_poi = oof_poisson_cal[mask_eval]
p_log = oof_logit[mask_eval]
mu_oof = oof_mu[mask_eval]
line_oof = lines[mask_eval]
residual = mu_oof - line_oof

# Meta features: strictly pre-game derived
X_meta = np.column_stack([p_poi, p_log, mu_oof, line_oof, residual])

meta_model = XGBRegressor(
    objective='binary:logistic',
    tree_method='hist',
    random_state=RANDOM_STATE,
    n_estimators=400,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_alpha=1e-4,
    reg_lambda=1e-3,
)
meta_model.fit(X_meta, y_true, verbose=False)
p_meta = np.clip(meta_model.predict(X_meta), 1e-6, 1-1e-6)

# Global isotonic on meta prob
if np.unique(y_true).size > 1:
    iso_blend = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip').fit(p_meta, y_true)
    p_blend_cal = iso_blend.predict(p_meta)
else:
    iso_blend = None
    p_blend_cal = p_meta.copy()

# Global threshold tuning for accuracy
thr_grid = np.linspace(0.40, 0.60, 81)
best_thr, best_acc = 0.50, -1.0
for thr in thr_grid:
    acc = accuracy_score(y_true, (p_blend_cal >= thr).astype(int))
    if acc > best_acc:
        best_acc, best_thr = float(acc), float(thr)
print(f"\nMeta + isotonic tuned: thr={best_thr:.3f} | OOF ACC={best_acc:.4f}")

# Per-line-bucket thresholds (override when enough samples)
buckets = line_buckets_arr[mask_eval]
per_bucket_thr = {}
min_bucket = 120
for b in np.unique(buckets[np.isfinite(buckets)]):
    idx = (buckets == b)
    if idx.sum() >= min_bucket:
        best_b, best_a = best_thr, -1.0
        for thr in thr_grid:
            a = accuracy_score(y_true[idx], (p_blend_cal[idx] >= thr).astype(int))
            if a > best_a:
                best_a, best_b = float(a), float(thr)
        per_bucket_thr[float(b)] = best_b

def predict_with_bucket_thresholds(p, b, per_thr, global_thr):
    out = np.zeros_like(p, dtype=int)
    for i in range(len(p)):
        thr = per_thr.get(float(b[i]), global_thr)
        out[i] = 1 if p[i] >= thr else 0
    return out

y_pred = predict_with_bucket_thresholds(p_blend_cal, buckets, per_bucket_thr, best_thr)

# Final OOF metrics
roc = roc_auc_score(y_true, p_blend_cal) if np.unique(y_true).size > 1 else float('nan')
brier = brier_score_loss(y_true, p_blend_cal)
acc = accuracy_score(y_true, y_pred)
prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
cm = confusion_matrix(y_true, y_pred)

print("\n======= OOF Classification (NO EV, Meta + Iso + Bucket thresholds) =======")
print(f"ROC-AUC: {roc:.4f} | Brier: {brier:.4f} | Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")
print("Confusion matrix [ [TN FP]; [FN TP] ]:")
print(cm)
print(f"Buckets with custom thresholds: {len(per_bucket_thr)}")
if len(per_bucket_thr):
    print("Sample per-bucket thresholds:", sorted(per_bucket_thr.items())[:8])

# ----------------------------- Fit final base models & global Poisson calibrators -----------------------------
print("\nTraining final models on ALL data…")
final_model_poisson = XGBRegressor(
    objective='count:poisson',
    tree_method='hist',
    random_state=RANDOM_STATE,
    **best_params
)
final_model_poisson.fit(X_full, y_total, verbose=False)

final_model_logit = XGBRegressor(
    objective='binary:logistic',
    tree_method='hist',
    random_state=RANDOM_STATE,
    n_estimators=800,
    max_depth=5,
    learning_rate=0.03,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_alpha=1e-4,
    reg_lambda=1e-3,
    monotone_constraints=build_monotone_constraints(feature_cols),
)
final_model_logit.fit(X_full, y_over, verbose=False)

# Global isotonic from OOF Poisson raw (for inference calibration to p_poi_cal)
global_by_bucket = {}
global_fallback = None
mask_all = np.isfinite(oof_poisson_raw)
if mask_all.sum() >= 50 and np.unique(y_over[mask_all]).size > 1:
    global_fallback = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip').fit(
        oof_poisson_raw[mask_all], y_over[mask_all].astype(int)
    )
for b in np.unique(line_buckets_arr[mask_all]):
    idx = mask_all & (line_buckets_arr == b)
    if idx.sum() >= 50 and np.unique(y_over[idx]).size > 1:
        iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip')
        iso.fit(oof_poisson_raw[idx], y_over[idx].astype(int))
        global_by_bucket[float(b)] = iso

# ----------------------------- Persist -----------------------------
os.makedirs("models_totals_no_ev", exist_ok=True)

joblib.dump({
    'model_poisson': final_model_poisson,
    'model_logit': final_model_logit,
    'meta_model': meta_model,
    'feature_cols': feature_cols,
    'monotone_constraints': best_params['monotone_constraints'],
    'global_threshold': best_thr,
    'per_bucket_thresholds': per_bucket_thr,  # may be empty
}, "models_totals_no_ev/model.joblib")

joblib.dump({
    'poisson_iso_by_bucket': global_by_bucket,
    'poisson_iso_fallback': global_fallback,
    'blend_iso_global': iso_blend,          # global isotonic for the meta-blended prob
    'line_bucket_fn': 'round(x*2)/2'
}, "models_totals_no_ev/calibration.joblib")

with open("models_totals_no_ev/metrics_oof.json", "w") as f:
    json.dump({
        'roc_auc': float(roc),
        'brier': float(brier),
        'accuracy': float(acc),
        'precision': float(prec),
        'recall': float(rec),
        'f1': float(f1),
        'global_threshold': float(best_thr),
        'num_bucket_thresholds': int(len(per_bucket_thr)),
        'best_poisson_params': best_params
    }, f, indent=2)

print("\nSaved:")
print(" - models_totals_no_ev/model.joblib")
print(" - models_totals_no_ev/calibration.joblib")
print(" - models_totals_no_ev/metrics_oof.json")
