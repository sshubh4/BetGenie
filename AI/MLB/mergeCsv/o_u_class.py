#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MLB Over/Under: XGBoost Poisson -> per-line Isotonic (train-only per fold) -> EV threshold

- Trains an XGBRegressor with objective='count:poisson' on TOTAL RUNS (R_home+R_away)
- Converts predicted mean runs (mû) into P(Over) via Poisson tail vs each game's total line
- Calibrates P(Over) per total-line bucket using Isotonic Regression
  *Fitted on the TRAIN SLICE of each fold*, then applied to that fold’s validation (leak-free)
- Searches EV threshold (with -110 odds) to maximize historical profit (using OOF, leak-free)
- Reports OOF metrics and saves model + global calibrators (fit on OOF) + threshold for deployment
"""

import os
import math
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import TimeSeriesSplit
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, brier_score_loss, f1_score
from xgboost import XGBRegressor

# ----------------------------- Helpers -----------------------------
def poisson_cdf(k: int, mu: float) -> float:
    """P(N <= k) for Poisson(mu)."""
    if mu <= 0:
        return 1.0 if k >= 0 else 0.0
    term = math.exp(-mu)   # n=0
    s = term
    for n in range(1, max(0, k) + 1):
        term *= mu / n
        s += term
    return min(1.0, max(0.0, s))

def poisson_sf(k: int, mu: float) -> float:
    """P(N > k) = 1 - P(N <= k)."""
    return max(0.0, 1.0 - poisson_cdf(k, mu))

def line_bucket(x: float) -> float:
    """Bucket total line to nearest 0.5 to increase data per calibrator."""
    if pd.isna(x):
        return np.nan
    return round(float(x) * 2.0) / 2.0

def pick_features(df: pd.DataFrame) -> list:
    """Automatic, leakage-aware feature selection from jordan_final.csv."""
    drop_exact = {
        # Labels / totals / ids / explicit drops
        'over_result','over_under_label','total_line',
        'Date_Parsed','date','game_id','home_team','away_team'
    }
    keep = []
    for c in df.columns:
        if c in drop_exact:
            continue
        if any(tok in c for tok in ['Boxscore','merge_key']):
            continue
        # numeric only
        if np.issubdtype(df[c].dtype, np.number):
            keep.append(c)
    # hard guard against known leak-prone names if they slipped through
    forbidden = {"R_home","R_away","over_result","over_under_label"}
    keep = [c for c in keep if c not in forbidden]
    return keep

def total_runs_from_df(df: pd.DataFrame) -> np.ndarray:
    r_home = pd.to_numeric(df['R_home'], errors='coerce')
    r_away = pd.to_numeric(df['R_away'], errors='coerce')
    return (r_home + r_away).values

def p_over_from_mu_and_line(mu: np.ndarray, line: np.ndarray) -> (np.ndarray, np.ndarray):
    """Vectorized P(Over) and P(Push) under Poisson(mu) vs total_line."""
    p_over = np.zeros_like(mu, dtype=float)
    p_push = np.zeros_like(mu, dtype=float)
    for i, (m, L) in enumerate(zip(mu, line)):
        if pd.isna(L) or not np.isfinite(m) or m <= 0:
            p_over[i] = np.nan
            p_push[i] = np.nan
            continue
        if float(L).is_integer():
            k = int(L)
            p_over[i] = poisson_sf(k, m)
            # P(N == k) via log gamma
            lg = -m + k * math.log(m) - math.lgamma(k + 1)
            p_push[i] = math.exp(lg)
        else:
            k = math.floor(L)
            p_over[i] = poisson_sf(k, m)
            p_push[i] = 0.0
    return p_over, p_push

def ev_for_sides(p_over: float, p_push: float) -> (float, float):
    """Return EV for Over and Under with -110 on both sides (profit +0.909 win, -1 loss)."""
    p_push = max(0.0, p_push if np.isfinite(p_push) else 0.0)
    p_under = max(0.0, 1.0 - p_over - p_push)
    ev_over  = p_over  * 0.909 - p_under * 1.0
    ev_under = p_under * 0.909 - p_over  * 1.0
    return ev_over, ev_under

# ----------------------------- Load data -----------------------------
data_path = "jordan_final.csv"
if not os.path.exists(data_path):
    raise FileNotFoundError("jordan_final.csv not found. Run your merge script first to produce it.")

df = pd.read_csv(data_path)

# Require these columns
required_cols = {'R_home','R_away','total_line','over_result'}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"Your dataset is missing required columns: {missing}")

# drop rows without needed fields
df = df.copy()
df['total_line'] = pd.to_numeric(df['total_line'], errors='coerce')
mask_ok = df['total_line'].notna() & pd.to_numeric(df['R_home'], errors='coerce').notna() & pd.to_numeric(df['R_away'], errors='coerce').notna()
df = df.loc[mask_ok].reset_index(drop=True)

# Build labels for evaluation (exclude pushes)
y_over = df['over_result'].copy()
if str(y_over.dtype) != 'Int64':
    y_over = pd.to_numeric(y_over, errors='coerce').astype('Int64')
y_eval_mask = y_over.notna().values
y_bin = y_over.fillna(-1).astype(int).values  # -1 for push; we'll mask later

# ----------------------------- Features & Target -----------------------------
# Sort by date if present to ensure temporal splits
if 'Date_Parsed' in df.columns:
    order = pd.to_datetime(df['Date_Parsed'], errors='coerce').fillna(pd.Timestamp('1900-01-01'))
    df = df.iloc[np.argsort(order.values)].reset_index(drop=True)

# Target for Poisson reg: total runs
y_total = total_runs_from_df(df)

# Features
feature_cols = pick_features(df)

# Final safety guard
forbidden = {"R_home","R_away","over_result","over_under_label","total_line"}
bad = forbidden & set(feature_cols)
assert not bad, f"Forbidden features present in features: {bad}"

X = df[feature_cols].astype(float).fillna(0.0).values
lines = df['total_line'].values
line_buckets = np.array([line_bucket(x) for x in lines])

# Rebuild labels/masks aligned to sorted df
y_over = df['over_result']
if str(y_over.dtype) != 'Int64':
    y_over = pd.to_numeric(y_over, errors='coerce').astype('Int64')
y_eval_mask = y_over.notna().values
y_bin = y_over.fillna(-1).astype(int).values

# ----------------------------- CV Train & (train-only) Calibrate -----------------------------
tscv = TimeSeriesSplit(n_splits=5)
oof_p_over_raw = np.full(len(df), np.nan)
oof_p_over_cal = np.full(len(df), np.nan)
oof_p_push     = np.full(len(df), np.nan)

fold_id = 1
for tr_idx, va_idx in tscv.split(X):
    Xtr, Xva = X[tr_idx], X[va_idx]
    ytr_total, yva_total = y_total[tr_idx], y_total[va_idx]
    lines_tr, lines_va = lines[tr_idx], lines[va_idx]
    buckets_tr, buckets_va = line_buckets[tr_idx], line_buckets[va_idx]

    # XGB Poisson model for total runs
    model = XGBRegressor(
        objective='count:poisson',
        tree_method='hist',
        n_estimators=800,
        max_depth=6,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=42
    )
    model.fit(Xtr, ytr_total, eval_set=[(Xva, yva_total)], verbose=False)

    # Predict on TRAIN (for fitting calibrators) and on VAL (for evaluation)
    mu_tr = np.clip(model.predict(Xtr), 1e-6, 50)
    p_tr_raw, _ = p_over_from_mu_and_line(mu_tr, lines_tr)

    mu_va = np.clip(model.predict(Xva), 1e-6, 50)
    p_va_raw, p_va_push = p_over_from_mu_and_line(mu_va, lines_va)

    # Save raw OOF for this fold
    oof_p_over_raw[va_idx] = p_va_raw
    oof_p_push[va_idx]     = p_va_push

    # ------ Fit per-bucket isotonic on TRAIN slice ONLY ------
    cal_by_bucket = {}
    tr_mask_eval = (y_over.iloc[tr_idx].notna().values) & np.isfinite(p_tr_raw)
    y_tr_over = (y_bin[tr_idx][tr_mask_eval] == 1).astype(int)
    p_tr_use = p_tr_raw[tr_mask_eval]
    b_tr = buckets_tr[tr_mask_eval]

    for b in np.unique(b_tr):
        m = (b_tr == b)
        if m.sum() >= 25:
            iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip')
            iso.fit(p_tr_use[m], y_tr_over[m])  # <-- fit on TRAIN only
            cal_by_bucket[b] = iso

    # Apply to validation predictions
    p_va_cal = p_va_raw.copy()
    for b, iso in cal_by_bucket.items():
        m = (buckets_va == b)
        if m.any():
            p_va_cal[m] = iso.predict(p_va_raw[m])

    oof_p_over_cal[va_idx] = p_va_cal

    print(f"Fold {fold_id}: train-fitted calibrators for {len(cal_by_bucket)} buckets")
    fold_id += 1

# ----------------------------- OOF Metrics & EV threshold -----------------------------
mask_eval = y_eval_mask & np.isfinite(oof_p_over_cal)
y_true = (y_bin[mask_eval] == 1).astype(int)
p_cal  = oof_p_over_cal[mask_eval]
p_push_eval = oof_p_push[mask_eval]

roc  = roc_auc_score(y_true, p_cal) if len(np.unique(y_true)) > 1 else np.nan
brier = brier_score_loss(y_true, p_cal)
f1_05 = f1_score(y_true, (p_cal >= 0.5).astype(int))

# EV sweep to pick deployment threshold
thr_grid = np.arange(0.0, 0.151, 0.005)
best_thr, best_profit, best_bets, best_winrate = 0.0, -1e9, 0, 0.0
for thr in thr_grid:
    profit = 0.0
    bets = 0
    wins = 0
    for i, prob in enumerate(p_cal):
        ev_over, ev_under = ev_for_sides(prob, p_push_eval[i] if np.isfinite(p_push_eval[i]) else 0.0)
        chosen = 1 if ev_over >= ev_under else 0
        ev = max(ev_over, ev_under)
        if ev > thr:
            bets += 1
            won = (y_true[i] == chosen)
            if won:
                profit += 0.909
                wins += 1
            else:
                profit -= 1.0
    if bets > 0 and profit > best_profit:
        best_profit = profit
        best_thr = float(thr)
        best_bets = bets
        best_winrate = wins / bets if bets else 0.0

print("\n======= OOF Performance (XGB Poisson -> train-only per-line Isotonic -> EV threshold) =======")
print(f"OOF ROC-AUC: {roc:.4f} | OOF Brier: {brier:.4f} | OOF F1@0.50: {f1_05:.4f}")
print(f"Best EV threshold: {best_thr:.3f} | Bets: {best_bets} | Win%: {best_winrate:.3f} | Total units: {best_profit:.2f}")

# ----------------------------- Fit final model & global calibrators -----------------------------
# Train final model on ALL data
final_model = XGBRegressor(
    objective='count:poisson',
    tree_method='hist',
    n_estimators=800,
    max_depth=6,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    random_state=42
)
final_model.fit(X, y_total, verbose=False)

# Build global per-bucket calibrators using OOF RAW (for deployment only)
global_cals = {}
all_mask = y_eval_mask & np.isfinite(oof_p_over_raw)
df_cal = pd.DataFrame({
    'bucket': line_buckets[all_mask],
    'p_raw': oof_p_over_raw[all_mask],
    'y_over': (y_bin[all_mask] == 1).astype(int)
})
for b, g in df_cal.groupby('bucket'):
    if len(g) >= 50:
        iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip')
        iso.fit(g['p_raw'].values, g['y_over'].values)
        global_cals[float(b)] = iso

# ----------------------------- Save artifacts -----------------------------
os.makedirs("models_xgb_poisson", exist_ok=True)

joblib.dump({
    'model': final_model,
    'feature_cols': feature_cols
}, "models_xgb_poisson/model.joblib")

joblib.dump({
    'global_calibrators': global_cals,
    'line_bucket_fn': 'round(x*2)/2',
    'ev_threshold': best_thr
}, "models_xgb_poisson/calibration.joblib")

with open("models_xgb_poisson/metrics_oof.json", "w") as f:
    json.dump({
        'oof_roc_auc': float(roc),
        'oof_brier': float(brier),
        'oof_f1_at_0.50': float(f1_05),
        'best_ev_threshold': best_thr,
        'best_total_units': float(best_profit),
        'best_bets': int(best_bets),
        'best_winrate': float(best_winrate)
    }, f, indent=2)

print("\nSaved:")
print(" - models_xgb_poisson/model.joblib")
print(" - models_xgb_poisson/calibration.joblib")
print(" - models_xgb_poisson/metrics_oof.json")
