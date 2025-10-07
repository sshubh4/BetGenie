#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MLB Over/Under: XGBoost Poisson (Optuna-tuned) -> per-line Isotonic -> price-aware EV sweep (leak-safe)

- Trains an XGBRegressor with objective='count:poisson' on total runs (R_home + R_away).
- Converts predicted mean runs (mu) into P(Over) via Poisson tail against the market total line.
- Calibrates probabilities per total-line bucket using isotonic regression with a global fallback.
- Sweeps EV thresholds using market prices (falls back to -110/-110 if missing) to pick a deployment threshold.
- Uses time-ordered CV and strictly pre-game features to avoid leakage.
- Saves model, calibrators, and OOF metrics.

Outputs:
  models_xgb_poisson/model.joblib
  models_xgb_poisson/calibration.joblib
  models_xgb_poisson/metrics_oof.json
"""

import os
import math
import json
import joblib
import numpy as np
import pandas as pd
import optuna
from typing import Tuple, Dict

from sklearn.model_selection import TimeSeriesSplit
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, brier_score_loss, f1_score
from xgboost import XGBRegressor

# Silence Optuna info logs
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ----------------------------- Core helpers -----------------------------
def poisson_cdf(k: int, mu: float) -> float:
    """P(N <= k) for Poisson(mu) computed via a stable iterative sum."""
    if mu <= 0:
        return 1.0 if k >= 0 else 0.0
    term = math.exp(-mu)   # n=0
    s = term
    up_to = max(0, k)
    for n in range(1, up_to + 1):
        term *= mu / n
        s += term
    return min(1.0, max(0.0, s))

def poisson_sf(k: int, mu: float) -> float:
    """P(N > k) = 1 - P(N <= k)."""
    return max(0.0, 1.0 - poisson_cdf(k, mu))

def line_bucket(x: float) -> float:
    """Bucket total line to nearest 0.5 (e.g., 8.1 -> 8.0, 8.26 -> 8.5)."""
    if pd.isna(x):
        return np.nan
    return round(float(x) * 2.0) / 2.0

def american_to_prob(odds):
    if odds is None or (isinstance(odds, float) and not np.isfinite(odds)):
        return np.nan
    o = float(odds)
    return (100.0 / (o + 100.0)) if o > 0 else ((-o) / (-o + 100.0))

def american_payout_per_unit(odds):
    """Return profit per 1u risked when winning a bet at 'odds' (e.g., -110 => +0.909)."""
    o = float(odds)
    return (o / 100.0) if o > 0 else (100.0 / (-o))

def p_over_from_mu_and_line(mu: np.ndarray, line: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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
            # P(N == k) in log-space for stability
            try:
                lg = -m + k * math.log(m) - math.lgamma(k + 1)
                p_push[i] = math.exp(lg)
            except ValueError:
                p_push[i] = 0.0
        else:
            k = math.floor(L)  # e.g., 7.5 => Over if N >= 8 -> sf(7, mu)
            p_over[i] = poisson_sf(k, m)
            p_push[i] = 0.0
    return p_over, p_push

def ev_for_sides(p_over: float, p_push: float, price_over: float | None, price_under: float | None) -> Tuple[float, float]:
    """
    Price-aware EV per 1 unit risked:
      EV_over  = p_over * payout(over)  - p_under * 1
      EV_under = p_under * payout(under) - p_over * 1
    where p_under = 1 - p_over - p_push and payout(-110) = 0.909...
    Missing prices default to -110 both sides.
    """
    if price_over is None or not np.isfinite(price_over):
        price_over = -110.0
    if price_under is None or not np.isfinite(price_under):
        price_under = -110.0

    p_over = min(max(p_over, 0.0), 1.0)
    p_push = 0.0 if (p_push is None or not np.isfinite(p_push)) else max(0.0, p_push)
    p_under = max(0.0, 1.0 - p_over - p_push)

    pay_over  = american_payout_per_unit(price_over)
    pay_under = american_payout_per_unit(price_under)

    ev_over  = p_over  * pay_over  - p_under * 1.0
    ev_under = p_under * pay_under - p_over  * 1.0
    return ev_over, ev_under

# ----------------------------- Data & features -----------------------------
DATA_PATH = "./start_auto/jordan_final.csv"
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError("jordan_final.csv not found. Make sure your merge produced this file.")

df = pd.read_csv(DATA_PATH)

required_cols = {'R_home', 'R_away', 'total_line', 'over_result'}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"Dataset missing required columns: {missing}")

# Filter to rows with essentials
df = df.copy()
df['total_line'] = pd.to_numeric(df['total_line'], errors='coerce')
mask_ok = df['total_line'].notna() & pd.to_numeric(df['R_home'], errors='coerce').notna() & pd.to_numeric(df['R_away'], errors='coerce').notna()
df = df.loc[mask_ok].reset_index(drop=True)

# Time order for leak safety
if 'Date_Parsed' in df.columns:
    order = pd.to_datetime(df['Date_Parsed'], errors='coerce').fillna(pd.Timestamp('1900-01-01'))
    df = df.iloc[np.argsort(order.values)].reset_index(drop=True)

# Labels for evaluation
y_over = df['over_result']
if str(y_over.dtype) != 'Int64':
    y_over = pd.to_numeric(y_over, errors='coerce').astype('Int64')
y_eval_mask = y_over.notna().values
y_bin = y_over.fillna(-1).astype(int).values  # -1 for pushes; will mask later

# Target for Poisson regression (total runs)
r_home = pd.to_numeric(df['R_home'], errors='coerce')
r_away = pd.to_numeric(df['R_away'], errors='coerce')
y_total = (r_home + r_away).values

# Price columns (optional)
price_over_full  = pd.to_numeric(df.get('price_over'), errors='coerce') if 'price_over' in df.columns else pd.Series(np.nan, index=df.index)
price_under_full = pd.to_numeric(df.get('price_under'), errors='coerce') if 'price_under' in df.columns else pd.Series(np.nan, index=df.index)

def pick_features(d: pd.DataFrame) -> list:
    """
    Leak-aware: keep numeric pre-game features only.
    Drop explicit labels / targets, and non-informative identifiers.
    DO NOT blanket-drop '_home'/'_away' (these are pre-game signals you engineered).
    """
    drop_exact = {'over_result', 'over_under_label', 'R_home', 'R_away'}
    drop_startswith = ('Boxscore', 'merge_key')
    drop_contains = ('_label', 'actual', 'result')

    keep = []
    for c in d.columns:
        if c in drop_exact:
            continue
        if any(c.startswith(p) for p in drop_startswith):
            continue
        if any(tok in c for tok in drop_contains):
            continue
        # numeric only
        if np.issubdtype(d[c].dtype, np.number):
            keep.append(c)
    # Ensure total_line is included as a feature for μ learning context
    if 'total_line' not in keep and 'total_line' in d.columns and np.issubdtype(d['total_line'].dtype, np.number):
        keep.append('total_line')
    return keep

feature_cols = pick_features(df)
X_all = df[feature_cols].astype(float).fillna(0.0).values
lines_all = df['total_line'].values
line_buckets_all = np.array([line_bucket(x) for x in lines_all])

# ----------------------------- Optuna: tune μ model -----------------------------
def tune_mu_params(X, y, lines, y_over_series) -> dict:
    """Tune XGB Poisson hyperparameters using time-ordered CV maximizing ROC AUC of raw P(Over)."""
    tscv = TimeSeriesSplit(n_splits=5)

    def objective(trial):
        params = {
            'objective': 'count:poisson',
            'tree_method': 'hist',
            'n_estimators': trial.suggest_int('n_estimators', 500, 1400),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.07, log=True),
            'subsample': trial.suggest_float('subsample', 0.7, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 5.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 5.0, log=True),
            'random_state': 42
        }

        aucs = []
        for tr_idx, va_idx in tscv.split(X):
            Xtr, Xva = X[tr_idx], X[va_idx]
            ytr = y[tr_idx]
            lines_va = lines[va_idx]

            y_over_va = y_over_series.iloc[va_idx]
            mask_va = y_over_va.notna().values
            y_bin_va = y_over_va.fillna(-1).astype(int).values

            model = XGBRegressor(**params)
            model.fit(Xtr, ytr, verbose=False)
            mu_va = np.clip(model.predict(Xva), 1e-6, 50)
            p_over_raw, _ = p_over_from_mu_and_line(mu_va, lines_va)

            m = mask_va & np.isfinite(p_over_raw)
            y_true = (y_bin_va[m] == 1).astype(int)
            if y_true.size > 0 and np.unique(y_true).size > 1:
                aucs.append(roc_auc_score(y_true, p_over_raw[m]))

        return float(np.mean(aucs)) if aucs else 0.0

    print("🔎 Optuna tuning μ (mean) ...")
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50)  # Increase for deeper search
    best_params = study.best_params
    print(f"✅ Best μ params: {best_params}")
    return best_params

best_mu_params = tune_mu_params(X_all, y_total, lines_all, y_over)

# ----------------------------- CV with best μ params + isotonic -----------------------------
tscv = TimeSeriesSplit(n_splits=5)
oof_p_over_raw = np.full(len(df), np.nan)
oof_p_push = np.full(len(df), np.nan)
oof_p_over_cal = np.full(len(df), np.nan)

bucket_min_n = 25  # train isotonic only if bucket has enough samples
fold_id = 1

for tr_idx, va_idx in tscv.split(X_all):
    Xtr, Xva = X_all[tr_idx], X_all[va_idx]
    ytr, yva = y_total[tr_idx], y_total[va_idx]
    lines_va = lines_all[va_idx]
    buckets_va = line_buckets_all[va_idx]

    model = XGBRegressor(
        objective='count:poisson',
        tree_method='hist',
        random_state=42,
        **best_mu_params
    )
    model.fit(Xtr, ytr, verbose=False)

    mu_va = np.clip(model.predict(Xva), 1e-6, 50)
    p_over_raw_val, p_push_val = p_over_from_mu_and_line(mu_va, lines_va)

    # Save raw predictions (OOF arrays)
    oof_p_over_raw[va_idx] = p_over_raw_val
    oof_p_push[va_idx] = p_push_val

    # Build per-bucket isotonic on THIS fold's validation only (no leakage)
    y_over_va = y_over.iloc[va_idx]
    val_mask_eval = y_over_va.notna().values & np.isfinite(p_over_raw_val)
    yva_over = (y_over_va.fillna(-1).astype(int).values[val_mask_eval] == 1).astype(int)
    pva_raw = p_over_raw_val[val_mask_eval]
    bva = buckets_va[val_mask_eval]

    cals_this_fold: Dict[float, IsotonicRegression] = {}
    for b in np.unique(bva):
        idx_b = (bva == b)
        if idx_b.sum() >= bucket_min_n:
            iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip')
            iso.fit(pva_raw[idx_b], yva_over[idx_b])
            cals_this_fold[float(b)] = iso

    # Fallback: global iso on this fold’s validation (if enough samples)
    iso_fallback = None
    if len(pva_raw) >= max(50, bucket_min_n):
        iso_fallback = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip')
        iso_fallback.fit(pva_raw, yva_over)

    # ---- APPLY CALIBRATION STRICTLY ON THE VALIDATION SLICE ----
    # work on a buffer that is exactly the size of the val slice
    p_cal_val = p_over_raw_val.copy()

    # apply per-bucket calibrators
    for b, iso in cals_this_fold.items():
        idx_b_full_val = (buckets_va == b)        # boolean mask within val slice
        if idx_b_full_val.any():
            p_cal_val[idx_b_full_val] = iso.predict(p_over_raw_val[idx_b_full_val])

    # fallback for any remaining NaNs/inf in the val slice only
    if iso_fallback is not None:
        idx_missing_val = ~np.isfinite(p_cal_val)
        if idx_missing_val.any():
            p_cal_val[idx_missing_val] = iso_fallback.predict(p_over_raw_val[idx_missing_val])

    # write the calibrated slice back into the OOF array
    oof_p_over_cal[va_idx] = p_cal_val

    print(f"Fold {fold_id}: {len(cals_this_fold)} bucket calibrators + {'fallback' if iso_fallback is not None else 'no fallback'}")
    fold_id += 1

# ----------------------------- OOF metrics & EV sweep (aligned, price-aware) -----------------------------
mask_eval = y_eval_mask & np.isfinite(oof_p_over_cal)
p_cal = oof_p_over_cal[mask_eval]
p_push_eval = oof_p_push[mask_eval]
p_push_eval = np.where(np.isfinite(p_push_eval), p_push_eval, 0.0)
y_true = (y_bin[mask_eval] == 1).astype(int)

# align prices with mask
price_over_m = price_over_full.values[mask_eval]
price_under_m = price_under_full.values[mask_eval]

roc = roc_auc_score(y_true, p_cal) if np.unique(y_true).size > 1 else float('nan')
brier = brier_score_loss(y_true, p_cal)
f1_05 = f1_score(y_true, (p_cal >= 0.5).astype(int))
print(f"\nOOF base metrics -> ROC-AUC: {roc:.4f} | Brier: {brier:.4f} | F1@0.50: {f1_05:.4f}")

EV_GRID = np.arange(0.00, 0.151, 0.005)
best_thr, best_profit, best_bets, best_winrate = 0.0, -1e9, 0, 0.0

for thr in EV_GRID:
    profit = 0.0
    bets = 0
    wins = 0
    for i in range(len(p_cal)):
        ev_o, ev_u = ev_for_sides(p_cal[i], p_push_eval[i], price_over_m[i], price_under_m[i])
        chosen = 1 if ev_o >= ev_u else 0
        ev = ev_o if chosen == 1 else ev_u
        if ev > thr:
            bets += 1
            if y_true[i] == chosen:
                payout = american_payout_per_unit(price_over_m[i]) if chosen == 1 else american_payout_per_unit(price_under_m[i])
                profit += payout
                wins += 1
            else:
                profit -= 1.0
    if bets > 0 and profit > best_profit:
        best_thr = float(thr)
        best_profit = profit
        best_bets = bets
        best_winrate = wins / bets

print("\n======= OOF Performance (Tuned XGB Poisson -> per-line Isotonic -> EV threshold, price-aware) =======")
print(f"OOF ROC-AUC: {roc:.4f} | OOF Brier: {brier:.4f} | F1@0.50: {f1_05:.4f}")
print(f"Best EV threshold: {best_thr:.3f} | Bets: {best_bets} | Win%: {best_winrate:.3f} | Units: {best_profit:.2f}")

# ----------------------------- Final model & global calibrators -----------------------------
print("\nTraining final model on all data with best μ params...")
final_model = XGBRegressor(
    objective='count:poisson',
    tree_method='hist',
    random_state=42,
    **best_mu_params
)
final_model.fit(X_all, y_total, verbose=False)

# Global per-bucket calibrators from OOF raw
global_cals: Dict[float, IsotonicRegression] = {}
all_mask = y_eval_mask & np.isfinite(oof_p_over_raw)
df_cal = pd.DataFrame({
    'bucket': line_buckets_all[all_mask],
    'p_raw': oof_p_over_raw[all_mask],
    'y_over': (y_bin[all_mask] == 1).astype(int)
})
for b, g in df_cal.groupby('bucket'):
    if len(g) >= 50:
        iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip')
        iso.fit(g['p_raw'].values, g['y_over'].values)
        global_cals[float(b)] = iso

# Fallback global iso across all OOF if no bucket iso at inference
global_fallback_iso = None
if len(df_cal) >= 100:
    global_fallback_iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip')
    global_fallback_iso.fit(df_cal['p_raw'].values, df_cal['y_over'].values)

# ----------------------------- Save artifacts -----------------------------
os.makedirs("models_xgb_poisson", exist_ok=True)

joblib.dump({
    'model': final_model,
    'feature_cols': feature_cols
}, "models_xgb_poisson/model.joblib")

joblib.dump({
    'global_calibrators': global_cals,
    'global_fallback': global_fallback_iso,
    'line_bucket_fn': 'round(x*2)/2',
    'ev_threshold': best_thr,
    'best_mu_params': best_mu_params
}, "models_xgb_poisson/calibration.joblib")

with open("models_xgb_poisson/metrics_oof.json", "w") as f:
    json.dump({
        'oof_roc_auc': float(roc),
        'oof_brier': float(brier),
        'oof_f1_at_0_50': float(f1_05),
        'best_ev_threshold': best_thr,
        'best_total_units': float(best_profit),
        'best_bets': int(best_bets),
        'best_winrate': float(best_winrate),
        'best_mu_params': best_mu_params,
        'n_features': len(feature_cols)
    }, f, indent=2)

print("\nSaved:")
print(" - models_xgb_poisson/model.joblib")
print(" - models_xgb_poisson/calibration.joblib")
print(" - models_xgb_poisson/metrics_oof.json")
