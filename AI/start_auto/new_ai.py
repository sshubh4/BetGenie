#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MLB Over/Under (price-aware, NO PUSHES):
XGB Poisson (Optuna-tuned) -> per-line Isotonic (with global fallback) -> EV threshold
+ season-aware rolling CV
+ monotone constraints on interpretable features
+ labels derived from final scores vs line (no pre-made label)
Assumptions:
- Totals are .5 (no pushes). If an integer appears, we still use P(total > floor(line)).
- R_home, R_away (final scores) are NEVER used as features.
- If prices are missing, EV falls back to -110/-110 (synthetic).
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
from sklearn.metrics import roc_auc_score, brier_score_loss, f1_score
from xgboost import XGBRegressor

optuna.logging.set_verbosity(optuna.logging.WARNING)

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
    Over prob for .5 lines (no pushes). If an integer line appears, we still
    compute P(total > line) = 1 - CDF(floor(line)).
    """
    p_over = np.zeros_like(mu, dtype=float)
    for i, (m, L) in enumerate(zip(mu, line)):
        if pd.isna(L) or not np.isfinite(m) or m <= 0:
            p_over[i] = np.nan
            continue
        k = math.floor(float(L))
        p_over[i] = poisson_sf(k, float(m))
    return p_over

def american_to_multiplier(odds):
    if not np.isfinite(odds):
        odds = -110.0
    odds = float(odds)
    return (100.0 / abs(odds)) if odds < 0 else (odds / 100.0)

def make_season_folds(dates: pd.Series, n_splits=5):
    """Rolling chronological folds based on dates."""
    order = np.argsort(pd.to_datetime(dates, errors='coerce').fillna(pd.Timestamp('1900-01-01')))
    idx = np.arange(len(dates))[order]
    tscv = TimeSeriesSplit(n_splits=n_splits)
    for tr, va in tscv.split(idx):
        yield idx[tr], idx[va]

# ----------------------------- Load data -----------------------------
DATA_PATH = "home_games_with_point.csv"
if not os.path.exists(DATA_PATH):
    # also allow your alternate filename:
    if os.path.exists("home_games_with_point.csv"):
        DATA_PATH = "home_games_with_point.csv"
    else:
        raise FileNotFoundError("Dataset not found (jordan_final.csv or home_games_with_point.csv).")

df = pd.read_csv(DATA_PATH)

# Harmonize line column: support 'Point' (books) or 'total_line' (internal)
if 'total_line' not in df.columns:
    if 'Point' in df.columns:
        df['total_line'] = pd.to_numeric(df['Point'], errors='coerce')
    elif 'point' in df.columns:
        df['total_line'] = pd.to_numeric(df['point'], errors='coerce')
    else:
        raise ValueError("No total line found: expected 'total_line' or 'Point'/'point' in the dataset.")
else:
    df['total_line'] = pd.to_numeric(df['total_line'], errors='coerce')

required_raw = {'R_home', 'R_away', 'Date_Parsed', 'total_line'}
missing = required_raw - set(df.columns)
if missing:
    raise ValueError(f"Dataset is missing required columns: {missing}")

# Core filters
df = df.copy()
df['R_home'] = pd.to_numeric(df['R_home'], errors='coerce')
df['R_away'] = pd.to_numeric(df['R_away'], errors='coerce')
mask_ok = df['total_line'].notna() & df['R_home'].notna() & df['R_away'].notna()
df = df.loc[mask_ok].reset_index(drop=True)

# time order (also used in folds)
df['Date_Parsed'] = pd.to_datetime(df['Date_Parsed'], errors='coerce')
df = df.sort_values('Date_Parsed').reset_index(drop=True)

# --- Derive labels from final score vs line (NO pushes) ---
df['total_runs'] = df['R_home'] + df['R_away']
df['over_binary'] = (df['total_runs'] > df['total_line']).astype(int)
df['over_under_label'] = np.where(df['over_binary'] == 1, 'over', 'under')

# Ensure price columns exist (may be NaN, fallback -110 is used for EV)
for c in ['price_over', 'price_under']:
    if c not in df.columns:
        df[c] = np.nan
    df[c] = pd.to_numeric(df[c], errors='coerce')

# ----------------------------- Features -----------------------------
# Keep numerics; exclude only true outcomes/labels (keep all rolling_* features)
numeric_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]
exclusions = {'R_home', 'R_away', 'total_runs', 'over_binary', 'over_under_label'}
feature_cols = [c for c in numeric_cols if c not in exclusions]

X_full = df[feature_cols].astype(float).fillna(0.0).values
lines = df['total_line'].to_numpy()
line_buckets_arr = np.array([line_bucket(x) for x in lines], dtype=float)
dates = df['Date_Parsed'].copy()
seasons_arr = dates.dt.year.to_numpy()

# Labels for evaluation
y_total = df['total_runs'].to_numpy()
y_over_bin = df['over_binary'].to_numpy()
y_eval_mask = np.isfinite(y_over_bin)

# ----------------------------- Monotone constraints -----------------------------
def build_monotone_constraints(cols):
    cons = []
    for c in cols:
        if c == 'total_line':
            cons.append(1)      # higher line -> higher P(over)
        elif c == 'temperature_2m':
            cons.append(1)      # warmer -> higher scoring tendency
        elif c == 'starter_era_sum':
            cons.append(-1)     # better (lower) ERA -> lower scoring
        else:
            cons.append(0)
    return "(" + ",".join(str(x) for x in cons) + ")"

# ----------------------------- Optuna tuning (Poisson) -----------------------------
def objective(trial):
    params = {
        'objective': 'count:poisson',
        'tree_method': 'hist',
        'n_estimators': trial.suggest_int('n_estimators', 600, 1400),
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'learning_rate': trial.suggest_float('learning_rate', 0.015, 0.08, log=True),
        'subsample': trial.suggest_float('subsample', 0.7, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 3.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 3.0, log=True),
        'random_state': 42,
        'monotone_constraints': build_monotone_constraints(feature_cols),
    }

    aucs = []
    for tr_idx, va_idx in make_season_folds(dates, n_splits=5):
        Xtr, Xva = X_full[tr_idx], X_full[va_idx]
        ytr = y_total[tr_idx]
        y_over_va = y_over_bin[va_idx]

        m = XGBRegressor(**params)
        m.fit(Xtr, ytr, verbose=False)
        mu_va = np.clip(m.predict(Xva), 1e-6, 50)
        p_over_raw = p_over_from_mu_and_line_no_push(mu_va, lines[va_idx])
        mask = np.isfinite(p_over_raw)
        y_true = y_over_va[mask]
        if y_true.size > 0 and np.unique(y_true).size > 1:
            aucs.append(roc_auc_score(y_true, p_over_raw[mask]))

    return float(np.mean(aucs)) if aucs else 0.0

print("🔎 Optuna tuning (Poisson)...")
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=40)
best_params = study.best_params
best_params['monotone_constraints'] = build_monotone_constraints(feature_cols)
print("✅ Best params:", best_params)

# ----------------------------- CV Train & Calibrate (plain per-line iso) -----------------------------
oof_p_over_raw = np.full(len(df), np.nan)
oof_p_over_cal = np.full(len(df), np.nan)

for fold_id, (tr_idx, va_idx) in enumerate(make_season_folds(dates, n_splits=5), start=1):
    Xtr, Xva = X_full[tr_idx], X_full[va_idx]
    ytr, yva_total = y_total[tr_idx], y_total[va_idx]
    lines_va = lines[va_idx]
    buckets_va = line_buckets_arr[va_idx]

    model = XGBRegressor(
        objective='count:poisson',
        tree_method='hist',
        random_state=42,
        **best_params
    )
    model.fit(Xtr, ytr, eval_set=[(Xva, yva_total)], verbose=False)

    mu_va = np.clip(model.predict(Xva), 1e-6, 50)
    p_over_raw = p_over_from_mu_and_line_no_push(mu_va, lines_va)
    oof_p_over_raw[va_idx] = p_over_raw

    # per-bucket isotonic + global fallback
    mask_eval = np.isfinite(p_over_raw)
    yva_over = y_over_bin[va_idx][mask_eval]
    pva_raw = p_over_raw[mask_eval]
    bva = buckets_va[mask_eval]

    cals_this_fold = {}
    iso_fallback = None
    if len(yva_over) >= 50 and np.unique(yva_over).size > 1:
        iso_fallback = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip').fit(pva_raw, yva_over)

    for b in np.unique(bva):
        idx_b = (bva == b)
        if idx_b.sum() >= 25:
            yb = yva_over[idx_b]
            if np.unique(yb).size > 1:
                iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip')
                iso.fit(pva_raw[idx_b], yb)
                cals_this_fold[float(b)] = iso

    # apply iso calibration
    p_cal = p_over_raw.copy()
    for i in range(len(p_cal)):
        if not np.isfinite(p_cal[i]):
            continue
        b = buckets_va[i]
        if b in cals_this_fold:
            p_cal[i] = cals_this_fold[b].predict([p_cal[i]])[0]
        elif iso_fallback is not None:
            p_cal[i] = iso_fallback.predict([p_cal[i]])[0]

    oof_p_over_cal[va_idx] = p_cal
    print(f"Fold {fold_id}: {len(cals_this_fold)} bucket calibrators"
          f"{' + fallback' if iso_fallback is not None else ''}")

# ----------------------------- OOF Metrics -----------------------------
mask_eval = y_eval_mask & np.isfinite(oof_p_over_cal)
y_true = y_over_bin[mask_eval].astype(int)
p_cal = oof_p_over_cal[mask_eval]

roc = roc_auc_score(y_true, p_cal) if np.unique(y_true).size > 1 else float('nan')
brier = brier_score_loss(y_true, p_cal)
f1_05 = f1_score(y_true, (p_cal >= 0.5).astype(int))
print("\nOOF base metrics -> ROC-AUC: %.4f | Brier: %.4f | F1@0.50: %.4f" % (roc, brier, f1_05))

# ----------------------------- EV threshold (price-aware; defaults to -110 if missing) -----------------------------
EV_GRID = np.arange(0.0, 0.151, 0.005)
MIN_BETS = 12000
SEASON_OK_RATIO = 0.60
CAP_PER_BET = 0.25

prices_over  = pd.to_numeric(df.loc[mask_eval, 'price_over' ]).to_numpy()
prices_under = pd.to_numeric(df.loc[mask_eval, 'price_under']).to_numpy()
seasons_eval = seasons_arr[mask_eval]

best_thr, best_profit, best_bets, best_winrate = 0.0, -1e9, 0, 0.0
for thr in EV_GRID:
    profit = 0.0
    bets = 0
    wins = 0
    season_pnl = {}

    for i, prob in enumerate(p_cal):
        # price-aware EV (no push); falls back to -110/-110 if NaN
        ev_o = prob * american_to_multiplier(prices_over[i]) - (1.0 - prob) * 1.0
        ev_u = (1.0 - prob) * american_to_multiplier(prices_under[i]) - prob * 1.0
        # cap outliers
        ev_o = max(-CAP_PER_BET, min(CAP_PER_BET, ev_o))
        ev_u = max(-CAP_PER_BET, min(CAP_PER_BET, ev_u))

        chosen = 1 if ev_o >= ev_u else 0
        ev_chosen = ev_o if chosen == 1 else ev_u

        if ev_chosen > thr:
            bets += 1
            win_mult = american_to_multiplier(prices_over[i] if chosen == 1 else prices_under[i])
            if y_true[i] == chosen:
                profit += win_mult
                wins += 1
                season_pnl[seasons_eval[i]] = season_pnl.get(seasons_eval[i], 0.0) + win_mult
            else:
                profit -= 1.0
                season_pnl[seasons_eval[i]] = season_pnl.get(seasons_eval[i], 0.0) - 1.0

    if bets >= MIN_BETS and season_pnl:
        ok_ratio = np.mean([1.0 if v >= -1e-9 else 0.0 for v in season_pnl.values()])
        if ok_ratio >= SEASON_OK_RATIO and profit > best_profit:
            best_profit = profit
            best_thr = float(thr)
            best_bets = bets
            best_winrate = wins / bets if bets else 0.0

print("\n======= OOF Performance (Poisson → per-line Isotonic → EV threshold, no pushes) =======")
print(f"OOF ROC-AUC: {roc:.4f} | OOF Brier: {brier:.4f} | F1@0.50: {f1_05:.4f}")
print(f"Best EV threshold: {best_thr:.3f} | Bets: {best_bets} | Win%: {best_winrate:.3f} | Units (approx): {best_profit:.2f}")

# ----------------------------- Fit final model & global calibrators -----------------------------
print("\nTraining final model on all data...")
final_model = XGBRegressor(
    objective='count:poisson',
    tree_method='hist',
    random_state=42,
    **best_params
)
final_model.fit(X_full, y_total, verbose=False)

# Global isotonic from OOF raw (unweighted), by bucket + fallback
global_by_bucket = {}
global_fallback = None
mask_all = np.isfinite(oof_p_over_raw)
if mask_all.sum() >= 50 and np.unique(y_over_bin[mask_all]).size > 1:
    global_fallback = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip').fit(
        oof_p_over_raw[mask_all], y_over_bin[mask_all].astype(int)
    )
for b in np.unique(line_buckets_arr[mask_all]):
    idx = mask_all & (line_buckets_arr == b)
    if idx.sum() >= 50 and np.unique(y_over_bin[idx]).size > 1:
        iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip')
        iso.fit(oof_p_over_raw[idx], y_over_bin[idx].astype(int))
        global_by_bucket[float(b)] = iso

# ----------------------------- Persist -----------------------------
os.makedirs("models_xgb_poisson", exist_ok=True)

joblib.dump({
    'model': final_model,
    'feature_cols': feature_cols,
    'monotone_constraints': best_params['monotone_constraints'],
}, "models_xgb_poisson/model.joblib")

joblib.dump({
    'by_bucket': global_by_bucket,
    'fallback': global_fallback,
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
        'best_winrate': float(best_winrate),
        'best_params': best_params
    }, f, indent=2)

# Preview derived labels (for sanity)
df_out = df.copy()
df_out[['total_runs', 'over_under_label']].to_csv("models_xgb_poisson/labels_preview.csv", index=False)

print("\nSaved:")
print(" - models_xgb_poisson/model.joblib")
print(" - models_xgb_poisson/calibration.joblib")
print(" - models_xgb_poisson/metrics_oof.json")
print(" - models_xgb_poisson/labels_preview.csv")
