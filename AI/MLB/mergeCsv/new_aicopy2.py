#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MLB Over/Under: XGB Poisson -> per-line Isotonic -> EV threshold (with pushes + per-season EV check)

Changes:
- Expanded EV threshold sweep from -0.05 → 0.25 (step 0.005)
- Added per-season profit breakdown for robustness check
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
    if mu <= 0:
        return 1.0 if k >= 0 else 0.0
    term = math.exp(-mu)
    s = term
    for n in range(1, max(0, k) + 1):
        term *= mu / n
        s += term
    return min(1.0, max(0.0, s))

def poisson_sf(k: int, mu: float) -> float:
    return max(0.0, 1.0 - poisson_cdf(k, mu))

def line_bucket(x: float) -> float:
    if pd.isna(x):
        return np.nan
    return round(float(x) * 2.0) / 2.0

def pick_features(df: pd.DataFrame) -> list:
    drop_exact = {'over_result','over_under_label','total_line',
                  'Date_Parsed','date','game_id','home_team','away_team'}
    leakage_tokens = {'_home','_away','_label','actual','result'}
    keep = []
    for c in df.columns:
        if c in drop_exact:
            continue
        if any(tok in c for tok in ['Boxscore','merge_key']):
            continue
        if np.issubdtype(df[c].dtype, np.number):
            if any(c.endswith(tok) for tok in leakage_tokens):
                continue
            keep.append(c)
    return keep

def total_runs_from_df(df: pd.DataFrame) -> np.ndarray:
    r_home = pd.to_numeric(df['R_home'], errors='coerce')
    r_away = pd.to_numeric(df['R_away'], errors='coerce')
    return (r_home + r_away).values

def p_over_from_mu_and_line(mu: np.ndarray, line: np.ndarray) -> (np.ndarray, np.ndarray):
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
            lg = -m + k * math.log(m) - math.lgamma(k + 1)
            p_push[i] = math.exp(lg)
        else:
            k = math.floor(L)
            p_over[i] = poisson_sf(k, m)
            p_push[i] = 0.0
    return p_over, p_push

def ev_for_sides(p_over: float, p_push: float) -> (float, float):
    p_under = max(0.0, 1.0 - p_over - max(0.0, p_push))
    ev_over  = p_over  * 0.909 - p_under * 1.0
    ev_under = p_under * 0.909 - p_over  * 1.0
    return ev_over, ev_under

# ----------------------------- Load data -----------------------------
data_path = "jordan_final.csv"
if not os.path.exists(data_path):
    raise FileNotFoundError("jordan_final.csv not found. Run your merge script first.")

df = pd.read_csv(data_path)

required_cols = {'R_home','R_away','total_line','over_result'}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"Dataset missing required columns: {missing}")

df['total_line'] = pd.to_numeric(df['total_line'], errors='coerce')
mask_ok = df['total_line'].notna() & pd.to_numeric(df['R_home'], errors='coerce').notna() & pd.to_numeric(df['R_away'], errors='coerce').notna()
df = df.loc[mask_ok].reset_index(drop=True)

y_over = df['over_result'].copy()
if str(y_over.dtype) != 'Int64':
    y_over = pd.to_numeric(y_over, errors='coerce').astype('Int64')
y_eval_mask = y_over.notna().values
y_bin = y_over.fillna(-1).astype(int).values

# ----------------------------- Features & Target -----------------------------
y_total = total_runs_from_df(df)
feature_cols = pick_features(df)
for drop_col in ['R_home','R_away']:
    if drop_col in feature_cols:
        feature_cols.remove(drop_col)

X = df[feature_cols].astype(float).fillna(0.0).values
lines = df['total_line'].values
line_buckets = np.array([line_bucket(x) for x in lines])

if 'Date_Parsed' in df.columns:
    order = pd.to_datetime(df['Date_Parsed'], errors='coerce').fillna(pd.Timestamp('1900-01-01'))
    df = df.iloc[np.argsort(order.values)].reset_index(drop=True)
    X = df[feature_cols].astype(float).fillna(0.0).values
    lines = df['total_line'].values
    line_buckets = np.array([line_bucket(x) for x in lines])
    y_total = total_runs_from_df(df)
    y_over = df['over_result']
    if str(y_over.dtype) != 'Int64':
        y_over = pd.to_numeric(y_over, errors='coerce').astype('Int64')
    y_eval_mask = y_over.notna().values
    y_bin = y_over.fillna(-1).astype(int).values

# ----------------------------- CV Train & Calibrate -----------------------------
tscv = TimeSeriesSplit(n_splits=5)
oof_p_over_raw = np.full(len(df), np.nan)
oof_p_push = np.full(len(df), np.nan)
oof_p_over_cal = np.full(len(df), np.nan)

fold_bucket_cals = []
fold_id = 1
for tr_idx, va_idx in tscv.split(X):
    Xtr, Xva = X[tr_idx], X[va_idx]
    ytr, yva_total = y_total[tr_idx], y_total[va_idx]
    lines_va = lines[va_idx]
    buckets_va = line_buckets[va_idx]

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
    model.fit(Xtr, ytr, eval_set=[(Xva, yva_total)], verbose=False)

    mu_va = model.predict(Xva)
    mu_va = np.clip(mu_va, 1e-6, 50)
    p_over_raw, p_push = p_over_from_mu_and_line(mu_va, lines_va)

    oof_p_over_raw[va_idx] = p_over_raw
    oof_p_push[va_idx] = p_push

    cals_this_fold = {}
    va_mask_eval = (y_over.iloc[va_idx].notna().values) & np.isfinite(p_over_raw)
    yva_over = (y_bin[va_idx][va_mask_eval] == 1).astype(int)
    pva_raw = p_over_raw[va_mask_eval]
    bva = buckets_va[va_mask_eval]

    for b in np.unique(bva):
        idx_b = (bva == b)
        if idx_b.sum() >= 25:
            iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip')
            iso.fit(pva_raw[idx_b], yva_over[idx_b])
            cals_this_fold[b] = iso
    fold_bucket_cals.append(cals_this_fold)

    p_cal = p_over_raw.copy()
    for b, iso in cals_this_fold.items():
        idx_b = (buckets_va == b)
        p_cal[idx_b] = iso.predict(p_over_raw[idx_b])
    oof_p_over_cal[va_idx] = p_cal

    print(f"Fold {fold_id}: calibrators built for buckets: {sorted(list(cals_this_fold.keys()))[:6]}{' ...' if len(cals_this_fold)>6 else ''}")
    fold_id += 1

# ----------------------------- OOF Metrics & EV threshold -----------------------------
mask_eval = y_eval_mask & np.isfinite(oof_p_over_cal)
y_true = (y_bin[mask_eval] == 1).astype(int)
p_cal = oof_p_over_cal[mask_eval]
p_push_eval = oof_p_push[mask_eval]

roc = roc_auc_score(y_true, p_cal) if len(np.unique(y_true)) > 1 else np.nan
brier = brier_score_loss(y_true, p_cal)
f1_05 = f1_score(y_true, (p_cal >= 0.5).astype(int))

thr_grid = np.arange(-0.05, 0.251, 0.005)
best_thr, best_profit, best_bets, best_winrate = 0.0, -1e9, 0, 0.0
for thr in thr_grid:
    profit = 0.0
    bets = 0
    wins = 0
    for i, prob in enumerate(p_cal):
        ev_over, ev_under = ev_for_sides(prob, p_push_eval[i] if np.isfinite(p_push_eval[i]) else 0.0)
        if ev_over >= ev_under:
            chosen = 1
            ev = ev_over
        else:
            chosen = 0
            ev = ev_under
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

print("\n======= OOF Performance (XGB Poisson -> per-line Isotonic -> EV threshold) =======")
print(f"OOF ROC-AUC: {roc:.4f} | OOF Brier: {brier:.4f} | OOF F1@0.50: {f1_05:.4f}")
print(f"Best EV threshold: {best_thr:.3f} | Bets: {best_bets} | Win%: {best_winrate:.3f} | Total units: {best_profit:.2f}")

# --- Per-season EV check ---
if 'Season' in df.columns:
    df_eval = df.loc[mask_eval].copy()
    df_eval['y_true'] = y_true
    df_eval['p_cal'] = p_cal
    df_eval['p_push'] = p_push_eval
    df_eval['ev_over'], df_eval['ev_under'] = zip(*[ev_for_sides(prob, push if np.isfinite(push) else 0.0)
                                                    for prob, push in zip(p_cal, p_push_eval)])
    df_eval['chosen'] = (df_eval['ev_over'] >= df_eval['ev_under']).astype(int)
    df_eval['ev'] = df_eval[['ev_over','ev_under']].max(axis=1)
    df_eval['bet'] = df_eval['ev'] > best_thr
    df_eval['won'] = (df_eval['y_true'] == df_eval['chosen'])
    df_eval['units'] = np.where(df_eval['bet'] & df_eval['won'], 0.909,
                                np.where(df_eval['bet'] & ~df_eval['won'], -1.0, 0.0))
    per_season = df_eval.groupby('Season')['units'].sum()
    print("\nPer-season units (robustness check):")
    print(per_season)

# ----------------------------- Fit final model & Save -----------------------------
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

os.makedirs("models_xgb_poisson", exist_ok=True)

joblib.dump({'model': final_model, 'feature_cols': feature_cols},
            "models_xgb_poisson/model.joblib")
joblib.dump({'global_calibrators': global_cals,
             'line_bucket_fn': 'round(x*2)/2',
             'ev_threshold': best_thr},
            "models_xgb_poisson/calibration.joblib")
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
