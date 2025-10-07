#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MLB O/U — Two-head XGB (μ_home + μ_away) -> Negative Binomial tail -> per-line Isotonic -> EV threshold

What this script does
---------------------
1) Loads jordan_final.csv produced by your merge script.
2) Picks leakage-safe numeric features automatically.
3) Trains two XGBRegressor heads with Poisson objective to predict μ_home and μ_away.
4) Sums them to get μ_total, then converts to probabilities with a Negative Binomial tail.
   - Dispersion κ is selected per fold to MAXIMIZE OOF ROC-AUC (on non-push rows).
5) Calibrates p(Over) per 0.5 total-line bucket via isotonic regression (OOF).
6) Evaluates OOF ROC-AUC / Brier / F1@0.5 and runs an EV sweep to choose a deployment threshold.
   - EV uses market prices if present (price_over/price_under); otherwise assumes -110 both ways.
7) Fits final models on ALL data, builds global per-bucket calibrators (from OOF), and saves artifacts.

Artifacts saved in ./models_nb_twohead/
- model.joblib        : {'home_model','away_model','feature_cols','global_kappa'}
- calibration.joblib  : {'global_calibrators','line_bucket_fn','ev_threshold'}
- metrics_oof.json    : summary metrics and chosen params
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

def american_to_winprofit_per_unit(odds):
    """
    Convert American odds to profit per 1 unit stake (excluding stake).
    +150 -> 1.50 ; -110 -> 100/110 ≈ 0.909 ; +100 -> 1.00
    """
    if odds is None or (isinstance(odds, float) and not np.isfinite(odds)):
        return None
    o = float(odds)
    if o > 0:
        return o / 100.0
    else:
        return 100.0 / abs(o)

def odds_pair_to_profits(over_odds, under_odds, default=-110):
    """
    Return (profit_if_over_wins, profit_if_under_wins) per 1u stake.
    Falls back to default odds when missing.
    """
    po = american_to_winprofit_per_unit(over_odds) if over_odds is not None else None
    pu = american_to_winprofit_per_unit(under_odds) if under_odds is not None else None
    if po is None:
        po = american_to_winprofit_per_unit(default)
    if pu is None:
        pu = american_to_winprofit_per_unit(default)
    return po, pu

def line_bucket(x: float) -> float:
    """Bucket total line to nearest 0.5 to increase data per calibrator."""
    if pd.isna(x):
        return np.nan
    return round(float(x) * 2.0) / 2.0

def pick_features(df: pd.DataFrame) -> list:
    """
    Automatic, leakage-aware feature selection from jordan_final.csv.
    Keeps numeric features; drops labels, explicit outputs, ids, and obvious leakage tokens.
    """
    drop_exact = {
        'over_result', 'over_under_label', 'total_line',
        'Date_Parsed', 'date', 'game_id', 'home_team', 'away_team'
    }
    leakage_suffix = {'_label', '_actual', '_result'}  # explicit labels
    # We will explicitly drop R_home/R_away later (they’re the target components)
    keep = []
    for c in df.columns:
        if c in drop_exact:
            continue
        if any(tok in c for tok in ['Boxscore', 'merge_key']):
            continue
        if np.issubdtype(df[c].dtype, np.number):
            if any(c.endswith(tok) for tok in leakage_suffix):
                continue
            keep.append(c)
    # Safety: never keep the raw outcomes
    for drop_col in ['R_home', 'R_away']:
        if drop_col in keep:
            keep.remove(drop_col)
    return keep

def total_runs_from_df(df: pd.DataFrame) -> np.ndarray:
    r_home = pd.to_numeric(df['R_home'], errors='coerce')
    r_away = pd.to_numeric(df['R_away'], errors='coerce')
    return (r_home + r_away).values

# ---------------- Negative Binomial tail (μ, κ) --------------------

def nb_logpmf(k: int, mu: float, kappa: float) -> float:
    """
    Negative Binomial parameterized by mean mu and dispersion kappa (>0).
    Var = mu + kappa * mu^2. Size r = 1/kappa, p = r/(r+mu).
    Returns log P(N=k).
    """
    if mu <= 0 or k < 0:
        return -1e30 if k != 0 else 0.0
    r = max(1e-8, 1.0 / kappa)
    p = r / (r + mu)
    return (math.lgamma(k + r) - math.lgamma(r) - math.lgamma(k + 1)
            + r * math.log(p) + k * math.log(1.0 - p))

def nb_cdf(k: int, mu: float, kappa: float) -> float:
    """P(N <= k) for NB(mu, kappa) via stable sum of PMF terms."""
    if mu <= 0:
        return 1.0 if k >= 0 else 0.0
    s = 0.0
    maxk = max(0, k)
    for n in range(0, maxk + 1):
        s += math.exp(nb_logpmf(n, mu, kappa))
    return min(1.0, max(0.0, s))

def nb_sf(k: int, mu: float, kappa: float) -> float:
    """P(N > k)"""
    return max(0.0, 1.0 - nb_cdf(k, mu, kappa))

def p_over_push_from_nb(mu: np.ndarray, line: np.ndarray, kappa: float) -> (np.ndarray, np.ndarray):
    """
    Vectorized P(Over) and P(Push) under NB(mu, kappa) vs total_line.

    If line is integer: Over is P(N > k), Push is P(N == k).
    If line is half-number (e.g. 7.5): Over is P(N >= ceil(line)) = P(N > floor(line)); Push = 0.
    """
    p_over = np.zeros_like(mu, dtype=float)
    p_push = np.zeros_like(mu, dtype=float)
    for i, (m, L) in enumerate(zip(mu, line)):
        if pd.isna(L) or not np.isfinite(m) or m <= 0:
            p_over[i] = np.nan
            p_push[i] = np.nan
            continue
        if float(L).is_integer():
            k = int(L)
            p_over[i] = nb_sf(k, m, kappa)           # P(N > k)
            p_push[i] = math.exp(nb_logpmf(k, m, kappa))  # P(N == k)
        else:
            k = math.floor(L)                        # e.g. 7.5 -> Over when N>=8 => P(N > 7)
            p_over[i] = nb_sf(k, m, kappa)
            p_push[i] = 0.0
    return p_over, p_push


# -------------------------- Load & Prepare -------------------------

data_path = "jordan_final.csv"
if not os.path.exists(data_path):
    raise FileNotFoundError("jordan_final.csv not found. Run your merge script first to produce it.")

df = pd.read_csv(data_path)

# Basic checks
required_cols = {'R_home', 'R_away', 'total_line', 'over_result'}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"Your dataset is missing required columns: {missing}")

# keep only rows with line + outcomes
df['total_line'] = pd.to_numeric(df['total_line'], errors='coerce')
mask_ok = df['total_line'].notna() & pd.to_numeric(df['R_home'], errors='coerce').notna() & pd.to_numeric(df['R_away'], errors='coerce').notna()
df = df.loc[mask_ok].reset_index(drop=True)

# Prepare label (mask out pushes later)
y_over = df['over_result'].copy()
if str(y_over.dtype) != 'Int64':
    y_over = pd.to_numeric(y_over, errors='coerce').astype('Int64')  # {0,1,<NA>}
y_eval_mask = y_over.notna().values
y_bin = y_over.fillna(-1).astype(int).values  # -1 for push

# Time ordering
if 'Date_Parsed' in df.columns:
    order = pd.to_datetime(df['Date_Parsed'], errors='coerce').fillna(pd.Timestamp('1900-01-01'))
    df = df.iloc[np.argsort(order.values)].reset_index(drop=True)
    y_over = df['over_result']
    if str(y_over.dtype) != 'Int64':
        y_over = pd.to_numeric(y_over, errors='coerce').astype('Int64')
    y_eval_mask = y_over.notna().values
    y_bin = y_over.fillna(-1).astype(int).values

# Features
feature_cols = pick_features(df)
X = df[feature_cols].astype(float).fillna(0.0).values  # base features for both heads
X_home = X
X_away = X

# Targets for heads
y_home = pd.to_numeric(df['R_home'], errors='coerce').values
y_away = pd.to_numeric(df['R_away'], errors='coerce').values

lines = df['total_line'].values
line_buckets = np.array([line_bucket(x) for x in lines])


# ------------------ CV Train, κ selection, Calibration ------------------

tscv = TimeSeriesSplit(n_splits=5)

oof_mu = np.full(len(df), np.nan)           # μ_total
oof_p_over_raw = np.full(len(df), np.nan)   # raw NB p(Over)
oof_p_push = np.full(len(df), np.nan)       # NB push prob
oof_p_over_cal = np.full(len(df), np.nan)   # calibrated p(Over)

fold_bucket_cals = []                       # per-fold calibrators
fold_best_kappas = []                       # keep chosen κ per fold

# κ grid (tune if you want finer search)
kappa_grid = [0.1, 0.2, 0.3, 0.5, 0.8, 1.0]

fold_id = 1
for tr_idx, va_idx in tscv.split(X_home):
    # Train two Poisson heads for mean runs by side
    mdl_h = XGBRegressor(
        objective='count:poisson', tree_method='hist',
        n_estimators=700, max_depth=6, learning_rate=0.035,
        subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0, random_state=42)
    mdl_a = XGBRegressor(
        objective='count:poisson', tree_method='hist',
        n_estimators=700, max_depth=6, learning_rate=0.035,
        subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0, random_state=43)

    mdl_h.fit(X_home[tr_idx], y_home[tr_idx],
              eval_set=[(X_home[va_idx], y_home[va_idx])],
              verbose=False)
    mdl_a.fit(X_away[tr_idx], y_away[tr_idx],
              eval_set=[(X_away[va_idx], y_away[va_idx])],
              verbose=False)

    mu_va = np.clip(mdl_h.predict(X_home[va_idx]) + mdl_a.predict(X_away[va_idx]), 1e-6, 60.0)
    oof_mu[va_idx] = mu_va

    # Pick kappa to maximize ROC-AUC on non-push rows (before calibration)
    yva_over = y_over.iloc[va_idx]
    non_push_mask = yva_over.notna().values
    y_bin_va = (yva_over.fillna(-1).astype(int).values == 1).astype(int)

    best_kappa = None
    best_auc = -1.0
    for kappa in kappa_grid:
        p_over_raw_candidate, p_push_candidate = p_over_push_from_nb(mu_va, lines[va_idx], kappa)
        mm = non_push_mask & np.isfinite(p_over_raw_candidate)
        if mm.sum() < 10:
            continue
        try:
            auc = roc_auc_score(y_bin_va[mm], p_over_raw_candidate[mm])
        except ValueError:
            auc = np.nan
        if np.isfinite(auc) and auc > best_auc:
            best_auc = auc
            best_kappa = kappa

    if best_kappa is None:
        best_kappa = 0.3  # safe fallback

    fold_best_kappas.append(best_kappa)

    # Final raw probs for this fold using best_kappa
    p_over_raw, p_push = p_over_push_from_nb(mu_va, lines[va_idx], best_kappa)
    oof_p_over_raw[va_idx] = p_over_raw
    oof_p_push[va_idx] = p_push

    # Per-bucket isotonic calibration
    cals_this_fold = {}
    buckets_va = line_buckets[va_idx]
    va_mask_eval = (y_over.iloc[va_idx].notna().values) & np.isfinite(p_over_raw)
    yva_over_bin = (y_bin[va_idx][va_mask_eval] == 1).astype(int)
    pva_raw = p_over_raw[va_mask_eval]
    bva = buckets_va[va_mask_eval]

    for b in np.unique(bva):
        idx_b = (bva == b)
        if idx_b.sum() >= 25:
            iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip')
            iso.fit(pva_raw[idx_b], yva_over_bin[idx_b])
            cals_this_fold[b] = iso

    fold_bucket_cals.append(cals_this_fold)

    # Apply calibration to this validation fold
    p_cal = p_over_raw.copy()
    for b, iso in cals_this_fold.items():
        idx_b = (buckets_va == b)
        p_cal[idx_b] = iso.predict(p_over_raw[idx_b])
    oof_p_over_cal[va_idx] = p_cal

    print(f"Fold {fold_id}: picked kappa={best_kappa:.3f}; calibrators for buckets: {sorted(list(cals_this_fold.keys()))[:6]}{' ...' if len(cals_this_fold)>6 else ''}")
    fold_id += 1


# ------------------ OOF Metrics & EV threshold search ------------------

mask_eval = y_eval_mask & np.isfinite(oof_p_over_cal)
y_true = (y_bin[mask_eval] == 1).astype(int)
p_cal = oof_p_over_cal[mask_eval]
p_push_eval = oof_p_push[mask_eval]

# ROC / Brier / F1@0.5
roc = roc_auc_score(y_true, p_cal) if len(np.unique(y_true)) > 1 else np.nan
brier = brier_score_loss(y_true, p_cal)
f1_05 = f1_score(y_true, (p_cal >= 0.5).astype(int))

# EV computation uses market prices if present, else -110
have_prices = {'price_over', 'price_under'}.issubset(df.columns)
po_series = pd.to_numeric(df.loc[mask_eval, 'price_over'], errors='coerce') if have_prices else pd.Series([np.nan]*mask_eval.sum())
pu_series = pd.to_numeric(df.loc[mask_eval, 'price_under'], errors='coerce') if have_prices else pd.Series([np.nan]*mask_eval.sum())

thr_grid = np.arange(0.0, 0.151, 0.005)
best_thr, best_profit, best_bets, best_winrate = 0.0, -1e18, 0, 0.0

for thr in thr_grid:
    profit = 0.0
    bets = 0
    wins = 0
    for i in range(p_cal.shape[0]):
        # payouts per 1u
        prof_over, prof_under = odds_pair_to_profits(
            po_series.iloc[i] if have_prices else None,
            pu_series.iloc[i] if have_prices else None,
            default=-110
        )
        # push prob only affects expected value (skipped implicitly since we use realized outcomes to settle)
        p_over = p_cal[i]
        p_under = max(0.0, 1.0 - p_over)  # calibrated is already conditional on non-push bucket-wise; push EV ≈ 0

        ev_over = p_over * prof_over - p_under * 1.0
        ev_under = p_under * prof_under - p_over * 1.0

        chosen = 1 if ev_over >= ev_under else 0
        ev = max(ev_over, ev_under)

        if ev > thr:
            bets += 1
            # settle with realized label (non-push rows only in mask_eval)
            won = (y_true[i] == chosen)
            if won:
                profit += prof_over if chosen == 1 else prof_under
                wins += 1
            else:
                profit -= 1.0
    if bets > 0 and profit > best_profit:
        best_profit = profit
        best_thr = float(thr)
        best_bets = bets
        best_winrate = wins / bets if bets else 0.0

print("\n======= OOF Performance (Two-head μ -> NB tail -> per-line Isotonic -> EV threshold) =======")
print(f"OOF ROC-AUC: {roc:.4f} | OOF Brier: {brier:.4f} | OOF F1@0.50: {f1_05:.4f}")
print(f"Best EV threshold: {best_thr:.3f} | Bets: {best_bets} | Win%: {best_winrate:.3f} | Total units: {best_profit:.2f}")


# ------------------ Final fit on ALL data & Save artifacts ------------------

# Fit final heads
final_home = XGBRegressor(
    objective='count:poisson', tree_method='hist',
    n_estimators=700, max_depth=6, learning_rate=0.035,
    subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0, random_state=42
).fit(X_home, y_home, verbose=False)

final_away = XGBRegressor(
    objective='count:poisson', tree_method='hist',
    n_estimators=700, max_depth=6, learning_rate=0.035,
    subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0, random_state=43
).fit(X_away, y_away, verbose=False)

# Choose a global κ (median of per-fold best kappas is a stable default)
global_kappa = float(np.median(fold_best_kappas)) if len(fold_best_kappas) else 0.3

# Build GLOBAL per-bucket calibrators from all OOF raw predictions
global_cals = {}
all_mask = y_eval_mask & np.isfinite(oof_p_over_raw)
if all_mask.sum() > 0:
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

# Save
outdir = "models_nb_twohead"
os.makedirs(outdir, exist_ok=True)

joblib.dump({
    'home_model': final_home,
    'away_model': final_away,
    'feature_cols': feature_cols,
    'global_kappa': global_kappa
}, os.path.join(outdir, "model.joblib"))

joblib.dump({
    'global_calibrators': global_cals,
    'line_bucket_fn': 'round(x*2)/2',
    'ev_threshold': best_thr
}, os.path.join(outdir, "calibration.joblib"))

with open(os.path.join(outdir, "metrics_oof.json"), "w") as f:
    json.dump({
        'oof_roc_auc': float(roc),
        'oof_brier': float(brier),
        'oof_f1_at_0.50': float(f1_05),
        'best_ev_threshold': best_thr,
        'best_total_units': float(best_profit),
        'best_bets': int(best_bets),
        'best_winrate': float(best_winrate),
        'kappa_grid': kappa_grid,
        'chosen_kappas_per_fold': [float(k) for k in fold_best_kappas],
        'global_kappa': float(global_kappa),
        'used_market_prices': bool(have_prices)
    }, f, indent=2)

print("\nSaved:")
print(f" - {outdir}/model.joblib")
print(f" - {outdir}/calibration.joblib")
print(f" - {outdir}/metrics_oof.json")
