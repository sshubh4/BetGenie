#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MLB Over/Under (price-aware):
XGB Poisson (Optuna-tuned) -> per-line Isotonic (with global fallback) -> EV threshold
+ season-aware rolling CV
+ guardrails: min bets & per-season robustness
+ monotone constraints on a few interpretable features
"""
import os
import math
import json
import joblib
import numpy as np
import pandas as pd
import optuna

from collections import defaultdict
from sklearn.model_selection import TimeSeriesSplit
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, brier_score_loss, f1_score
from xgboost import XGBRegressor

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ----------------------------- Helpers -----------------------------
def poisson_cdf(k: int, mu: float) -> float:
    if mu <= 0:
        return 1.0 if k >= 0 else 0.0
    term = math.exp(-mu)  # n=0
    s = term
    up_to = max(0, k)
    for n in range(1, up_to + 1):
        term *= mu / n
        s += term
    return min(1.0, max(0.0, s))

def poisson_sf(k: int, mu: float) -> float:
    return max(0.0, 1.0 - poisson_cdf(k, mu))

def line_bucket(x: float) -> float:
    if pd.isna(x):
        return np.nan
    return round(float(x) * 2.0) / 2.0

def p_over_from_mu_and_line(mu: np.ndarray, line: np.ndarray):
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

def ev_for_sides(p_over: float, p_push: float, price_over: float, price_under: float):
    """
    Price-aware EV for Over/Under given American odds.
    If price_* is NaN, assume -110.
    Profit on win = 100/|odds| for negative odds, or odds/100 for positive odds.
    """
    def american_to_multiplier(odds):
        if not np.isfinite(odds):
            odds = -110.0
        odds = float(odds)
        return (100.0/abs(odds)) if odds < 0 else (odds/100.0)

    mult_over  = american_to_multiplier(price_over)
    mult_under = american_to_multiplier(price_under)

    p_push = 0.0 if (p_push is None or not np.isfinite(p_push)) else max(0.0, p_push)
    p_under = max(0.0, 1.0 - p_over - p_push)

    ev_over  = p_over  * mult_over  - p_under * 1.0
    ev_under = p_under * mult_under - p_over  * 1.0
    return ev_over, ev_under

def make_season_folds(dates: pd.Series, seasons: pd.Series, n_splits=5):
    """
    Rolling, season-aware folds:
    - Sort by date
    - Split across the full span into n_splits chronological folds
    """
    order = np.argsort(pd.to_datetime(dates, errors='coerce').fillna(pd.Timestamp('1900-01-01')))
    idx = np.arange(len(dates))[order]
    # simple rolling blocks
    tscv = TimeSeriesSplit(n_splits=n_splits)
    for tr, va in tscv.split(idx):
        yield idx[tr], idx[va]

# ----------------------------- Load data -----------------------------
DATA_PATH = "jordan_final.csv"
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError("jordan_final.csv not found. Run your merge script first.")

df = pd.read_csv(DATA_PATH)

required = {'R_home','R_away','total_line','over_result','Date_Parsed'}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Dataset is missing required columns: {missing}")

# Core filters
df = df.copy()
df['total_line'] = pd.to_numeric(df['total_line'], errors='coerce')
mask_ok = df['total_line'].notna() & pd.to_numeric(df['R_home'], errors='coerce').notna() & pd.to_numeric(df['R_away'], errors='coerce').notna()
df = df.loc[mask_ok].reset_index(drop=True)

# time order (will also use in folds)
df['Date_Parsed'] = pd.to_datetime(df['Date_Parsed'], errors='coerce')
df = df.sort_values('Date_Parsed').reset_index(drop=True)

# Labels
y_over = df['over_result']
if str(y_over.dtype) != 'Int64':
    y_over = pd.to_numeric(y_over, errors='coerce').astype('Int64')
y_eval_mask = y_over.notna().values
y_bin = y_over.fillna(-1).astype(int).values
y_total = (pd.to_numeric(df['R_home'], errors='coerce') + pd.to_numeric(df['R_away'], errors='coerce')).values

# ----------------------------- Feature selection -----------------------------
# Keep only numerics. We'll drop labels and IDs below; model script will also guard.
num_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]
feature_cols = [c for c in num_cols if c not in {'R_home','R_away','over_result'}]

# Safety: ensure we have market prices columns (may be NaN sometimes)
for c in ['price_over','price_under']:
    if c not in df.columns:
        df[c] = np.nan
feature_cols = [c for c in feature_cols]  # already numeric

# Small monotone constraint map (by feature index order)
# +1: total_line, temperature_2m ; -1: starter_era_sum ; others 0
def build_monotone_constraints(cols):
    cons = []
    for c in cols:
        if c == 'total_line':
            cons.append(1)
        elif c == 'temperature_2m':
            cons.append(1)
        elif c == 'starter_era_sum':
            cons.append(-1)
        else:
            cons.append(0)
    return "(" + ",".join(str(x) for x in cons) + ")"

# Build matrices
X_full = df[feature_cols].astype(float).fillna(0.0).values
lines = df['total_line'].values
line_buckets = np.array([line_bucket(x) for x in lines])
seasons = df['Date_Parsed'].dt.year.fillna(1900).astype(int)

# ----------------------------- Optuna tuning -----------------------------
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
    for tr_idx, va_idx in make_season_folds(df['Date_Parsed'], seasons, n_splits=5):
        Xtr, Xva = X_full[tr_idx], X_full[va_idx]
        ytr = y_total[tr_idx]
        y_over_va = y_over.iloc[va_idx]
        y_bin_va = y_over_va.fillna(-1).astype(int).values
        m_eval_va = y_over_va.notna().values

        m = XGBRegressor(**params)
        m.fit(Xtr, ytr, verbose=False)
        mu_va = np.clip(m.predict(Xva), 1e-6, 50)
        p_over_raw, _ = p_over_from_mu_and_line(mu_va, lines[va_idx])
        mask = m_eval_va & np.isfinite(p_over_raw)
        y_true = (y_bin_va[mask] == 1).astype(int)
        if y_true.size > 0 and np.unique(y_true).size > 1:
            aucs.append(roc_auc_score(y_true, p_over_raw[mask]))

    return float(np.mean(aucs)) if aucs else 0.0

print("🔎 Optuna tuning...")
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=40)  # increase for deeper search
best_params = study.best_params
best_params['monotone_constraints'] = build_monotone_constraints(feature_cols)
print("✅ Best params:", best_params)

# ----------------------------- CV Train & Calibrate (with fallback) -----------------------------
oof_p_over_raw = np.full(len(df), np.nan)
oof_p_push = np.full(len(df), np.nan)
oof_p_over_cal = np.full(len(df), np.nan)

fold_cals = []
for fold_id, (tr_idx, va_idx) in enumerate(make_season_folds(df['Date_Parsed'], seasons, n_splits=5), start=1):
    Xtr, Xva = X_full[tr_idx], X_full[va_idx]
    ytr, yva_total = y_total[tr_idx], y_total[va_idx]
    lines_va = lines[va_idx]
    buckets_va = line_buckets[va_idx]

    model = XGBRegressor(
        objective='count:poisson',
        tree_method='hist',
        random_state=42,
        **best_params
    )
    model.fit(Xtr, ytr, eval_set=[(Xva, yva_total)], verbose=False)

    mu_va = np.clip(model.predict(Xva), 1e-6, 50)
    p_over_raw, p_push = p_over_from_mu_and_line(mu_va, lines_va)
    oof_p_over_raw[va_idx] = p_over_raw
    oof_p_push[va_idx] = p_push

    # per-bucket isotonic; build also a fold-global fallback
    mask_eval = y_over.iloc[va_idx].notna().values & np.isfinite(p_over_raw)
    yva_over = (y_bin[va_idx][mask_eval] == 1).astype(int)
    pva_raw = p_over_raw[mask_eval]
    bva = buckets_va[mask_eval]

    cals_this_fold = {}
    # global fallback
    iso_fallback = None
    if len(yva_over) >= 50:
        iso_fallback = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip').fit(pva_raw, yva_over)

    for b in np.unique(bva):
        idx_b = (bva == b)
        if idx_b.sum() >= 25:
            iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip')
            iso.fit(pva_raw[idx_b], yva_over[idx_b])
            cals_this_fold[float(b)] = iso

    # apply
    p_cal = p_over_raw.copy()
    for i in range(len(p_cal)):
        if not np.isfinite(p_cal[i]):
            continue
        b = buckets_va[i]
        if b in cals_this_fold:
            p_cal[i] = cals_this_fold[b].predict([p_cal[i]])[0]
        elif iso_fallback is not None:
            p_cal[i] = iso_fallback.predict([p_cal[i]])[0]
        # else: leave raw

    oof_p_over_cal[va_idx] = p_cal
    fold_cals.append({'by_bucket': cals_this_fold, 'fallback': iso_fallback})
    print(f"Fold {fold_id}: {len(cals_this_fold)} bucket calibrators"
          f"{' + fallback' if iso_fallback is not None else ''}")

# ----------------------------- OOF Metrics & EV threshold (guardrailed) -----------------------------
# ----------------------------- OOF Metrics & EV threshold (guardrailed) -----------------------------
mask_eval = y_eval_mask & np.isfinite(oof_p_over_cal)
y_true = (y_bin[mask_eval] == 1).astype(int)
p_cal = oof_p_over_cal[mask_eval]
p_push_eval = oof_p_push[mask_eval]

# convert to numpy arrays to avoid label-vs-position KeyErrors
prices_over  = df.loc[mask_eval, 'price_over' ].to_numpy()
prices_under = df.loc[mask_eval, 'price_under'].to_numpy()
seasons_eval = seasons.loc[mask_eval].to_numpy()

roc = roc_auc_score(y_true, p_cal) if np.unique(y_true).size > 1 else float('nan')
brier = brier_score_loss(y_true, p_cal)
f1_05 = f1_score(y_true, (p_cal >= 0.5).astype(int))
print("\nOOF base metrics -> ROC-AUC: %.4f | Brier: %.4f | F1@0.50: %.4f" % (roc, brier, f1_05))

EV_GRID = np.arange(0.0, 0.151, 0.005)
MIN_BETS = 1500
SEASON_OK_RATIO = 0.60
CAP_PER_BET = 0.25

def american_to_multiplier(odds):
    # Return profit-per-unit on win (e.g., -110 -> 0.909, +120 -> 1.20)
    if not np.isfinite(odds):
        odds = -110.0
    odds = float(odds)
    return (100.0 / abs(odds)) if odds < 0 else (odds / 100.0)

best_thr, best_profit, best_bets, best_winrate = 0.0, -1e9, 0, 0.0
for thr in EV_GRID:
    profit = 0.0
    bets = 0
    wins = 0
    # per-season realized PnL
    season_pnl = {}

    for i, prob in enumerate(p_cal):
        # expected values (still used only for gating on thr; capped to avoid outliers)
        ev_o, ev_u = ev_for_sides(prob, p_push_eval[i], prices_over[i], prices_under[i])
        ev_o = max(-CAP_PER_BET, min(CAP_PER_BET, ev_o))
        ev_u = max(-CAP_PER_BET, min(CAP_PER_BET, ev_u))

        # choose side with higher EV
        chosen = 1 if ev_o >= ev_u else 0
        ev_chosen = ev_o if chosen == 1 else ev_u

        if ev_chosen > thr:
            bets += 1
            # realized outcome with true payout
            if chosen == 1:
                win_mult = american_to_multiplier(prices_over[i])
            else:
                win_mult = american_to_multiplier(prices_under[i])

            if y_true[i] == chosen:    # win
                profit += win_mult      # +profit multiplier
                wins += 1
                season_pnl[seasons_eval[i]] = season_pnl.get(seasons_eval[i], 0.0) + win_mult
            else:                       # loss
                profit -= 1.0
                season_pnl[seasons_eval[i]] = season_pnl.get(seasons_eval[i], 0.0) - 1.0

    # season robustness: >=60% seasons non-negative
    if bets >= MIN_BETS and season_pnl:
        ok_ratio = np.mean([1.0 if v >= -1e-9 else 0.0 for v in season_pnl.values()])
        if ok_ratio >= SEASON_OK_RATIO and profit > best_profit:
            best_profit = profit
            best_thr = float(thr)
            best_bets = bets
            best_winrate = wins / bets if bets else 0.0

print("\n======= OOF Performance (Tuned XGB Poisson -> per-line Isotonic -> EV threshold, price-aware) =======")
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

# Build global calibrators from all OOF raw (bucketed) with fallback
global_by_bucket = {}
global_fallback = None
mask_all = y_eval_mask & np.isfinite(oof_p_over_raw)
if mask_all.sum() >= 50:
    global_fallback = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip').fit(
        oof_p_over_raw[mask_all], (y_bin[mask_all] == 1).astype(int)
    )
for b in np.unique(line_buckets[mask_all]):
    idx = mask_all & (line_buckets == b)
    if idx.sum() >= 50:
        iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip')
        iso.fit(oof_p_over_raw[idx], (y_bin[idx] == 1).astype(int))
        global_by_bucket[float(b)] = iso

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

print("\nSaved:")
print(" - models_xgb_poisson/model.joblib")
print(" - models_xgb_poisson/calibration.joblib")
print(" - models_xgb_poisson/metrics_oof.json")
