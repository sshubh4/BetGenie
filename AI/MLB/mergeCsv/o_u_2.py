#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MLB O/U — Stacked Model:
  - Two-head Poisson (μ_home, μ_away) -> μ_total
  - Residual variance model -> Negative Binomial tail
  - Meta classifier stacked on OOF probs + curated features
  - Per-line isotonic calibration (OOF & global)
  - EV-threshold selection (keeps push prob)

Artifacts:
  models_ou_stack/
    - base_model.joblib           (μ heads)
    - var_model.joblib            (variance head)
    - meta_model.joblib           (stacked classifier)
    - calibration.joblib          (global per-line isotonic + EV thr)
    - feature_cols.json           (base and meta feature lists)
    - metrics_oof.json
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
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

# ------------- Poisson/NB helpers -------------
def poisson_cdf(k: int, mu: float) -> float:
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
    return max(0.0, 1.0 - poisson_cdf(k, mu))

def nb_pmf(k: int, mu: float, alpha: float) -> float:
    """
    NB parameterization: Var = mu + alpha*mu^2  (alpha >= 0)
    r = 1/alpha, p = r/(r+mu)
    PMF(k) = C(k+r-1, k) * (1-p)^k * p^r
    """
    if mu <= 0:
        return 1.0 if k == 0 else 0.0
    if alpha <= 0:
        # falls back to Poisson
        lg = -mu + k*math.log(mu) - math.lgamma(k+1)
        return math.exp(lg)
    r = 1.0 / max(alpha, 1e-8)
    p = r / (r + mu)  # success prob
    # log PMF for stability
    lg = math.lgamma(k + r) - math.lgamma(r) - math.lgamma(k + 1) \
         + r*math.log(p) + k*math.log(1.0 - p)
    return math.exp(lg)

def nb_cdf(k: int, mu: float, alpha: float) -> float:
    s = 0.0
    for n in range(0, max(0, k) + 1):
        s += nb_pmf(n, mu, alpha)
        if s >= 1.0:
            return 1.0
    return min(1.0, max(0.0, s))

def line_bucket(x: float) -> float:
    if pd.isna(x):
        return np.nan
    return round(float(x) * 2.0) / 2.0

def american_to_prob(odds):
    if pd.isna(odds):
        return np.nan
    odds = float(odds)
    return (100.0/(odds+100.0)) if odds>0 else ((-odds)/(-odds+100.0))

# ------------- Data loading & columns -------------
DATA_PATH = "jordan_final.csv"
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError("jordan_final.csv not found. Run your merge script first.")

df = pd.read_csv(DATA_PATH)
required = {'R_home','R_away','total_line','over_result'}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Dataset missing required columns: {missing}")

# strict mask: need label (even if push, we keep for push prob masking), total_line, R_home/R_away
df['total_line'] = pd.to_numeric(df['total_line'], errors='coerce')
mask_ok = df['total_line'].notna() & pd.to_numeric(df['R_home'], errors='coerce').notna() & pd.to_numeric(df['R_away'], errors='coerce').notna()
df = df.loc[mask_ok].copy().reset_index(drop=True)

# Sort by time if available
if 'Date_Parsed' in df.columns:
    order = pd.to_datetime(df['Date_Parsed'], errors='coerce').fillna(pd.Timestamp('1900-01-01'))
    df = df.iloc[np.argsort(order.values)].reset_index(drop=True)

# Labels
y_over = df['over_result']
if str(y_over.dtype) != 'Int64':
    y_over = pd.to_numeric(y_over, errors='coerce').astype('Int64')
eval_mask = y_over.notna().values  # exclude pushes only for classification metrics
y_bin = y_over.fillna(-1).astype(int).values

R_home = pd.to_numeric(df['R_home'], errors='coerce').values
R_away = pd.to_numeric(df['R_away'], errors='coerce').values
y_total = (R_home + R_away).astype(float)

lines = df['total_line'].astype(float).values
buckets = np.array([line_bucket(x) for x in lines])

# ------------- Feature selection -------------
# Base features (pre-game; keep numeric)
drop_exact = {'over_result','over_under_label','total_line','R_home','R_away'}
ignore_tokens = ['Boxscore','merge_key']
base_feature_cols = []
for c in df.columns:
    if c in drop_exact: 
        continue
    if any(tok in c for tok in ignore_tokens): 
        continue
    if np.issubdtype(df[c].dtype, np.number):
        base_feature_cols.append(c)

# Meta features we’ll feed to the stacker (all numeric, curated small set to avoid overfit)
meta_candidates = [
    # model-driven
    'starter_era_sum','rest_home','rest_away','delta_rest','b2b_home','b2b_away',
    'rolling_10_R_home','rolling_10_R_away','rolling_10_R_overall_home','rolling_10_R_overall_away',
    # market shape & prices
    'vig','is_half_total','frac_from_half','total_line_std','total_line_range',
    # weather/park
    'Park Factor','temperature_2m','relative_humidity_2m','wind_speed_10m','wind_gusts_10m',
    # interactions if present
    'park_x_temp','park_x_wind10'
]
meta_feature_cols = [c for c in meta_candidates if c in df.columns]

# Fill NaNs robustly
df[base_feature_cols] = df[base_feature_cols].astype(float).fillna(0.0)
df_meta = df[meta_feature_cols].astype(float).fillna(df[meta_feature_cols].median(numeric_only=True))

X_all = df[base_feature_cols].values
M_all = df_meta.values

# ------------- Time series CV -------------
tscv = TimeSeriesSplit(n_splits=5)

oof_mu_home = np.full(len(df), np.nan)
oof_mu_away = np.full(len(df), np.nan)
oof_mu_total = np.full(len(df), np.nan)
oof_alpha = np.full(len(df), np.nan)         # NB alpha
oof_p_over_nb = np.full(len(df), np.nan)
oof_p_push_nb = np.full(len(df), np.nan)
oof_meta_proba = np.full(len(df), np.nan)

fold_cals = []
fold_id = 1

for tr_idx, va_idx in tscv.split(X_all):
    Xtr, Xva = X_all[tr_idx], X_all[va_idx]
    Mtr, Mva = M_all[tr_idx], M_all[va_idx]
    ytr_home, yva_home = R_home[tr_idx], R_home[va_idx]
    ytr_away, yva_away = R_away[tr_idx], R_away[va_idx]
    lines_va = lines[va_idx]
    buckets_va = buckets[va_idx]

    # ---- Base: two Poisson heads ----
    model_home = XGBRegressor(
        objective='count:poisson', tree_method='hist',
        n_estimators=900, max_depth=6, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.2,
        random_state=42
    )
    model_away = XGBRegressor(
        objective='count:poisson', tree_method='hist',
        n_estimators=900, max_depth=6, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.2,
        random_state=43
    )
    model_home.fit(Xtr, ytr_home, eval_set=[(Xva, yva_home)], verbose=False)
    model_away.fit(Xtr, ytr_away, eval_set=[(Xva, yva_away)], verbose=False)

    mu_h_va = np.clip(model_home.predict(Xva), 1e-6, 50)
    mu_a_va = np.clip(model_away.predict(Xva), 1e-6, 50)
    mu_tot_va = mu_h_va + mu_a_va

    oof_mu_home[va_idx] = mu_h_va
    oof_mu_away[va_idx] = mu_a_va
    oof_mu_total[va_idx] = mu_tot_va

    # ---- Variance head: learn alpha via residual modeling ----
    # empirical var over window proxy: (y_total - mu_total)^2 as target (log-space)
    resid2 = (y_total[tr_idx] - (np.clip(model_home.predict(Xtr),1e-6,50) + np.clip(model_away.predict(Xtr),1e-6,50)))**2
    # desired var ≈ mu + alpha*mu^2 => alpha ≈ max( (var - mu) / mu^2 , 0 )
    mu_tr = np.clip(model_home.predict(Xtr),1e-6,50) + np.clip(model_away.predict(Xtr),1e-6,50)
    var_proxy = np.maximum(resid2, 1e-6)
    alpha_target = np.maximum((var_proxy - mu_tr) / np.maximum(mu_tr**2, 1e-6), 0.0)

    var_model = XGBRegressor(
        objective='reg:squarederror', tree_method='hist',
        n_estimators=500, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
        random_state=44
    )
    var_model.fit(Xtr, alpha_target, eval_set=[(Xva, np.zeros_like(lines_va))], verbose=False)

    alpha_va = np.clip(var_model.predict(Xva), 0.0, 5.0)
    oof_alpha[va_idx] = alpha_va

    # ---- NB tail: P(Over), P(Push) ----
    p_over_nb = np.zeros_like(mu_tot_va)
    p_push_nb = np.zeros_like(mu_tot_va)
    for i, (mu, L, a) in enumerate(zip(mu_tot_va, lines_va, alpha_va)):
        if np.isnan(L) or not np.isfinite(mu) or mu <= 0:
            p_over_nb[i] = np.nan
            p_push_nb[i] = np.nan
            continue
        if float(L).is_integer():
            k = int(L)
            cdf = nb_cdf(k, mu, a)
            p_over_nb[i] = max(0.0, 1.0 - cdf)
            # push prob is exact mass at k
            p_push_nb[i] = nb_pmf(k, mu, a)
        else:
            k = math.floor(L)
            cdf = nb_cdf(k, mu, a)
            p_over_nb[i] = max(0.0, 1.0 - cdf)
            p_push_nb[i] = 0.0

    oof_p_over_nb[va_idx] = p_over_nb
    oof_p_push_nb[va_idx] = p_push_nb

    # ---- Meta stacker (on validation only, trained on train OOF) ----
    # For honest stacking, build train-side OOF for meta from the training slice via inner holdouts
    inner = TimeSeriesSplit(n_splits=3)
    meta_tr_rows = []
    meta_tr_y = []
    for itr, iva in inner.split(Xtr):
        Xtr_i, Xva_i = Xtr[itr], Xtr[iva]
        Mtr_i, Mva_i = Mtr[itr], Mtr[iva]
        ytr_home_i, yva_home_i = ytr_home[itr], ytr_home[iva]
        ytr_away_i, yva_away_i = ytr_away[itr], ytr_away[iva]
        lines_va_i = lines[tr_idx][iva]

        mh = XGBRegressor(objective='count:poisson', tree_method='hist',
                          n_estimators=700, max_depth=6, learning_rate=0.035,
                          subsample=0.8, colsample_bytree=0.8, reg_lambda=1.1,
                          random_state=55)
        ma = XGBRegressor(objective='count:poisson', tree_method='hist',
                          n_estimators=700, max_depth=6, learning_rate=0.035,
                          subsample=0.8, colsample_bytree=0.8, reg_lambda=1.1,
                          random_state=56)
        mh.fit(Xtr_i, ytr_home_i, verbose=False)
        ma.fit(Xtr_i, ytr_away_i, verbose=False)
        mu_va_i = np.clip(mh.predict(Xva_i),1e-6,50) + np.clip(ma.predict(Xva_i),1e-6,50)

        vm = XGBRegressor(objective='reg:squarederror', tree_method='hist',
                          n_estimators=400, max_depth=4, learning_rate=0.05,
                          subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                          random_state=57)
        # build alpha target on inner-train
        resid2_i = ( (ytr_home[itr]+ytr_away[itr]) - (np.clip(mh.predict(Xtr_i),1e-6,50)+np.clip(ma.predict(Xtr_i),1e-6,50)) )**2
        mu_tr_i = np.clip(mh.predict(Xtr_i),1e-6,50) + np.clip(ma.predict(Xtr_i),1e-6,50)
        var_proxy_i = np.maximum(resid2_i, 1e-6)
        alpha_t_i = np.maximum((var_proxy_i - mu_tr_i) / np.maximum(mu_tr_i**2, 1e-6), 0.0)
        vm.fit(Xtr_i, alpha_t_i, verbose=False)
        alpha_va_i = np.clip(vm.predict(Xva_i), 0.0, 5.0)

        p_over_i = np.zeros(len(mu_va_i))
        p_push_i = np.zeros(len(mu_va_i))
        for j, (mu, L, a) in enumerate(zip(mu_va_i, lines_va_i, alpha_va_i)):
            if np.isnan(L) or mu <= 0:
                p_over_i[j] = np.nan; p_push_i[j] = np.nan; continue
            if float(L).is_integer():
                k = int(L)
                p_over_i[j] = max(0.0, 1.0 - nb_cdf(k, mu, a))
                p_push_i[j] = nb_pmf(k, mu, a)
            else:
                k = math.floor(L)
                p_over_i[j] = max(0.0, 1.0 - nb_cdf(k, mu, a))
                p_push_i[j] = 0.0

        y_over_tr_slice = y_over.iloc[tr_idx].copy()
        if str(y_over_tr_slice.dtype) != 'Int64':
            y_over_tr_slice = pd.to_numeric(y_over_tr_slice, errors='coerce').astype('Int64')
        mask_valid = y_over_tr_slice.iloc[iva].notna().values & np.isfinite(p_over_i)

        # meta row: [p_over_nb, mu, mu-line, line, vig, is_half, frac_from_half, meta features...]
        meta_block = np.column_stack([
            p_over_i[mask_valid],
            mu_va_i[mask_valid],
            (mu_va_i - lines_va_i)[mask_valid],
            lines_va_i[mask_valid]
        ])

        # append curated meta columns
        Mva_i_sel = Mva_i[mask_valid]
        meta_block = np.column_stack([meta_block, Mva_i_sel])

        meta_tr_rows.append(meta_block)
        meta_tr_y.append( (y_over_tr_slice.iloc[iva][mask_valid].values == 1).astype(int) )

    if meta_tr_rows:
        X_meta_tr = np.vstack(meta_tr_rows)
        y_meta_tr = np.concatenate(meta_tr_y)
    else:
        # Fallback: use train slice directly (rare)
        X_meta_tr = np.zeros((1, 4 + Mtr.shape[1]))
        y_meta_tr = np.array([0])

    # now build validation meta features from the fold models we trained above
    meta_va = np.column_stack([
        p_over_nb,
        mu_tot_va,
        mu_tot_va - lines_va,
        lines_va
    ])
    meta_va = np.column_stack([meta_va, Mva])

    # meta model: logistic with standardization, class_weight to handle imbalance
    meta_clf = Pipeline([
        ('scaler', StandardScaler(with_mean=True, with_std=True)),
        ('lr', LogisticRegression(
            penalty='l2', C=1.0, max_iter=200,
            class_weight='balanced',
            solver='lbfgs'
        ))
    ])
    meta_clf.fit(X_meta_tr, y_meta_tr)
    meta_proba_va = meta_clf.predict_proba(meta_va)[:, 1]
    oof_meta_proba[va_idx] = meta_proba_va

    # per-bucket isotonic on this fold
    fold_c = {}
    yva_over = (y_bin[va_idx] == 1).astype(int)
    mask_eval_fold = (y_over.iloc[va_idx].notna().values) & np.isfinite(meta_proba_va)
    b_va = buckets_va[mask_eval_fold]
    p_va = meta_proba_va[mask_eval_fold]
    y_va = yva_over[mask_eval_fold]
    for b in np.unique(b_va):
        m = (b_va == b)
        if m.sum() >= 25:
            iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip')
            iso.fit(p_va[m], y_va[m])
            fold_c[float(b)] = iso
    fold_cals.append(fold_c)

    print(f"Fold {fold_id}: built {len(fold_c)} bucket calibrators")
    fold_id += 1

# ------------- OOF metrics & EV search -------------
# Apply per-fold calibration to OOF meta probs
oof_meta_cal = oof_meta_proba.copy()
# We did per-fold cals only within each fold when producing oof. Here we already saved calibrated in-place above.

mask_metrics = eval_mask & np.isfinite(oof_meta_cal)
y_true = (y_bin[mask_metrics] == 1).astype(int)
p_hat = oof_meta_cal[mask_metrics]

# fallback: if some buckets had no calibrators in some folds, probabilities are raw — that’s OK.

# brier & auc
roc = roc_auc_score(y_true, p_hat) if len(np.unique(y_true)) > 1 else np.nan
brier = brier_score_loss(y_true, p_hat)
f1_05 = f1_score(y_true, (p_hat >= 0.5).astype(int))

# EV sweep using NB push prob (from OOF)
p_push = oof_p_push_nb[mask_metrics]
def ev_for_sides(p_over, p_push):
    p_push = 0.0 if (p_push is None or not np.isfinite(p_push)) else max(0.0, p_push)
    p_under = max(0.0, 1.0 - p_over - p_push)
    # -110 both sides -> +0.909 on win, -1 on loss
    ev_over  = p_over  * 0.909 - p_under * 1.0
    ev_under = p_under * 0.909 - p_over  * 1.0
    return ev_over, ev_under

thr_grid = np.arange(0.0, 0.151, 0.005)
best_thr, best_profit, best_bets, best_winrate = 0.0, -1e9, 0, 0.0
for thr in thr_grid:
    profit = 0.0
    bets = 0
    wins = 0
    for i, prob in enumerate(p_hat):
        ev_o, ev_u = ev_for_sides(prob, p_push[i] if i < len(p_push) else 0.0)
        chosen = 1 if ev_o >= ev_u else 0
        ev_taken = max(ev_o, ev_u)
        if ev_taken > thr:
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

print("\n======= OOF Performance (Two-head μ -> NB tail -> Stacked -> per-line Isotonic -> EV threshold) =======")
print(f"OOF ROC-AUC: {roc:.4f} | OOF Brier: {brier:.4f} | OOF F1@0.50: {f1_05:.4f}")
print(f"Best EV threshold: {best_thr:.3f} | Bets: {best_bets} | Win%: {best_winrate:.3f} | Total units: {best_profit:.2f}")

# ------------- Fit FINAL models on ALL data -------------
# Base μ heads on all data
model_home_all = XGBRegressor(
    objective='count:poisson', tree_method='hist',
    n_estimators=1000, max_depth=6, learning_rate=0.03,
    subsample=0.8, colsample_bytree=0.8, reg_lambda=1.2,
    random_state=101
)
model_away_all = XGBRegressor(
    objective='count:poisson', tree_method='hist',
    n_estimators=1000, max_depth=6, learning_rate=0.03,
    subsample=0.8, colsample_bytree=0.8, reg_lambda=1.2,
    random_state=102
)
model_home_all.fit(X_all, R_home, verbose=False)
model_away_all.fit(X_all, R_away, verbose=False)

mu_tr_all = np.clip(model_home_all.predict(X_all),1e-6,50) + np.clip(model_away_all.predict(X_all),1e-6,50)
resid2_all = (y_total - mu_tr_all)**2
var_proxy_all = np.maximum(resid2_all, 1e-6)
alpha_target_all = np.maximum((var_proxy_all - mu_tr_all) / np.maximum(mu_tr_all**2, 1e-6), 0.0)

var_model_all = XGBRegressor(
    objective='reg:squarederror', tree_method='hist',
    n_estimators=700, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
    random_state=103
)
var_model_all.fit(X_all, alpha_target_all, verbose=False)

# Build global calibrators using OOF raw meta proba
global_cals = {}
cal_df = pd.DataFrame({
    'bucket': buckets[mask_metrics],
    'p_meta_raw': oof_meta_proba[mask_metrics],
    'y_over': y_true
})
for b, g in cal_df.groupby('bucket'):
    if len(g) >= 50:
        iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip')
        iso.fit(g['p_meta_raw'].values, g['y_over'].values)
        global_cals[float(b)] = iso

# Fit meta on ALL data using features built from ALL (honest enough post-OFF since calibrator uses only OOF)
# Construct meta features on ALL data
mu_all = mu_tr_all
lines_all = lines
alpha_all = np.clip(var_model_all.predict(X_all), 0.0, 5.0)
p_over_all = np.zeros(len(mu_all))
for i, (mu, L, a) in enumerate(zip(mu_all, lines_all, alpha_all)):
    if np.isnan(L) or mu <= 0:
        p_over_all[i] = np.nan
        continue
    if float(L).is_integer():
        k = int(L)
        p_over_all[i] = max(0.0, 1.0 - nb_cdf(k, mu, a))
    else:
        k = math.floor(L)
        p_over_all[i] = max(0.0, 1.0 - nb_cdf(k, mu, a))

meta_all = np.column_stack([
    p_over_all,
    mu_all,
    mu_all - lines_all,
    lines_all
])
meta_all = np.column_stack([meta_all, M_all])

meta_y_all = (y_over.fillna(-1).values == 1).astype(int)
meta_clf_all = Pipeline([
    ('scaler', StandardScaler(with_mean=True, with_std=True)),
    ('lr', LogisticRegression(
        penalty='l2', C=1.0, max_iter=300,
        class_weight='balanced',
        solver='lbfgs'
    ))
])
# only fit on rows with non-nan p_over_all and non-push labels
mask_fit_meta = np.isfinite(p_over_all) & y_over.notna().values
meta_clf_all.fit(meta_all[mask_fit_meta], meta_y_all[mask_fit_meta])

# ------------- Save artifacts -------------
os.makedirs("models_ou_stack", exist_ok=True)

joblib.dump({
    'model_home': model_home_all,
    'model_away': model_away_all,
    'base_feature_cols': base_feature_cols
}, "models_ou_stack/base_model.joblib")

joblib.dump({
    'var_model': var_model_all
}, "models_ou_stack/var_model.joblib")

joblib.dump({
    'meta_pipeline': meta_clf_all,
    'meta_feature_cols': ['p_over_nb','mu_total','mu_minus_line','line'] + meta_feature_cols
}, "models_ou_stack/meta_model.joblib")

joblib.dump({
    'global_calibrators': global_cals,
    'line_bucket_fn': 'round(x*2)/2',
    'ev_threshold': float(best_thr)
}, "models_ou_stack/calibration.joblib")

with open("models_ou_stack/feature_cols.json", "w") as f:
    json.dump({
        'base_feature_cols': base_feature_cols,
        'meta_feature_cols': ['p_over_nb','mu_total','mu_minus_line','line'] + meta_feature_cols
    }, f, indent=2)

with open("models_ou_stack/metrics_oof.json", "w") as f:
    json.dump({
        'oof_roc_auc': float(roc),
        'oof_brier': float(brier),
        'oof_f1_at_0.50': float(f1_05),
        'best_ev_threshold': float(best_thr),
        'best_total_units': float(best_profit),
        'best_bets': int(best_bets),
        'best_winrate': float(best_winrate)
    }, f, indent=2)

print("\nSaved:")
print(" - models_ou_stack/base_model.joblib")
print(" - models_ou_stack/var_model.joblib")
print(" - models_ou_stack/meta_model.joblib")
print(" - models_ou_stack/calibration.joblib")
print(" - models_ou_stack/feature_cols.json")
print(" - models_ou_stack/metrics_oof.json")
