#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MLB O/U: Poisson μ (XGBRegressor) + Direct Classifier (XGBClassifier) -> blend -> per-line Isotonic -> price-aware EV sweep

- Leak-safe (time-ordered CV; no post-game fields as features).
- Two heads:
    (A) Poisson mean μ -> Poisson tail vs total_line -> p_over_μ
    (B) Direct classifier on over_result (push rows dropped within each fold) with price-aware weights
- Blend p_over = w * p_over_clf + (1-w) * p_over_μ (w tuned by AUC)
- Per-total-line isotonic calibration (+ global fallback)
- Price-aware EV sweep with MIN_BETS constraint
"""

import os, math, json
import numpy as np
import pandas as pd
import joblib
import optuna
from typing import Tuple, Dict

from sklearn.model_selection import TimeSeriesSplit
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, brier_score_loss, f1_score
from xgboost import XGBRegressor, XGBClassifier

# ----------------------------- vectorized odds helpers -----------------------------
def american_to_prob(odds):
    """Accepts scalar or array-like American odds, returns implied probability (no vig removal)."""
    arr = np.asarray(odds, dtype=float)
    pos = arr > 0
    out = np.empty_like(arr, dtype=float)
    out[pos]  = 100.0 / (arr[pos] + 100.0)
    out[~pos] = (-arr[~pos]) / ((-arr[~pos]) + 100.0)
    # if called with scalar, return scalar
    return out.item() if out.ndim == 0 else out

def american_payout_per_unit(odds):
    """Accepts scalar or array-like American odds, returns profit per 1 unit stake on a win."""
    arr = np.asarray(odds, dtype=float)
    pos = arr > 0
    out = np.empty_like(arr, dtype=float)
    out[pos]  = arr[pos] / 100.0       # +150 -> +1.5 per unit
    out[~pos] = 100.0 / (-arr[~pos])   # -120 -> +0.8333 per unit
    return out.item() if out.ndim == 0 else out

# ----------------------------- poisson helpers -----------------------------
def poisson_cdf(k: int, mu: float) -> float:
    if mu <= 0:
        return 1.0 if k >= 0 else 0.0
    term = math.exp(-mu)
    s = term
    up_to = max(0, k)
    for n in range(1, up_to + 1):
        term *= mu / n
        s += term
    return min(1.0, max(0.0, s))

def poisson_sf(k: int, mu: float) -> float:
    return max(0.0, 1.0 - poisson_cdf(k, mu))

def p_over_from_mu_and_line(mu: np.ndarray, line: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    p_over = np.zeros_like(mu, dtype=float)
    p_push = np.zeros_like(mu, dtype=float)
    for i, (m, L) in enumerate(zip(mu, line)):
        if pd.isna(L) or not np.isfinite(m) or m <= 0:
            p_over[i] = np.nan; p_push[i] = np.nan; continue
        if float(L).is_integer():
            k = int(L)
            p_over[i] = poisson_sf(k, m)
            try:
                lg = -m + k * math.log(m) - math.lgamma(k + 1)
                p_push[i] = math.exp(lg)
            except ValueError:
                p_push[i] = 0.0
        else:
            k = math.floor(L)
            p_over[i] = poisson_sf(k, m)
            p_push[i] = 0.0
    return p_over, p_push

def ev_for_sides(p_over: float, p_push: float, price_over: float | np.ndarray | None, price_under: float | np.ndarray | None) -> Tuple[float, float]:
    if price_over is None or (isinstance(price_over, float) and not np.isfinite(price_over)):
        price_over = -110.0
    if price_under is None or (isinstance(price_under, float) and not np.isfinite(price_under)):
        price_under = -110.0
    p_over = float(np.clip(p_over, 0.0, 1.0))
    p_push = 0.0 if (p_push is None or not np.isfinite(p_push)) else max(0.0, float(p_push))
    p_under = max(0.0, 1.0 - p_over - p_push)
    pay_o = american_payout_per_unit(price_over)
    pay_u = american_payout_per_unit(price_under)
    ev_o  = p_over  * float(pay_o) - p_under * 1.0
    ev_u  = p_under * float(pay_u) - p_over  * 1.0
    return ev_o, ev_u

def line_bucket(x: float) -> float:
    if pd.isna(x): return np.nan
    return round(float(x) * 2.0) / 2.0

# ----------------------------- data -----------------------------
DATA_PATH = "jordan_final.csv"
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError("jordan_final.csv not found.")

df = pd.read_csv(DATA_PATH)

required = {'R_home','R_away','total_line','over_result'}
miss = required - set(df.columns)
if miss:
    raise ValueError(f"Missing required columns: {miss}")

df = df.copy()
df['total_line'] = pd.to_numeric(df['total_line'], errors='coerce')
mask_ok = df['total_line'].notna() & pd.to_numeric(df['R_home'], errors='coerce').notna() & pd.to_numeric(df['R_away'], errors='coerce').notna()
df = df.loc[mask_ok].reset_index(drop=True)

# Order by date for leak safety
if 'Date_Parsed' in df.columns:
    order = pd.to_datetime(df['Date_Parsed'], errors='coerce').fillna(pd.Timestamp('1900-01-01'))
    df = df.iloc[np.argsort(order.values)].reset_index(drop=True)

# Labels
y_over_ser = df['over_result']
if str(y_over_ser.dtype) != 'Int64':
    y_over_ser = pd.to_numeric(y_over_ser, errors='coerce').astype('Int64')
y_eval_mask = y_over_ser.notna().values
y_bin_all = y_over_ser.fillna(-1).astype(int).values

R_home = pd.to_numeric(df['R_home'], errors='coerce')
R_away = pd.to_numeric(df['R_away'], errors='coerce')
y_total = (R_home + R_away).values

price_over_full  = pd.to_numeric(df.get('price_over'),  errors='coerce') if 'price_over'  in df.columns else pd.Series(np.nan, index=df.index)
price_under_full = pd.to_numeric(df.get('price_under'), errors='coerce') if 'price_under' in df.columns else pd.Series(np.nan, index=df.index)

# features
def pick_features(d: pd.DataFrame) -> list:
    drop_exact = {'over_result','over_under_label','R_home','R_away'}
    drop_starts = ('Boxscore','merge_key')
    drop_contains = ('_label','actual','result')
    keep = []
    for c in d.columns:
        if c in drop_exact: continue
        if any(c.startswith(s) for s in drop_starts): continue
        if any(tok in c for tok in drop_contains): continue
        if np.issubdtype(d[c].dtype, np.number):
            keep.append(c)
    if 'total_line' not in keep and 'total_line' in d.columns and np.issubdtype(d['total_line'].dtype, np.number):
        keep.append('total_line')
    return keep

feature_cols = pick_features(df)
X_all = df[feature_cols].astype(float).fillna(0.0).values
lines_all = df['total_line'].values
buckets_all = np.array([line_bucket(x) for x in lines_all])

# ----------------------------- tune Poisson μ -----------------------------
def tune_mu(X, y, lines, y_over_series) -> dict:
    tscv = TimeSeriesSplit(n_splits=5)
    def objective(trial):
        params = {
            'objective': 'count:poisson',
            'tree_method': 'hist',
            'n_estimators': trial.suggest_int('n_estimators', 500, 1400),
            'max_depth': trial.suggest_int('max_depth', 3, 7),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.07, log=True),
            'subsample': trial.suggest_float('subsample', 0.7, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 5.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 5.0, log=True),
            'random_state': 42
        }
        aucs = []
        for tr, va in tscv.split(X):
            Xtr, Xva = X[tr], X[va]
            ytr, Lva = y[tr], lines[va]
            y_over_va = y_over_series.iloc[va]
            mask_va = y_over_va.notna().values
            yb = y_over_va.fillna(-1).astype(int).values
            model = XGBRegressor(**params)
            model.fit(Xtr, ytr, verbose=False)
            mu_va = np.clip(model.predict(Xva), 1e-6, 50)
            p_mu, _ = p_over_from_mu_and_line(mu_va, Lva)
            m = mask_va & np.isfinite(p_mu)
            if m.sum() == 0: 
                continue
            y_true = (yb[m] == 1).astype(int)
            if np.unique(y_true).size > 1:
                aucs.append(roc_auc_score(y_true, p_mu[m]))
        return float(np.mean(aucs)) if aucs else 0.0
    print("🔎 Optuna tuning μ (mean) ...")
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50)
    best = study.best_params
    print(f"✅ Best μ params: {best}")
    return best

best_mu = tune_mu(X_all, y_total, lines_all, y_over_ser)

# ----------------------------- tune classifier (price-aware) -----------------------------
def tune_clf(X, y_over_series, price_over_s, price_under_s) -> dict:
    tscv = TimeSeriesSplit(n_splits=5)
    def objective(trial):
        params = {
            'objective': 'binary:logistic',
            'tree_method': 'hist',
            'n_estimators': trial.suggest_int('n_estimators', 400, 1200),
            'max_depth': trial.suggest_int('max_depth', 3, 7),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.07, log=True),
            'subsample': trial.suggest_float('subsample', 0.7, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 5.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 5.0, log=True),
            'random_state': 42
        }
        aucs = []
        for tr, va in tscv.split(X):
            y_va = y_over_series.iloc[va]
            keep_va = y_va.notna()
            if keep_va.sum() < 50:
                continue
            y_tr = y_over_series.iloc[tr]
            keep_tr = y_tr.notna()

            Xtr = X[tr][keep_tr.values]
            ytr = (y_tr[keep_tr].astype(int).values == 1).astype(int)
            Xva = X[va][keep_va.values]
            yva = (y_va[keep_va].astype(int).values == 1).astype(int)

            po_tr = price_over_s.iloc[tr][keep_tr].astype(float).values
            pu_tr = price_under_s.iloc[tr][keep_tr].astype(float).values
            pay_o = np.where(np.isfinite(po_tr), american_payout_per_unit(po_tr), american_payout_per_unit(-110.0))
            pay_u = np.where(np.isfinite(pu_tr), american_payout_per_unit(pu_tr), american_payout_per_unit(-110.0))
            w = np.where(ytr == 1, pay_o, pay_u)
            w = np.clip(w, 0.5, 2.5)

            clf = XGBClassifier(**params)
            clf.fit(Xtr, ytr, sample_weight=w, verbose=False)
            p = clf.predict_proba(Xva)[:, 1]
            if np.unique(yva).size > 1:
                aucs.append(roc_auc_score(yva, p))
        return float(np.mean(aucs)) if aucs else 0.0
    print("🔎 Optuna tuning classifier ...")
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=40)
    best = study.best_params
    print(f"✅ Best clf params: {best}")
    return best

best_clf = tune_clf(X_all, y_over_ser, price_over_full, price_under_full)

# ----------------------------- OOF preds for both heads -----------------------------
tscv = TimeSeriesSplit(n_splits=5)
oof_mu = np.full(len(df), np.nan)
oof_push = np.full(len(df), np.nan)
oof_clf = np.full(len(df), np.nan)

for tr, va in tscv.split(X_all):
    # μ head
    mu_model = XGBRegressor(objective='count:poisson', tree_method='hist', random_state=42, **best_mu)
    mu_model.fit(X_all[tr], y_total[tr], verbose=False)
    mu_va = np.clip(mu_model.predict(X_all[va]), 1e-6, 50)
    p_mu, p_psh = p_over_from_mu_and_line(mu_va, lines_all[va])
    oof_mu[va] = p_mu
    oof_push[va] = p_psh

    # classifier head (drop pushes)
    y_tr = y_over_ser.iloc[tr]; y_va = y_over_ser.iloc[va]
    keep_tr = y_tr.notna(); keep_va = y_va.notna()
    if keep_tr.sum() >= 100 and keep_va.sum() >= 50:
        Xtr = X_all[tr][keep_tr.values]
        ytr = (y_tr[keep_tr].astype(int).values == 1).astype(int)
        Xva = X_all[va][keep_va.values]
        po_tr = price_over_full.iloc[tr][keep_tr].astype(float).values
        pu_tr = price_under_full.iloc[tr][keep_tr].astype(float).values
        pay_o = np.where(np.isfinite(po_tr), american_payout_per_unit(po_tr), american_payout_per_unit(-110.0))
        pay_u = np.where(np.isfinite(pu_tr), american_payout_per_unit(pu_tr), american_payout_per_unit(-110.0))
        sw = np.where(ytr == 1, pay_o, pay_u)
        sw = np.clip(sw, 0.5, 2.5)

        clf = XGBClassifier(objective='binary:logistic', tree_method='hist', random_state=42, **best_clf)
        clf.fit(Xtr, ytr, sample_weight=sw, verbose=False)
        p_clf = np.full(len(va), np.nan, dtype=float)
        p_clf[keep_va.values] = clf.predict_proba(Xva)[:, 1]
        oof_clf[va] = p_clf

# ----------------------------- blend tuning -----------------------------
mask_eval = y_eval_mask & np.isfinite(oof_mu)
y_true = (y_bin_all[mask_eval] == 1).astype(int)
p_mu_eval = oof_mu[mask_eval]
p_clf_eval = np.where(np.isfinite(oof_clf[mask_eval]), oof_clf[mask_eval], p_mu_eval)
p_push_eval = np.where(np.isfinite(oof_push[mask_eval]), oof_push[mask_eval], 0.0)

ws = np.linspace(0.0, 1.0, 21)
best_w, best_auc, best_brier = 0.5, -1.0, 1.0
for w in ws:
    p_blend = w * p_clf_eval + (1 - w) * p_mu_eval
    if np.unique(y_true).size > 1:
        auc = roc_auc_score(y_true, p_blend)
        br  = brier_score_loss(y_true, p_blend)
        if (auc > best_auc) or (abs(auc - best_auc) < 1e-6 and br < best_brier):
            best_auc, best_brier, best_w = auc, br, w

print(f"\nBlending weight w={best_w:.2f} -> AUC {best_auc:.4f}, Brier {best_brier:.4f}")

p_blend_oof = np.full(len(df), np.nan)
p_blend_oof[mask_eval] = best_w * p_clf_eval + (1 - best_w) * p_mu_eval

# ----------------------------- per-line isotonic on blended OOF -----------------------------
buckets_eval = np.array([line_bucket(x) for x in df['total_line'].values])[mask_eval]
cals_per_bucket: Dict[float, IsotonicRegression] = {}
for b in np.unique(buckets_eval):
    idx = (buckets_eval == b)
    if idx.sum() >= 50:
        iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip')
        iso.fit(p_blend_oof[mask_eval][idx], y_true[idx])
        cals_per_bucket[float(b)] = iso

iso_global = None
if np.isfinite(p_blend_oof[mask_eval]).sum() >= 200:
    iso_global = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip')
    iso_global.fit(p_blend_oof[mask_eval], y_true)

# apply calibration to full OOF
p_cal_oof = p_blend_oof.copy()
buckets_all_vals = np.array([line_bucket(x) for x in df['total_line'].values])
for b, iso in cals_per_bucket.items():
    idx_all = (buckets_all_vals == b) & np.isfinite(p_cal_oof)
    if idx_all.any():
        p_cal_oof[idx_all] = iso.predict(p_cal_oof[idx_all])
# fallback (only if there are still finite yet uncalibrated points)
if iso_global is not None:
    idx_missing = np.isfinite(p_cal_oof) & np.isnan(p_cal_oof)
    if idx_missing.any():
        p_cal_oof[idx_missing] = iso_global.predict(p_cal_oof[idx_missing])

# ----------------------------- metrics & EV sweep -----------------------------
mask_eval2 = y_eval_mask & np.isfinite(p_cal_oof)
y_true2 = (y_bin_all[mask_eval2] == 1).astype(int)
p_cal2 = p_cal_oof[mask_eval2]
p_push2 = np.where(np.isfinite(oof_push[mask_eval2]), oof_push[mask_eval2], 0.0)

po_m = price_over_full.values[mask_eval2]
pu_m = price_under_full.values[mask_eval2]

roc = roc_auc_score(y_true2, p_cal2) if np.unique(y_true2).size > 1 else float('nan')
brier = brier_score_loss(y_true2, p_cal2)
f1_05 = f1_score(y_true2, (p_cal2 >= 0.5).astype(int))
print(f"\nOOF base metrics -> ROC-AUC: {roc:.4f} | Brier: {brier:.4f} | F1@0.50: {f1_05:.4f}")

EV_GRID = np.arange(0.00, 0.151, 0.005)
MIN_BETS = 1500

best_thr, best_profit, best_bets, best_win = 0.0, -1e9, 0, 0
for thr in EV_GRID:
    profit = 0.0; bets = 0; wins = 0
    for i in range(len(p_cal2)):
        ev_o, ev_u = ev_for_sides(p_cal2[i], p_push2[i], po_m[i], pu_m[i])
        chosen = 1 if ev_o >= ev_u else 0
        ev = ev_o if chosen == 1 else ev_u
        if ev > thr:
            bets += 1
            if y_true2[i] == chosen:
                payout = american_payout_per_unit(po_m[i]) if chosen == 1 else american_payout_per_unit(pu_m[i])
                profit += float(payout)
                wins += 1
            else:
                profit -= 1.0
    if bets >= MIN_BETS and profit > best_profit:
        best_thr, best_profit, best_bets, best_win = float(thr), profit, bets, wins

winrate = (best_win / best_bets) if best_bets else 0.0
print("\n======= OOF Performance (Ensembled -> per-line Isotonic -> EV threshold, price-aware) =======")
print(f"OOF ROC-AUC: {roc:.4f} | Brier: {brier:.4f} | F1@0.50: {f1_05:.4f}")
print(f"Best EV threshold: {best_thr:.3f} | Bets: {best_bets} | Win%: {winrate:.3f} | Units: {best_profit:.2f}")

# ----------------------------- fit final models on ALL data -----------------------------
print("\nTraining final models on all data...")
final_mu = XGBRegressor(objective='count:poisson', tree_method='hist', random_state=42, **best_mu)
final_mu.fit(X_all, y_total, verbose=False)

keep_all = y_over_ser.notna()
X_cls = X_all[keep_all.values]
y_cls = (y_over_ser[keep_all].astype(int).values == 1).astype(int)
po_all = price_over_full[keep_all].astype(float).values
pu_all = price_under_full[keep_all].astype(float).values
sw_all = np.where(y_cls == 1,
                  np.where(np.isfinite(po_all), american_payout_per_unit(po_all), american_payout_per_unit(-110.0)),
                  np.where(np.isfinite(pu_all), american_payout_per_unit(pu_all), american_payout_per_unit(-110.0)))
sw_all = np.clip(sw_all, 0.5, 2.5)
final_clf = XGBClassifier(objective='binary:logistic', tree_method='hist', random_state=42, **best_clf)
final_clf.fit(X_cls, y_cls, sample_weight=sw_all, verbose=False)

# save calibrators built from OOF
global_bucket_isos: Dict[float, IsotonicRegression] = {}
for b, iso in {}.items():  # (placeholder to emphasize dict type)
    pass
# reuse cals_per_bucket / iso_global computed on OOF
global_bucket_isos = {float(b): iso for b, iso in cals_per_bucket.items()}
global_fallback_iso = iso_global

# ----------------------------- save -----------------------------
os.makedirs("models_xgb_poisson", exist_ok=True)

joblib.dump({
    'mu_model': final_mu,
    'clf_model': final_clf,
    'feature_cols': feature_cols,
    'blend_w': float(best_w)
}, "models_xgb_poisson/model.joblib")

joblib.dump({
    'bucket_isos': global_bucket_isos,
    'global_iso': global_fallback_iso,
    'line_bucket_fn': 'round(x*2)/2',
    'ev_threshold': float(best_thr),
    'best_mu_params': best_mu,
    'best_clf_params': best_clf
}, "models_xgb_poisson/calibration.joblib")

with open("models_xgb_poisson/metrics_oof.json","w") as f:
    json.dump({
        'oof_roc_auc': float(roc),
        'oof_brier': float(brier),
        'oof_f1_at_0_50': float(f1_05),
        'best_ev_threshold': float(best_thr),
        'best_total_units': float(best_profit),
        'best_bets': int(best_bets),
        'best_winrate': float(winrate),
        'blend_w': float(best_w),
        'n_features': int(len(feature_cols))
    }, f, indent=2)

print("\nSaved:")
print(" - models_xgb_poisson/model.joblib")
print(" - models_xgb_poisson/calibration.joblib")
print(" - models_xgb_poisson/metrics_oof.json")
