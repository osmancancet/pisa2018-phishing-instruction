"""A2: within-country linear probability models of 'click' on 'taught'.

Same complete-case sample for all models. Country fixed effects. Senate-scaled
weights for the pooled model (each system counts equally); final student weight
for single-country models. BRR (Fay 0.5, 80 replicates) for sampling variance,
Rubin's rules across 10 reading plausible values where reading enters.
"""
import json
import sys
import numpy as np
import pandas as pd
from common import load, brr_var, rubin, senate_scale, REP_W, PV_READ

OTHER = [f'ST158Q0{i}_y' for i in range(1, 7)]
BASE = ['female', 'ESCS', 'imm1', 'imm2', 'imm_miss', 'GRADE', 'AGE', 'effort', 'effort_miss']

d = load()
d['effort_miss'] = d['EFFORT1'].isna().astype(float)
d['effort'] = d['EFFORT1'].fillna(d.groupby('CNT')['EFFORT1'].transform('mean'))
d['imm_miss'] = d['IMMIG'].isna().astype(float)  # e.g. Japan did not report immigrant status
d['imm1'] = (d['IMMIG'] == 2).astype(float)
d['imm2'] = (d['IMMIG'] == 3).astype(float)
d['safe_mean'] = d[['ST166Q02HA', 'ST166Q04HA', 'ST166Q05HA']].mean(axis=1, skipna=False)
d['click_ips'] = d['ST166Q03HA'] - d[[f'ST166Q0{i}HA' for i in range(1, 6)]].mean(axis=1, skipna=False)
d['inconsistent'] = ((d['ST166Q03HA'] >= 4) & (d['ST166Q04HA'] >= 4)).astype(float)
need = ['taught', 'click', 'safe_mean', 'PV1READ', 'female', 'ESCS', 'GRADE', 'AGE', 'effort'] + OTHER
d = d.dropna(subset=need).reset_index(drop=True)
for c in PV_READ:
    d[c] = d[c] / 100.0  # reading in 100-point units


def wls(X, y, w):
    Xw = X * w[:, None]
    beta = np.linalg.solve(X.T @ Xw, Xw.T @ y)
    return beta[0]  # coefficient on 'taught' (first column)


def design(g, cols, fe):
    X = g[['taught'] + cols].to_numpy(float)
    if fe:
        return X, pd.factorize(g['CNT'])[0]
    return np.hstack([X, np.ones((len(g), 1))]), None


def wls_fe(X, grp, y, w):
    """WLS with country fixed effects via weighted within-country demeaning."""
    sw = np.bincount(grp, weights=w)
    Xd = X - (np.vstack([np.bincount(grp, weights=w * X[:, j]) for j in range(X.shape[1])]).T / sw[:, None])[grp]
    yd = y - (np.bincount(grp, weights=w * y) / sw)[grp]
    # drop columns constant within all countries (e.g. imm_miss when only one system lacks it)
    keep = np.sqrt(np.average(Xd ** 2, axis=0, weights=w)) > 1e-10
    return wls(Xd[:, keep], yd, w)


def estimate(g, cols, pooled, uses_pv, yvar='click'):
    g = g[g[yvar].notna()].reset_index(drop=True)
    y = g[yvar].to_numpy(float)
    if pooled:
        wfull = senate_scale(g, 'W_FSTUWT').to_numpy()
        wreps = [senate_scale(g, w).to_numpy() for w in REP_W]
    else:
        wfull = g['W_FSTUWT'].to_numpy()
        wreps = [g[w].to_numpy() for w in REP_W]
    pvs = PV_READ if uses_pv else [None]
    ests, vars_ = [], []
    for pv in pvs:
        cc = cols + ([pv] if pv else [])
        X, grp = design(g, cc, fe=pooled)
        if pooled:
            b = wls_fe(X, grp, y, wfull)
            reps = [wls_fe(X, grp, y, w) for w in wreps]
        else:
            keep = np.std(X, axis=0) > 0
            keep[-1] = True  # intercept
            X = X[:, keep]
            b = wls(X, y, wfull)
            reps = [wls(X, y, w) for w in wreps]
        ests.append(b)
        vars_.append(brr_var(b, reps))
    est, se = rubin(ests, vars_)
    return {'b': float(est), 'se': float(se), 'n': int(len(g))}


MODELS = {
    'M0_none': ([], False, 'click'),
    'M1_background': (BASE, False, 'click'),
    'M2_plus_reading': (BASE, True, 'click'),
    'M3_plus_other_instruction': (BASE + OTHER, True, 'click'),
    'M4_plus_response_style': (BASE + OTHER + ['safe_mean'], True, 'click'),
    'M5_ipsative_outcome': (BASE + OTHER, True, 'click_ips'),
}

which = sys.argv[1] if len(sys.argv) > 1 else 'pooled'
res = {}
if which == 'extra':
    cols, pv, y = MODELS['M5_ipsative_outcome']
    res['M5_ipsative_outcome'] = estimate(d, cols, pooled=True, uses_pv=pv, yvar=y)
    t = d[d.CNT == 'TUR'].reset_index(drop=True)
    res['TUR_M5_ipsative_outcome'] = estimate(t, cols, pooled=False, uses_pv=pv, yvar=y)
    tc = t[t['inconsistent'] == 0].reset_index(drop=True)
    for name in ('M3_plus_other_instruction', 'M4_plus_response_style'):
        c2, pv2, y2 = MODELS[name]
        res['TUR_R1_consistent_' + name] = estimate(tc, c2, pooled=False, uses_pv=pv2, yvar=y2)
    res['inconsistent_share_all'] = float(d['inconsistent'].mean())
    res['inconsistent_share_TUR'] = float(t['inconsistent'].mean())
    print(res, flush=True)
    json.dump(res, open('results/a2_lpm_extra.json', 'w'), indent=1)
elif which == 'pooled':
    for name, (cols, pv, y) in MODELS.items():
        res[name] = estimate(d, cols, pooled=True, uses_pv=pv, yvar=y)
        print(name, res[name], flush=True)
    # robustness: drop internally inconsistent responders (click and delete both rated appropriate)
    dc = d[d['inconsistent'] == 0].reset_index(drop=True)
    for name in ('M3_plus_other_instruction', 'M4_plus_response_style'):
        cols, pv, y = MODELS[name]
        res['R1_consistent_' + name] = estimate(dc, cols, pooled=True, uses_pv=pv, yvar=y)
        print('R1', name, res['R1_consistent_' + name], flush=True)
    # robustness: OECD members only (Jerrim's population)
    do = d[d['OECD'] == 1].reset_index(drop=True)
    for name in ('M0_none', 'M3_plus_other_instruction', 'M4_plus_response_style'):
        cols, pv, y = MODELS[name]
        res['R2_oecd_' + name] = estimate(do, cols, pooled=True, uses_pv=pv, yvar=y)
        print('R2', name, res['R2_oecd_' + name], flush=True)
    json.dump(res, open('results/a2_lpm_pooled.json', 'w'), indent=1)
else:
    for cnt, g in d.groupby('CNT'):
        g = g.reset_index(drop=True)
        res[cnt] = {name: estimate(g, cols, pooled=False, uses_pv=pv, yvar=y)
                    for name, (cols, pv, y) in MODELS.items()
                    if name in ('M0_none', 'M3_plus_other_instruction', 'M4_plus_response_style')}
        print(cnt, res[cnt]['M4_plus_response_style'], flush=True)
    json.dump(res, open('results/a2_lpm_bycountry.json', 'w'), indent=1)
