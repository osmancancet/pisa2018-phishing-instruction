"""A6: item-level diagnostics and Türkiye moderation models.

(1) Weighted mean rating (1-6) of each of the five ST166 response options by
    phishing instruction, pooled (senate weights) and for Türkiye, with BRR SEs.
(2) Türkiye: LPM with taught x moderator interactions (reading, ESCS, female),
    M4 covariates, BRR + Rubin over 10 PVs.
(3) Random-effects meta-analysis (DerSimonian-Laird) of the 76 per-system M4
    estimates and Benjamini-Hochberg adjusted p-values.
"""
import json
import numpy as np
import pandas as pd
from scipy import stats
from common import load, brr_var, rubin, senate_scale, REP_W, PV_READ

ITEMS = {f'ST166Q0{i}HA': lab for i, lab in enumerate(
    ['answer_email', 'check_sender', 'click_link', 'delete_email', 'check_operator_site'], start=1)}
d = load()
res = {'items': {}}
for scope, g in (('pooled', d), ('TUR', d[d.CNT == 'TUR'])):
    g = g.reset_index(drop=True)
    res['items'][scope] = {}
    for it, lab in ITEMS.items():
        h = g[g[it].notna()]
        if scope == 'pooled':
            wf = senate_scale(h, 'W_FSTUWT').to_numpy(); wr = [senate_scale(h, w).to_numpy() for w in REP_W]
        else:
            wf = h['W_FSTUWT'].to_numpy(); wr = [h[w].to_numpy() for w in REP_W]
        t = h['taught'].to_numpy(); y = h[it].to_numpy()

        def diff(w):
            return np.average(y[t == 1], weights=w[t == 1]) - np.average(y[t == 0], weights=w[t == 0])
        full = diff(wf)
        se = np.sqrt(brr_var(full, [diff(w) for w in wr]))
        res['items'][scope][lab] = {'mean_taught': float(np.average(y[t == 1], weights=wf[t == 1])),
                                    'mean_not': float(np.average(y[t == 0], weights=wf[t == 0])),
                                    'diff': float(full), 'se': float(se)}
    print(scope, {k: round(v['diff'], 3) for k, v in res['items'][scope].items()}, flush=True)

# (2) Türkiye moderation
OTHER = [f'ST158Q0{i}_y' for i in range(1, 7)]
t = d[d.CNT == 'TUR'].copy()
t['effort_miss'] = t['EFFORT1'].isna().astype(float)
t['effort'] = t['EFFORT1'].fillna(t['EFFORT1'].mean())
t['imm1'] = (t['IMMIG'] == 2).astype(float); t['imm2'] = (t['IMMIG'] == 3).astype(float)
t['safe_mean'] = t[['ST166Q02HA', 'ST166Q04HA', 'ST166Q05HA']].mean(axis=1, skipna=False)
t = t.dropna(subset=['female', 'ESCS', 'IMMIG', 'GRADE', 'AGE', 'safe_mean'] + OTHER).reset_index(drop=True)
BASE = ['female', 'ESCS', 'imm1', 'imm2', 'GRADE', 'AGE', 'effort', 'effort_miss'] + OTHER + ['safe_mean']
y = t['click'].to_numpy(float)
mods = {}
for mod in ('read', 'ESCS', 'female'):
    ests, vars_ = [], []
    for pv in PV_READ:
        r = (t[pv] / 100.0).to_numpy()
        r_c = r - np.average(r, weights=t['W_FSTUWT'])
        m = {'read': r_c, 'ESCS': (t['ESCS'] - np.average(t['ESCS'], weights=t['W_FSTUWT'])).to_numpy(),
             'female': t['female'].to_numpy() - 0.5}[mod]
        X = np.column_stack([t['taught'], t['taught'] * m, r_c, t[BASE].to_numpy(float), np.ones(len(t))])

        def fit(w):
            Xw = X * w[:, None]
            return np.linalg.solve(X.T @ Xw, Xw.T @ y)[:2]
        b = fit(t['W_FSTUWT'].to_numpy())
        reps = np.array([fit(t[w].to_numpy()) for w in REP_W])
        ests.append(b); vars_.append(brr_var(b, reps))
    est, se = rubin(ests, vars_)
    mods[mod] = {'taught': float(est[0]), 'taught_se': float(se[0]), 'interaction': float(est[1]), 'interaction_se': float(se[1])}
    print('TUR moderation', mod, mods[mod], flush=True)
res['TUR_moderation'] = mods
res['TUR_moderation_n'] = int(len(t))

# (3) meta-analysis of per-system M4 estimates
bc = json.load(open('results/a2_lpm_bycountry.json'))
names = sorted(bc)
b = np.array([bc[c]['M4_plus_response_style']['b'] for c in names])
s = np.array([bc[c]['M4_plus_response_style']['se'] for c in names])
ok = np.isfinite(b) & np.isfinite(s) & (s > 0)
names = [n for n, k in zip(names, ok) if k]; b = b[ok]; s = s[ok]
w = 1 / s ** 2
fe = np.sum(w * b) / np.sum(w)
Q = np.sum(w * (b - fe) ** 2); k = len(b)
tau2 = max(0.0, (Q - (k - 1)) / (np.sum(w) - np.sum(w ** 2) / np.sum(w)))
wr = 1 / (s ** 2 + tau2)
re = np.sum(wr * b) / np.sum(wr); re_se = np.sqrt(1 / np.sum(wr))
I2 = max(0.0, (Q - (k - 1)) / Q)
p = 2 * stats.norm.sf(np.abs(b / s))
order = np.argsort(p); bh = np.empty(k); prev = 1.0
for rank, i in enumerate(order[::-1]):
    rk = k - rank
    prev = min(prev, p[i] * k / rk); bh[i] = prev
holm = np.empty(k); run = 0.0
for rank, i in enumerate(order):
    run = max(run, min(1.0, p[i] * (k - rank))); holm[i] = run
res['meta_M4'] = {'k': k, 'fixed': float(fe), 'random': float(re), 'random_se': float(re_se), 'tau': float(np.sqrt(tau2)),
                  'Q': float(Q), 'Q_p': float(stats.chi2.sf(Q, k - 1)), 'I2': float(I2),
                  'pred_interval': [float(re - 1.96 * np.sqrt(tau2 + re_se ** 2)), float(re + 1.96 * np.sqrt(tau2 + re_se ** 2))],
                  'n_pos_raw05': int(np.sum((p < .05) & (b > 0))), 'n_neg_raw05': int(np.sum((p < .05) & (b < 0))),
                  'n_pos_bh05': int(np.sum((bh < .05) & (b > 0))), 'n_neg_bh05': int(np.sum((bh < .05) & (b < 0))),
                  'bh_sig': {n: float(x) for n, x, q in zip(names, b, bh) if q < .05},
                  'TUR_p': float(p[names.index('TUR')]), 'TUR_bh': float(bh[names.index('TUR')]), 'TUR_holm': float(holm[names.index('TUR')]),
                  'TUR_rank_desc': int(1 + np.sum(b > b[names.index('TUR')]))}
print(res['meta_M4'], flush=True)
json.dump(res, open('results/a6_items_turkey.json', 'w'), indent=1)
