"""A4: heterogeneity of the instruction-endorsement association.

Causal forest DML (Wager & Athey, 2018; Athey et al., 2019; econml) on a
country-balanced subsample (up to 2,000 students per system). Treatment:
taught to detect phishing (ST158Q07). Outcome: rates 'click the link' as
appropriate. Nuisance models: LightGBM. Controls W include system dummies,
background, reading, other ST158 instruction items and the safe-action rating
mean (response-style proxy). Moderators X are a small set of interpretable
student characteristics. Heterogeneity is validated on a held-out half with
the DR-learner tests of econml.validate (BLP, calibration, QINI uplift).
Omitted-variable sensitivity follows Chernozhukov et al. (2022) as implemented
in econml (sensitivity_summary / robustness_value).
"""
import json
import sys
import numpy as np
import pandas as pd
import lightgbm as lgb
from econml.dml import CausalForestDML
from econml.validate import DRTester
from sklearn.model_selection import train_test_split
from common import load

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0
OTHER = [f'ST158Q0{i}_y' for i in range(1, 7)]
XCOLS = ['read', 'ESCS', 'female', 'effort', 'METASUM', 'COMPICT', 'ENTUSE', 'n_other_taught', 'safe_mean', 'oecd']

d = load()
d = d[d['CNT'] != 'VNM'].copy()
d['read'] = d['PV1READ'] / 100.0
d['effort'] = d['EFFORT1'].fillna(d.groupby('CNT')['EFFORT1'].transform('mean'))
d['effort_miss'] = d['EFFORT1'].isna().astype(float)
d['safe_mean'] = d[['ST166Q02HA', 'ST166Q04HA', 'ST166Q05HA']].mean(axis=1, skipna=False)
d['oecd'] = d['OECD'].astype(float)
d['imm1'] = (d['IMMIG'] == 2).astype(float)
d['imm2'] = (d['IMMIG'] == 3).astype(float)
d = d.dropna(subset=['read', 'ESCS', 'female', 'effort', 'safe_mean', 'GRADE', 'AGE'] + OTHER)
# COMPICT / ENTUSE / METASUM can be missing (ICT questionnaire optional): median-fill + flags
for c in ('METASUM', 'COMPICT', 'ENTUSE'):
    d[c + '_miss'] = d[c].isna().astype(float)
    d[c] = d[c].fillna(d[c].median())
rng = np.random.default_rng(SEED)
keep_idx = np.concatenate([g.sample(min(len(g), 2000), random_state=SEED).index.to_numpy() for _, g in d.groupby('CNT')])
d = d.loc[keep_idx].reset_index(drop=True)
WCOLS = OTHER + ['GRADE', 'AGE', 'imm1', 'imm2', 'effort_miss', 'METASUM_miss', 'COMPICT_miss', 'ENTUSE_miss',
                 'UNDREM', 'JOYREAD', 'ICTRES', 'HOMEPOS', 'USESCH', 'ICTCLASS', 'INTICT', 'PERFEED', 'DISCLIMA']
for c in ['UNDREM', 'JOYREAD', 'ICTRES', 'HOMEPOS', 'USESCH', 'ICTCLASS', 'INTICT', 'PERFEED', 'DISCLIMA']:
    d[c + '_miss'] = d[c].isna().astype(float)
    d[c] = d[c].fillna(d[c].median())
    WCOLS.append(c + '_miss')
dummies = pd.get_dummies(d['CNT'], prefix='c', drop_first=True).astype(float)
W = pd.concat([d[WCOLS].astype(float), dummies], axis=1).to_numpy()
X = d[XCOLS].astype(float).to_numpy()
T = d['taught'].astype(int).to_numpy()
Y = d['click'].astype(float).to_numpy()
idx_tr, idx_va = train_test_split(np.arange(len(d)), test_size=0.5, random_state=SEED, stratify=d['CNT'])


def reg():
    return lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=100,
                             random_state=SEED, n_jobs=4, verbose=-1)


def clf():
    return lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=100,
                              random_state=SEED, n_jobs=4, verbose=-1)


cf = CausalForestDML(model_y=reg(), model_t=clf(), discrete_treatment=True, cv=5, n_estimators=1000,
                     min_samples_leaf=50, max_samples=0.5, honest=True, inference=True, random_state=SEED, n_jobs=4)
cf.fit(Y[idx_tr], T[idx_tr], X=X[idx_tr], W=W[idx_tr])
ate = cf.ate_inference(X=X[idx_tr])
res = {'seed': SEED, 'n_total': int(len(d)), 'n_train': int(len(idx_tr)), 'n_systems': int(d['CNT'].nunique()),
       'ate': float(ate.mean_point), 'ate_se': float(ate.stderr_mean),
       'ate_ci': [float(v) for v in ate.conf_int_mean()],
       'feature_importances': dict(zip(XCOLS, map(float, cf.feature_importances_)))}
try:
    lo, hi = cf.sensitivity_interval(alpha=0.05, c_y=0.05, c_t=0.05, rho=1.0)
    res['sens_interval_cy05_ct05'] = [float(lo), float(hi)]
    res['robustness_value'] = float(cf.robustness_value(alpha=0.05))
except Exception as e:  # report rather than hide
    res['sensitivity_error'] = repr(e)

# CATE on validation half; group averages by moderator bins
tau = cf.effect(X[idx_va])
v = d.iloc[idx_va].copy()
v['tau'] = tau
res['tau_sd'] = float(np.std(tau))
res['tau_by_read_quintile'] = {str(k): float(x) for k, x in v.groupby(pd.qcut(v['read'], 5, labels=False))['tau'].mean().items()}
res['tau_by_safe_mean'] = {str(k): float(x) for k, x in v.groupby(pd.cut(v['safe_mean'], [0, 3, 4, 5, 6.01], labels=False))['tau'].mean().items()}
res['tau_by_country'] = {k: float(x) for k, x in v.groupby('CNT')['tau'].mean().items()}

tester = DRTester(model_regression=reg(), model_propensity=clf(), cate=cf)
tester.fit_nuisance(X[idx_va], T[idx_va], Y[idx_va], X[idx_tr], T[idx_tr], Y[idx_tr])
ev = tester.evaluate_all(X[idx_va], X[idx_tr], n_groups=5)
summ = ev.summary()
res['validation'] = json.loads(summ.to_json(orient='records'))
print(summ)
json.dump(res, open(f'results/a4_cf_seed{SEED}.json', 'w'), indent=1)
print({k: res[k] for k in ('ate', 'ate_se', 'ate_ci', 'tau_sd')}, res.get('sens_interval_cy05_ct05'), res.get('robustness_value'))
