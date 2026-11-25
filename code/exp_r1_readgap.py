"""Predicted endorsement (out-of-fold LightGBM) and observed endorsement by within-system reading
quintile (PV1), senate weights; paired LightGBM-minus-logit AUC by fold."""
import json
import numpy as np
import pandas as pd
from common import load

d = load(); d = d[d['CNT'] != 'VNM'].reset_index(drop=True)
o = pd.read_parquet('results/exp_r1_predict_oof.parquet')
assert (o['CNT'].values == d['CNT'].values).all()
o['q'] = d.groupby('CNT')['PV1READ'].transform(lambda x: pd.qcut(x.rank(method='first'), 5, labels=False)).values
o['taught'] = d['taught'].values
res = {}
for q, g in o.groupby('q'):
    res[f'q{int(q) + 1}_pred'] = float(np.average(g['p_lgbm'], weights=g['w']))
    res[f'q{int(q) + 1}_obs'] = float(np.average(g['click'], weights=g['w']))
res['gap_obs_q1_q5'] = res['q1_obs'] - res['q5_obs']; res['gap_pred_q1_q5'] = res['q1_pred'] - res['q5_pred']
f = pd.read_csv('results/exp_r1_predict_folds.csv')
diff = f[f.model == 'lgbm'].set_index('fold')['auc'] - f[f.model == 'logit'].set_index('fold')['auc']
res['auc_diff_min'] = float(diff.min()); res['auc_diff_max'] = float(diff.max()); res['auc_diff_mean'] = float(diff.mean())
res['prevalence'] = float(np.average(o['click'], weights=o['w']))
from sklearn.linear_model import LogisticRegression
for m in ('lgbm', 'logit'):
    lp = np.log(o['p_' + m] / (1 - o['p_' + m])).to_numpy()[:, None]
    cal = LogisticRegression(C=1e6).fit(lp, o['click'], sample_weight=o['w'] / o['w'].mean())
    res[f'cal_slope_{m}'] = float(cal.coef_[0][0]); res[f'cal_intercept_{m}'] = float(cal.intercept_[0])
    # per-system calibration-in-the-large: observed minus mean predicted
    cil = o.groupby('CNT').apply(lambda h: np.average(h['click'], weights=h['w']) - np.average(h['p_' + m], weights=h['w']))
    res[f'cil_abs_median_{m}'] = float(cil.abs().median()); res[f'cil_abs_max_{m}'] = float(cil.abs().max())
print(res)
json.dump(res, open('results/exp_r1_readgap.json', 'w'), indent=1)
