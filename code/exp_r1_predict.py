"""Prediction with system-grouped thresholds: LOCO LightGBM (seed 0) and logit with the F1 threshold tuned on
held-out *systems* (GroupShuffleSplit, 20% of training systems) instead of held-out students.
Saves out-of-fold predictions and reports per-system AUC on the held-out systems."""
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.metrics import roc_auc_score
from a3_predict import features, best_threshold, metrics, lgbm, logit
from common import load, senate_scale

d = load()
d = d[d['CNT'] != 'VNM'].reset_index(drop=True)
X = d[features(d)].astype(float)
y = d['click'].astype(int).to_numpy()
w = senate_scale(d, 'W_FSTUWT').to_numpy()
g = d['CNT'].to_numpy()
pred = pd.DataFrame({'CNT': g, 'click': y, 'w': w, 'fold': -1, 'p_lgbm': np.nan, 'p_logit': np.nan})
rows = []
for k, (tr, te) in enumerate(GroupKFold(n_splits=5).split(X, y, groups=g)):
    itr_i, ival_i = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=k).split(tr, groups=g[tr]))
    itr, ival = tr[itr_i], tr[ival_i]
    for name, mk in (('lgbm', lambda: lgbm(0)), ('logit', logit)):
        m = mk()
        if name == 'logit':
            m.fit(X.iloc[itr], y[itr], logisticregression__sample_weight=w[itr])
        else:
            m.fit(X.iloc[itr], y[itr], sample_weight=w[itr])
        thr = best_threshold(y[ival], m.predict_proba(X.iloc[ival])[:, 1], w[ival])
        p = m.predict_proba(X.iloc[te])[:, 1]
        pred.loc[te, 'p_' + name] = p; pred.loc[te, 'fold'] = k
        rows.append(dict(model=name, fold=k, **metrics(y[te], p, thr, w[te])))
        print(k, name, rows[-1]['auc'], rows[-1]['mcc'], thr, flush=True)
pred.to_parquet('results/exp_r1_predict_oof.parquet')
folds = pd.DataFrame(rows)
folds.to_csv('results/exp_r1_predict_folds.csv', index=False)
per = {}
for c, h in pred.groupby('CNT'):
    if h['click'].nunique() == 2:
        per[c] = {m: float(roc_auc_score(h['click'], h['p_' + m], sample_weight=h['w'])) for m in ('lgbm', 'logit')}
per_l = np.array([v['lgbm'] for v in per.values()])
out = {'folds_mean': folds.groupby('model')[['auc', 'mcc', 'f1', 'prauc', 'brier', 'thr']].mean().to_dict(),
       'folds_sd': folds.groupby('model')[['auc', 'mcc']].std().to_dict(),
       'pooled_auc': {m: float(roc_auc_score(pred['click'], pred['p_' + m], sample_weight=pred['w'])) for m in ('lgbm', 'logit')},
       'per_system_auc': per,
       'per_system_lgbm': {'median': float(np.median(per_l)), 'min': float(per_l.min()), 'max': float(per_l.max()),
                           'q25': float(np.quantile(per_l, .25)), 'q75': float(np.quantile(per_l, .75)), 'k': int(len(per_l))},
       'TUR_auc': per.get('TUR')}
print(json.dumps({k: v for k, v in out.items() if k != 'per_system_auc'}, indent=1))
json.dump(out, open('results/exp_r1_predict.json', 'w'), indent=1)
