"""A5: SHAP explanation of the endorsement model (LightGBM, all systems, seed 0).

TreeSHAP (Lundberg et al., 2020) on a 30,000-student country-balanced sample and on
all Turkish students. Reports mean |SHAP| ranking and the signed mean SHAP of 'taught'.
"""
import json
import numpy as np
import pandas as pd
import shap
from a3_predict import features, lgbm
from common import load, senate_scale

d = load()
d = d[d['CNT'] != 'VNM'].reset_index(drop=True)
F = features(d)
X = d[F].astype(float)
y = d['click'].astype(int).to_numpy()
w = senate_scale(d, 'W_FSTUWT').to_numpy()
m = lgbm(0).fit(X, y, sample_weight=w)
expl = shap.TreeExplainer(m)
res = {}
samp = pd.Index(np.concatenate([g.sample(min(len(g), 400), random_state=0).index.to_numpy() for _, g in d.groupby('CNT')]))
for name, idx in (('all', samp), ('TUR', d.index[d['CNT'] == 'TUR'])):
    sv = expl.shap_values(X.loc[idx])
    sv = sv[1] if isinstance(sv, list) else sv
    imp = pd.Series(np.abs(sv).mean(0), index=F).sort_values(ascending=False)
    signed = pd.Series(sv.mean(0), index=F)
    # direction: correlation between feature value and its SHAP value
    direc = {f: float(pd.Series(X.loc[idx, f].to_numpy()).corr(pd.Series(sv[:, i]))) for i, f in enumerate(F)}
    res[name] = {'n': int(len(idx)), 'mean_abs': imp.head(20).to_dict(), 'taught_rank': int(list(imp.index).index('taught')) + 1,
                 'taught_mean_abs': float(imp['taught']), 'taught_signed_mean': float(signed['taught']),
                 'direction_corr_top20': {f: direc[f] for f in imp.head(20).index}}
    np.save(f'results/a5_shap_{name}.npy', sv)
    X.loc[idx].to_parquet(f'results/a5_X_{name}.parquet')
    print(name, imp.head(12).round(4).to_dict(), 'taught rank', res[name]['taught_rank'])
json.dump(res, open('results/a5_shap.json', 'w'), indent=1)
