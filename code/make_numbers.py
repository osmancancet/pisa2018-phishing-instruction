"""Collect every number used in the manuscript into results/numbers.json (flat keys)."""
import json
import os
import numpy as np
import pandas as pd

R = 'results'
N = {}
a1 = json.load(open(f'{R}/a1_descriptives.json'))
N['n_students_a1'] = a1['summary']['n_students']
N['n_systems_a1'] = a1['summary']['n_systems']
N['mean_click_systems'] = a1['summary']['mean_click_systems']
N['mean_taught_systems'] = a1['summary']['mean_taught_systems']
N['corr_click_taught_systems'] = a1['summary']['corr_click_taught_systems']
c = a1['countries']
clicks = {k: v['click'] for k, v in c.items()}
taught = {k: v['taught'] for k, v in c.items()}
N['click_min_cnt'], N['click_min'] = min(clicks.items(), key=lambda x: x[1])
N['click_max_cnt'], N['click_max'] = max(clicks.items(), key=lambda x: x[1])
N['taught_min_cnt'], N['taught_min'] = min(taught.items(), key=lambda x: x[1])
N['taught_max_cnt'], N['taught_max'] = max(taught.items(), key=lambda x: x[1])
for k in ('click', 'click_se', 'taught', 'taught_se', 'click_t0', 'click_t1', 'n'):
    N[f'TUR_{k}'] = c['TUR'][k]
N['TUR_taught_rank_asc'] = 1 + sum(v < c['TUR']['taught'] for v in taught.values())

p = json.load(open(f'{R}/a2_lpm_pooled.json'))
ex = json.load(open(f'{R}/a2_lpm_extra.json'))
p['M5_ipsative_outcome'] = ex['M5_ipsative_outcome']
for k, v in p.items():
    N[f'pooled_{k}_b'] = v['b']; N[f'pooled_{k}_se'] = v['se']; N[f'pooled_{k}_n'] = v['n']
for k, v in ex.items():
    if isinstance(v, dict):
        N[f'{k}_b'] = v['b']; N[f'{k}_se'] = v['se']; N[f'{k}_n'] = v['n']
    else:
        N[k] = v
bc = json.load(open(f'{R}/a2_lpm_bycountry.json'))
N['n_systems_lpm'] = len(bc)
for m in ('M0_none', 'M3_plus_other_instruction', 'M4_plus_response_style'):
    N[f'TUR_{m}_b'] = bc['TUR'][m]['b']; N[f'TUR_{m}_se'] = bc['TUR'][m]['se']; N[f'TUR_{m}_n'] = bc['TUR'][m]['n']
    b = np.array([v[m]['b'] for v in bc.values()])
    N[f'median_{m}'] = float(np.nanmedian(b))
N['n_students_lpm_country_sum'] = int(sum(v['M0_none']['n'] for v in bc.values()))

a6 = json.load(open(f'{R}/a6_items_turkey.json'))
for scope, d in a6['items'].items():
    for lab, v in d.items():
        for kk in ('diff', 'se', 'mean_taught', 'mean_not'):
            N[f'item_{scope}_{lab}_{kk}'] = v[kk]
# Türkiye moderation now comes from exp_r1.py tur (same sample as the main models)
for kk, v in a6['meta_M4'].items():
    if kk == 'bh_sig':
        N['meta_bh_sig_systems'] = ', '.join(sorted(v))
    elif kk == 'pred_interval':
        N['meta_pi_lo'], N['meta_pi_hi'] = v
    else:
        N[f'meta_{kk}'] = v

# causal-forest results (a4) are no longer reported
if os.path.exists(f'{R}/a3_loco_folds.csv'):
    f = pd.read_csv(f'{R}/a3_loco_folds.csv')
    for (m), g in f.groupby('model'):
        # mean over seeds within fold, then mean and t-interval over folds
        per_fold = g.groupby('fold').mean(numeric_only=True)
        for kk in ('acc', 'prec', 'rec', 'f1', 'mcc', 'auc', 'prauc', 'brier'):
            x = per_fold[kk].to_numpy()
            from scipy import stats
            h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else float('nan')
            N[f'loco_{m}_{kk}'] = float(x.mean()); N[f'loco_{m}_{kk}_ci'] = float(h)
        N[f'loco_{m}_nfits'] = int(len(g))
        cm = g[['tn', 'fp', 'fn', 'tp']].sum()
        for kk in ('tn', 'fp', 'fn', 'tp'):
            N[f'loco_{m}_{kk}_share'] = float(cm[kk] / cm.sum())
    meta = json.load(open(f'{R}/a3_loco_meta.json'))
    N['loco_n'] = meta['n']; N['loco_n_features'] = len(meta['features']); N['loco_lgbm_seeds'] = int(f[f.model == 'lgbm'].seed.nunique())
if os.path.exists(f'{R}/a5_shap.json'):
    s = json.load(open(f'{R}/a5_shap.json'))
    for scope, v in s.items():
        N[f'shap_{scope}_n'] = v['n']
        N[f'shap_{scope}_taught_rank'] = v['taught_rank']
        N[f'shap_{scope}_taught_mean_abs'] = v['taught_mean_abs']
        N[f'shap_{scope}_top5'] = ', '.join(list(v['mean_abs'])[:5])
json.dump(N, open(f'{R}/numbers.json', 'w'), indent=1, default=float)
print(len(N), 'numbers')

# ---- additional analyses (exp_r1*.py) ----
def flat(prefix, obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            flat(f'{prefix}_{k}', v)
    elif isinstance(obj, list) and all(isinstance(x, (int, float)) for x in obj):
        for i, v in enumerate(obj):
            N[f'{prefix}_{i}'] = v
    elif isinstance(obj, list):
        N[prefix] = ', '.join(NAMES.get(x, x) for x in obj); N[prefix + '_count'] = len(obj)
    else:
        N[prefix] = obj


NAMES = json.load(open('code/system_names.json'))
for part in ('ars', 'alt', 'school', 'excl', 'tur', 'items', 'sep', 'inter', 'sens', 'sens2', 'meta', 'readgap', 'bal', 'rules', 'turb', 'logit'):
    fn = f'{R}/exp_r1_{part}.json'
    if os.path.exists(fn):
        flat(f'r1_{part}', json.load(open(fn)))
if os.path.exists(f'{R}/exp_r1_jerrim.json'):
    jj = json.load(open(f'{R}/exp_r1_jerrim.json'))
    flat('r1_jerrim', {k: v for k, v in jj.items() if k != 'by_country'})
    for c in ('NOR', 'SWE', 'DNK', 'TUR', 'CHL', 'KOR'):
        flat(f'r1_jerrim_{c}', jj['by_country'][c])
# Cinelli-Hazlett: bias bound for a confounder k times as strong as reading (k = 1, 3)
sens = json.load(open(f'{R}/exp_r1_sens.json'))
from scipy import stats as _st
e = sens['M4ars']; br = sens['bench_PV1READ']
dof = e['t'] ** 2 * (1 - e['partial_r2_treat']) / e['partial_r2_treat']
for kx in (1, 3):
    r2d = kx * br['r2_with_treatment']; r2y = kx * br['r2_with_outcome']
    bias = e['se'] * np.sqrt(dof) * np.sqrt(r2y * r2d / (1 - r2d))
    N[f'r1_sens_bias_read_k{kx}'] = float(bias)
    N[f'r1_sens_adj_read_k{kx}'] = float(e['b'] - bias)
# derived conveniences
N['r1_TUR_moderation_female_boys'] = N['r1_tur_moderation_female_taught'] - 0.5 * N['r1_tur_moderation_female_interaction']
N['r1_TUR_moderation_female_girls'] = N['r1_tur_moderation_female_taught'] + 0.5 * N['r1_tur_moderation_female_interaction']
N['r1_school_M4ars_per10'] = N['r1_school_school_M4ars_b'] / 10
N['r1_school_M4ars_per10_se'] = N['r1_school_school_M4ars_se'] / 10
if os.path.exists(f'{R}/exp_r1_predict.json'):
    pr = json.load(open(f'{R}/exp_r1_predict.json'))
    flat('r1_pred', {k: v for k, v in pr.items() if k != 'per_system_auc'})
json.dump(N, open(f'{R}/numbers.json', 'w'), indent=1, default=float)
print(len(N), 'numbers')
_inter = json.load(open(f'{R}/exp_r1_inter.json'))['moderators']
N['r1_inter_min_other_holm'] = min(_inter[k]['p_holm'] for k in ('female', 'ESCS', 'effort', 'ENTUSE'))
json.dump(N, open(f'{R}/numbers.json', 'w'), indent=1, default=float)
# complete-case loss and REML/HKSJ meta-analysis check
N['n_dropped_missing_cov'] = N['loco_n'] - N['pooled_M0_none_n']
N['share_dropped_missing_cov'] = N['n_dropped_missing_cov'] / N['loco_n']
if os.path.exists(f'{R}/exp_r1_meta_reml.json'):
    flat('r1_reml', json.load(open(f'{R}/exp_r1_meta_reml.json')))
json.dump(N, open(f'{R}/numbers.json', 'w'), indent=1, default=float)
# round 2: breadth of instruction, pooled school fixed effects, outcome-defined rules under a permutation null,
# extended balanced index, same-sample reading interaction, logit AME, item checks, sub-national units, meta-regression
for part in ('count', 'schfe', 'rules2', 'bal2', 'inter2', 'items2', 'subnat', 'meta'):
    fn = f'{R}/exp_r2_{part}.json'
    if os.path.exists(fn):
        flat(f'r2_{part}', json.load(open(fn)))
import pandas as _pd
_f = _pd.read_csv(f'{R}/exp_r1_predict_folds.csv')
N['r1_pred_acc_lgbm'] = float(_f[_f.model == 'lgbm']['acc'].mean())
N['r1_pred_majority_acc'] = 1 - N['r1_readgap_prevalence']
if 'r2_rules2_obs_M0_full' in N:
    for k in ('rule2a_allhigh', 'rule3_inconsistent'):
        N[f'r2_rules2_obs_shift_{k}'] = N[f'r2_rules2_obs_M0_{k}'] - N['r2_rules2_obs_M0_full']
        for nl in ('strata', 'system'):
            N[f'r2_rules2_share_{nl}_{k}'] = N[f'r2_rules2_perm_{nl}_{k}_bias_mean'] / N[f'r2_rules2_obs_shift_{k}']
if 'r2_meta_taught_beta_1' in N:
    N['r2_meta_taught_pi_reduction'] = 1 - N['r2_meta_taught_tau'] / N['r2_meta_tau_intercept_only']
json.dump(N, open(f'{R}/numbers.json', 'w'), indent=1, default=float)
if 'r2_meta_click_p_1' in N:
    N['r2_meta_other_min_p'] = min(N[f'r2_meta_{m}_p_1'] for m in ('click', 'ars', 'read', 'oecd'))
    json.dump(N, open(f'{R}/numbers.json', 'w'), indent=1, default=float)
_a1 = json.load(open(f'{R}/a1_descriptives.json'))['countries']
N['ITA_minus_TUR_taught'] = _a1['ITA']['taught'] - _a1['TUR']['taught']
json.dump(N, open(f'{R}/numbers.json', 'w'), indent=1, default=float)
if 'r2_meta_taught_p_1' in N:
    N['r2_meta_taught_p_bonf5'] = min(1.0, 5 * N['r2_meta_taught_p_1'])
    json.dump(N, open(f'{R}/numbers.json', 'w'), indent=1, default=float)
# round 4: specification curve and overlap
import glob as _glob
_cf = sorted(_glob.glob(f'{R}/exp_r3_curve_*.json'))
if len(_cf) == 6:
    _c = {}
    for f in _cf:
        _c.update(json.load(open(f)))
    _out = {'not_allhigh', 'consistent', 'combined'}
    _b = {k: v['b'] for k, v in _c.items()}
    _z = {k: v['b'] / v['se'] for k, v in _c.items()}
    N['r3_curve_n'] = len(_c)
    for grp, sel in (('all', lambda s: True), ('outside', lambda s: s not in _out), ('outcome', lambda s: s in _out)):
        ks = [k for k in _c if sel(k.split('|')[0])]
        N[f'r3_curve_{grp}_n'] = len(ks)
        N[f'r3_curve_{grp}_pos_sig'] = sum(_z[k] > 1.96 for k in ks)
        N[f'r3_curve_{grp}_neg_sig'] = sum(_z[k] < -1.96 for k in ks)
        N[f'r3_curve_{grp}_min'] = min(_b[k] for k in ks)
        N[f'r3_curve_{grp}_max'] = max(_b[k] for k in ks)
        N[f'r3_curve_{grp}_median'] = float(np.median([_b[k] for k in ks]))
    for o in ('click', 'click5', 'click6', 'click_top'):
        ks = [k for k in _c if k.split('|')[1] == o and k.split('|')[0] not in _out]
        N[f'r3_curve_outside_{o}_neg_sig'] = sum(_z[k] < -1.96 for k in ks)
        N[f'r3_curve_outside_{o}_n'] = len(ks)
    for nm, ms in (('M34', ('M3', 'M4')), ('M012', ('M0', 'M1', 'M2'))):
        for grp, sel in (('outside', lambda x: x not in _out), ('outcome', lambda x: x in _out)):
            ks = [k for k in _c if k.split('|')[2] in ms and sel(k.split('|')[0]) and k.split('|')[1] != 'click_top']
            N[f'r3_curve_{grp}_{nm}_noTop_n'] = len(ks)
            N[f'r3_curve_{grp}_{nm}_noTop_pos_sig'] = sum(_z[k] > 1.96 for k in ks)
            N[f'r3_curve_{grp}_{nm}_noTop_neg_sig'] = sum(_z[k] < -1.96 for k in ks)
            N[f'r3_curve_{grp}_{nm}_noTop_min'] = min(_b[k] for k in ks)
            N[f'r3_curve_{grp}_{nm}_noTop_max'] = max(_b[k] for k in ks)
    ks = [k for k in _c if k.split('|')[1] == 'click_top']
    N['r3_curve_top_n'] = len(ks); N['r3_curve_top_neg_sig'] = sum(_z[k] < -1.96 for k in ks)
    N['r3_curve_top_min'] = min(_b[k] for k in ks); N['r3_curve_top_max'] = max(_b[k] for k in ks)
    for m in ('M3', 'M4'):
        ks = [k for k in _c if k.split('|')[2] == m and k.split('|')[0] not in _out]
        N[f'r3_curve_outside_{m}_max_abs'] = max(abs(_b[k]) for k in ks)
    json.dump(N, open(f'{R}/numbers.json', 'w'), indent=1, default=float)
if os.path.exists(f'{R}/exp_r3_overlap.json'):
    _o = json.load(open(f'{R}/exp_r3_overlap.json'))
    for k, v in _o.items():
        if isinstance(v, dict):
            for kk in ('b', 'se', 'n'):
                N[f'r3_overlap_{k}_{kk}'] = v[kk]
        else:
            N[f'r3_overlap_{k}'] = v
    N['r3_overlap_t1_share_3plus'] = sum(_o[f'share_t1_n{k}'] for k in range(3, 7))
    N['r3_overlap_t0_share_3plus'] = sum(_o[f'share_t0_n{k}'] for k in range(3, 7))
    json.dump(N, open(f'{R}/numbers.json', 'w'), indent=1, default=float)
