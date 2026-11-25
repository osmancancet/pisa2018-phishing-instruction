"""Additional analyses: response style, alternative outcomes, exclusions, heterogeneity (separate from A1-A6, which are unchanged).

Same estimator as A2 (weighted LPM, country fixed effects, senate weights when pooled,
BRR Fay 0.5 x 80, Rubin over 10 reading PVs). Sample = A2 complete-case sample.
  ars  : external acquiescence (ARS) and extreme responding (ERS) from 38 unrelated
         agree/disagree items replace or complement the outcome-derived 'safe_mean'.
  alt  : alternative outcomes (rating >= 5, rating = 6, click rated strictly highest
         of the five options, METASPAM in SD units).
  school: leave-one-out school share of students taught (Jerrim-style school exposure).
  excl : exclusions defined outside the outcome (high ARS, high ERS, low effort) and a
         Jerrim-like restricted sample.
  country: per-system M3/M4-ARS estimates, random-effects meta-analysis, empirical Bayes.
  tur  : Türkiye on the LPM sample: moderation (Holm across 3 moderators), programme
         and school fixed effects.
  bal  : balanced acquiescence index (belonging scale, 3 positive and 3 negative items).
  rules: each exclusion rule of Jerrim (2023) on its own.
  turb : Türkiye outcome-based bound with M4 covariates.
  sens2: Cinelli-Hazlett robustness values and benchmark bounds on point estimates.
  logit: reading x instruction on the log-odds scale.
  jerrim: combined restricted sample, pooled and by system.
Usage: python exp_r1.py <part>
"""
import json
import sys
import numpy as np
import pandas as pd
from scipy import stats
from common import brr_var, rubin, senate_scale, REP_W, PV_READ

OTHER = [f'ST158Q0{i}_y' for i in range(1, 7)]
BASE = ['female', 'ESCS', 'imm1', 'imm2', 'imm_miss', 'GRADE', 'AGE', 'effort', 'effort_miss']
ARSC = ['ARS', 'ars_miss']

KEEP = (['CNT', 'CNTSCHID', 'OECD', 'PROGN', 'click', 'taught', 'ST004D01T', 'ST158Q07HA', 'EFFORT1', 'IMMIG', 'ESCS',
         'GRADE', 'AGE', 'female', 'METASPAM', 'ENTUSE', 'n_other_taught', 'ARS', 'ERS', 'ARS_bal', 'W_FSTUWT']
        + [f'ST166Q0{i}HA' for i in range(1, 6)] + OTHER + REP_W + PV_READ)
d = pd.read_parquet('data/analysis_r1.parquet', columns=KEEP)
d = d.dropna(subset=['click', 'taught']).reset_index(drop=True)
d['effort_miss'] = d['EFFORT1'].isna().astype(float)
d['effort'] = d['EFFORT1'].fillna(d.groupby('CNT')['EFFORT1'].transform('mean'))
d['imm_miss'] = d['IMMIG'].isna().astype(float)
d['imm1'] = (d['IMMIG'] == 2).astype(float)
d['imm2'] = (d['IMMIG'] == 3).astype(float)
d['safe_mean'] = d[['ST166Q02HA', 'ST166Q04HA', 'ST166Q05HA']].mean(axis=1, skipna=False)
d['inconsistent'] = ((d['ST166Q03HA'] >= 4) & (d['ST166Q04HA'] >= 4)).astype(float)
need = ['taught', 'click', 'safe_mean', 'PV1READ', 'female', 'ESCS', 'GRADE', 'AGE', 'effort'] + OTHER
d = d.dropna(subset=need).reset_index(drop=True)
for c in PV_READ:
    d[c] = d[c] / 100.0
d['ars_miss'] = d['ARS'].isna().astype(float)
d['arsb_miss'] = d['ARS_bal'].isna().astype(float)
d['ARS_bal'] = d['ARS_bal'].fillna(d.groupby('CNT')['ARS_bal'].transform('mean')).fillna(d['ARS_bal'].mean()) * 10  # Israel: no ST034 data
d['ERS_raw'] = d['ERS']
for c in ('ARS', 'ERS'):
    d[c] = d[c].fillna(d.groupby('CNT')[c].transform('mean'))
    d[c] = d[c] * 10  # per 0.1 (= about 4 of 38 items)
it = d[[f'ST166Q0{i}HA' for i in range(1, 6)]]
d['click5'] = (d['ST166Q03HA'] >= 5).astype(float)
d['click6'] = (d['ST166Q03HA'] == 6).astype(float)
others = it.drop(columns='ST166Q03HA').max(axis=1)
d['click_top'] = np.where(it.notna().all(axis=1), (d['ST166Q03HA'] > others).astype(float), np.nan)
d['metaspam_sd'] = d['METASPAM'] / d['METASPAM'].std()
# leave-one-out school share taught (students in schools with >= 5 sampled students)
gs = d.groupby(['CNT', 'CNTSCHID'])['taught']
s_sum, s_n = gs.transform('sum'), gs.transform('count')
d['school_taught'] = np.where(s_n >= 5, (s_sum - d['taught']) / (s_n - 1), np.nan)


def wls(X, y, w):
    Xw = X * w[:, None]
    return np.linalg.solve(X.T @ Xw, Xw.T @ y)[0]


def demean(X, grp, w):
    sw = np.bincount(grp, weights=w)
    return X - (np.vstack([np.bincount(grp, weights=w * X[:, j]) for j in range(X.shape[1])]).T / sw[:, None])[grp]


def fit_fe(X, grp, y, w):
    Z = demean(np.column_stack([y, X]), grp, w)
    yd, Xd = Z[:, 0], Z[:, 1:]
    keep = np.sqrt(np.average(Xd ** 2, axis=0, weights=w)) > 1e-10
    keep[0] = True
    return wls(Xd[:, keep], yd, w)


def estimate(g, cols, yvar='click', treat='taught', pooled=True, uses_pv=True, fe='CNT'):
    """Coefficient on `treat`; each weight vector demeans [y, treat, cols, PV1..PV10] once."""
    g = g[g[yvar].notna() & g[treat].notna()].reset_index(drop=True)
    if pooled:
        wfull = senate_scale(g, 'W_FSTUWT').to_numpy(); wreps = [senate_scale(g, w).to_numpy() for w in REP_W]
    else:
        wfull = g['W_FSTUWT'].to_numpy(); wreps = [g[w].to_numpy() for w in REP_W]
    grp = pd.factorize(g[fe].astype(str))[0] if fe else np.zeros(len(g), int)
    pvs = PV_READ if uses_pv else []
    A = g[[yvar, treat] + cols + pvs].to_numpy(float)
    k = 2 + len(cols)

    def fits(w):
        Z = demean(A, grp, w)
        out = []
        for j in (range(len(pvs)) if pvs else [None]):
            sel = list(range(k)) + ([k + j] if j is not None else [])
            yd, Xd = Z[:, 0], Z[:, sel[1:]]
            keep = np.sqrt(np.average(Xd ** 2, axis=0, weights=w)) > 1e-10
            keep[0] = True
            out.append(wls(Xd[:, keep], yd, w))
        return np.array(out)
    full = fits(wfull)
    reps = np.array([fits(w) for w in wreps])  # 80 x M
    vars_ = [brr_var(full[m], reps[:, m]) for m in range(len(full))]
    est, se = rubin(full, vars_)
    return {'b': float(est), 'se': float(se), 'n': int(len(g))}


part = sys.argv[1]
res = {}
M3 = BASE + OTHER
M4A = BASE + OTHER + ARSC
if part == 'ars':
    res['corr_ARS_safe_mean'] = float(np.corrcoef(d['ARS'], d['safe_mean'])[0, 1])
    res['corr_ARS_click'] = float(np.corrcoef(d['ARS'], d['click'])[0, 1])
    res['corr_ARS_taught'] = float(np.corrcoef(d['ARS'], d['taught'])[0, 1])
    res['ars_missing_share'] = float(d['ars_miss'].mean())
    specs = {'M4ars': M4A, 'M4ars_ers': M4A + ['ERS'], 'M4ars_safe': M4A + ['safe_mean'],
             'M2ars': BASE + ARSC}
    for k, cols in specs.items():
        res[k] = estimate(d, cols); print(k, res[k], flush=True)
    # coefficient on ARS itself in M4ars (is ARS related to clicking?)
    res['M4ars_ARScoef'] = estimate(d, [c for c in M4A if c != 'ARS'] + ['taught'], treat='ARS')
    print(res, flush=True)
elif part == 'alt':
    for y in ('click5', 'click6', 'click_top', 'metaspam_sd'):
        for k, cols in (('M0', []), ('M3', M3), ('M4ars', M4A)):
            res[f'{y}_{k}'] = estimate(d, cols, yvar=y, uses_pv=(k != 'M0')); print(y, k, res[f'{y}_{k}'], flush=True)
    res['share_click5'] = float(np.average(d['click5'], weights=senate_scale(d, 'W_FSTUWT')))
    res['share_click6'] = float(np.average(d['click6'], weights=senate_scale(d, 'W_FSTUWT')))
    ct = d[d['click_top'].notna()]
    res['share_click_top'] = float(np.average(ct['click_top'], weights=senate_scale(ct, 'W_FSTUWT')))
elif part == 'school':
    s = d[d['school_taught'].notna()].reset_index(drop=True)
    res['n_school_sample'] = int(len(s))
    res['school_M0'] = estimate(s, [], treat='school_taught', uses_pv=False)
    res['school_M3'] = estimate(s, M3, treat='school_taught')
    res['school_M4ars'] = estimate(s, M4A, treat='school_taught')
    res['school_M4ars_plus_own'] = estimate(s, M4A + ['taught'], treat='school_taught')
    res['own_M4ars_plus_school'] = estimate(s, M4A + ['school_taught'])
    print(res, flush=True)
elif part == 'excl':
    q90 = d.groupby('CNT')['ARS'].transform(lambda x: x.quantile(0.9))
    sets = {
        'ars_top10_dropped': d[(d['ARS'] < q90) & (d['ars_miss'] == 0)],
        'ers_ge_09_dropped': d[(d['ERS_raw'] < 0.9) & (d['ars_miss'] == 0)],
        'low_effort_dropped': d[(d['EFFORT1'] >= 5)],
        'jerrim_like': d[(d['EFFORT1'] >= 5) & (d['ERS_raw'] < 0.9) & (d['inconsistent'] == 0)],
    }
    for k, s in sets.items():
        s = s.reset_index(drop=True)
        res[k + '_share_kept'] = float(len(s) / len(d))
        for m, cols in (('M0', []), ('M3', M3), ('M4ars', M4A)):
            res[f'{k}_{m}'] = estimate(s, cols, uses_pv=(m != 'M0')); print(k, m, res[f'{k}_{m}'], flush=True)
elif part == 'country':
    for cnt, g in d.groupby('CNT'):
        g = g.reset_index(drop=True)
        res[cnt] = {m: estimate(g, cols, pooled=False, fe=None) for m, cols in (('M3', M3), ('M4ars', M4A))}
        print(cnt, res[cnt]['M4ars'], flush=True)
elif part == 'tur':
    t = d[d.CNT == 'TUR'].reset_index(drop=True)
    res['n'] = int(len(t))
    for m, cols in (('M3', M3), ('M4ars', M4A)):
        res[m] = estimate(t, cols, pooled=False, fe=None)
        res[m + '_progFE'] = estimate(t, cols, pooled=False, fe='PROGN')
        res[m + '_schoolFE'] = estimate(t, cols, pooled=False, fe='CNTSCHID')
        print(m, res[m], res[m + '_progFE'], res[m + '_schoolFE'], flush=True)
    res['n_prog'] = int(t['PROGN'].nunique()); res['n_school'] = int(t['CNTSCHID'].nunique())
    # moderation on the LPM sample, M4ars covariates
    w0 = t['W_FSTUWT'].to_numpy()
    mods = {}
    for mod in ('read', 'ESCS', 'female'):
        ests, vars_ = [], []
        for pv in PV_READ:
            r = t[pv].to_numpy(); rc = r - np.average(r, weights=w0)
            m = {'read': rc, 'ESCS': t['ESCS'].to_numpy() - np.average(t['ESCS'], weights=w0),
                 'female': t['female'].to_numpy() - 0.5}[mod]
            X = np.column_stack([t['taught'], t['taught'] * m, rc, t[M4A].to_numpy(float), np.ones(len(t))])
            X = X[:, np.r_[np.std(X[:, :-1], axis=0) > 0, True]]
            y = t['click'].to_numpy(float)

            def fit(w):
                Xw = X * w[:, None]
                return np.linalg.solve(X.T @ Xw, Xw.T @ y)[:2]
            b = fit(w0); ests.append(b); vars_.append(brr_var(b, np.array([fit(t[w].to_numpy()) for w in REP_W])))
        est, se = rubin(ests, vars_)
        p = 2 * stats.norm.sf(abs(est[1] / se[1]))
        mods[mod] = {'taught': float(est[0]), 'taught_se': float(se[0]), 'interaction': float(est[1]),
                     'interaction_se': float(se[1]), 'p': float(p)}
    ps = sorted(mods, key=lambda k: mods[k]['p']); run = 0
    for i, k in enumerate(ps):
        run = max(run, min(1.0, mods[k]['p'] * (3 - i))); mods[k]['p_holm'] = float(run)
    res['moderation'] = mods
    print(res, flush=True)
elif part == 'jerrim':
    # Jerrim (2023)-type restricted sample, with the restrictions available in the public file:
    # test effort > 4, no extreme pattern (all five ST166 ratings >= 4 or all <= 2), not inconsistent
    # (click and delete both >= 4). Response-time restrictions cannot be applied (no timing data used).
    it5 = d[[f'ST166Q0{i}HA' for i in range(1, 6)]]
    extreme_pat = (it5.ge(4).all(axis=1) | it5.le(2).all(axis=1)) & it5.notna().all(axis=1)
    keep = (d['EFFORT1'] > 4) & ~extreme_pat & (d['inconsistent'] == 0)
    res['share_extreme_pattern'] = float(extreme_pat.mean()); res['share_low_effort'] = float((d['EFFORT1'] <= 4).mean())
    res['share_kept'] = float(keep.mean())
    j = d[keep].reset_index(drop=True)
    jo = j[j['OECD'] == 1].reset_index(drop=True)
    w = senate_scale(jo, 'W_FSTUWT')
    res['oecd_click_share'] = float(np.average(jo['click'], weights=w))
    res['oecd_click_share_unrestricted'] = float(np.average(d[d.OECD == 1]['click'], weights=senate_scale(d[d.OECD == 1], 'W_FSTUWT')))
    for name, s_, cols in (('oecd_M0', jo, []), ('oecd_M3', jo, M3), ('oecd_M4ars', jo, M4A),
                           ('all_M0', j, []), ('all_M3', j, M3), ('all_M4ars', j, M4A)):
        res[name] = estimate(s_, cols, uses_pv=bool(cols)); print(name, res[name], flush=True)
    res['by_country'] = {}
    for cnt, g in j.groupby('CNT'):
        g = g.reset_index(drop=True)
        res['by_country'][cnt] = {'M0': estimate(g, [], pooled=False, fe=None, uses_pv=False),
                                  'M4ars': estimate(g, M4A, pooled=False, fe=None),
                                  'click_share': float(np.average(g['click'], weights=g['W_FSTUWT'])), 'n': int(len(g))}
elif part == 'bal':
    BAL = ['ARS_bal', 'arsb_miss']
    ok = (d['arsb_miss'] == 0) & (d['ars_miss'] == 0)
    for v in ('click', 'taught', 'ARS'):
        res['corr_bal_' + v] = float(np.corrcoef(d.loc[ok, 'ARS_bal'], d.loc[ok, v])[0, 1])
    res['systems_without_bal'] = sorted(d.loc[d.groupby('CNT')['arsb_miss'].transform('min') == 1, 'CNT'].unique().tolist())
    res['arsb_missing_share'] = float(d['arsb_miss'].mean())
    res['M4bal'] = estimate(d, M3 + BAL); print('M4bal', res['M4bal'], flush=True)
    res['M4both'] = estimate(d, M4A + BAL); print('M4both', res['M4both'], flush=True)
    res['M4bal_coef'] = estimate(d, M3 + ['arsb_miss', 'taught'], treat='ARS_bal'); print(res['M4bal_coef'], flush=True)
elif part == 'rules':
    it5 = d[[f'ST166Q0{i}HA' for i in range(1, 6)]]
    allhigh = it5.ge(4).all(axis=1) & it5.notna().all(axis=1)
    alllow = it5.le(2).all(axis=1) & it5.notna().all(axis=1)
    sets = {'effort': d['EFFORT1'] > 4, 'not_allhigh': ~allhigh, 'not_alllow': ~alllow, 'consistent': d['inconsistent'] == 0}
    for k, m in sets.items():
        s_ = d[m].reset_index(drop=True)
        res[k + '_share_kept'] = float(m.mean())
        res[k + '_M0'] = estimate(s_, [], uses_pv=False); res[k + '_M4ars'] = estimate(s_, M4A)
        print(k, res[k + '_M0'], res[k + '_M4ars'], flush=True)
elif part == 'turb':
    t = d[d.CNT == 'TUR'].reset_index(drop=True)
    res['TUR_M4ars_safe'] = estimate(t, M4A + ['safe_mean'], pooled=False, fe=None); print(res, flush=True)
elif part == 'sens2':
    # Cinelli & Hazlett (2020) on point estimates (weighted within-system OLS, PV1), formulas as in sensemakr
    g = d; w = senate_scale(g, 'W_FSTUWT').to_numpy(); grp = pd.factorize(g['CNT'])[0]

    def resid(target, others):
        Z = demean(g[[target] + others].to_numpy(float), grp, w)
        yv, O = Z[:, 0], Z[:, 1:]
        if O.shape[1] == 0:
            return yv
        keep = np.sqrt(np.average(O ** 2, axis=0, weights=w)) > 1e-10
        O = O[:, keep]; Ow = O * w[:, None]
        return yv - O @ np.linalg.solve(O.T @ Ow, Ow.T @ yv)

    def pr2(a, b):
        return float(np.sum(w * a * b) ** 2 / (np.sum(w * a * a) * np.sum(w * b * b)))
    for name, cov in (('M0', []), ('M3', M3 + ['PV1READ']), ('M4ars', M4A + ['PV1READ'])):
        dr = resid('taught', cov); yr = resid('click', cov)
        b = float(np.sum(w * dr * yr) / np.sum(w * dr * dr))
        r2 = pr2(dr, yr); f2 = r2 / (1 - r2)
        rv = 0.5 * (np.sqrt(f2 ** 2 + 4 * f2) - f2)
        res[name] = {'b': b, 'partial_r2_treat': r2, 'RV_q1': float(rv)}
    cov = M4A + ['PV1READ']
    dr = resid('taught', cov); yr = resid('click', cov)
    yres = yr - res['M4ars']['b'] * dr  # residual of y on d and covariates
    sd_ratio = np.sqrt(np.average(yres ** 2, weights=w) / np.average(dr ** 2, weights=w))
    for bm in ('PV1READ', 'ESCS'):
        oth = [c for c in cov if c != bm]
        r2dxj = pr2(resid(bm, oth), resid('taught', oth))
        r2yxj = pr2(resid(bm, oth + ['taught']), resid('click', oth + ['taught']))
        for k in (1, 2, 3):
            r2dz = k * r2dxj / (1 - r2dxj)
            r2zxj = k * r2dxj ** 2 / ((1 - k * r2dxj) * (1 - r2dxj))
            r2yz = ((np.sqrt(k) + np.sqrt(r2zxj)) / np.sqrt(1 - r2zxj)) ** 2 * (r2yxj / (1 - r2yxj))
            r2yz = min(r2yz, 1.0)
            bias = np.sqrt(r2yz * r2dz / (1 - r2dz)) * sd_ratio
            res[f'bound_{bm}_k{k}'] = {'r2dz': float(r2dz), 'r2yz': float(r2yz), 'bias': float(bias), 'adjusted': float(res['M4ars']['b'] - bias)}
        res[f'bench_{bm}'] = {'r2_treat': r2dxj, 'r2_outcome': r2yxj}
    print(res, flush=True)
elif part == 'logit':
    # reading x instruction on the log-odds scale: pooled weighted logit with system dummies, PV1, BRR SE
    from sklearn.linear_model import LogisticRegression
    g = d
    r = (g['PV1READ'] - np.average(g['PV1READ'], weights=senate_scale(g, 'W_FSTUWT'))).to_numpy()
    r = r / np.sqrt(np.average(r ** 2, weights=senate_scale(g, 'W_FSTUWT')))
    X = np.column_stack([g['taught'] * r, g['taught'], r, g[M4A].to_numpy(float), pd.get_dummies(g['CNT'], drop_first=True).to_numpy(float)])
    mu, sd = X.mean(0), X.std(0); sd[sd == 0] = 1
    Xs = (X - mu) / sd
    y = g['click'].to_numpy(int)

    def fit(wv):
        m = LogisticRegression(C=1e6, max_iter=3000, tol=1e-6).fit(Xs, y, sample_weight=wv / wv.mean())
        return m.coef_[0][:2] / sd[:2]
    full = fit(senate_scale(g, 'W_FSTUWT').to_numpy())
    reps = np.array([fit(senate_scale(g, x).to_numpy()) for x in REP_W])
    se = np.sqrt(brr_var(full, reps))
    res = {'inter_read_logodds': float(full[0]), 'inter_read_se': float(se[0]), 'taught_logodds': float(full[1]), 'taught_se': float(se[1])}
    print(res, flush=True)
elif part == 'items':
    # within-system item differences (M3 + ARS covariates) and a risky-minus-safe contrast
    d['risky_minus_safe'] = d[['ST166Q01HA', 'ST166Q03HA']].mean(axis=1) - d['safe_mean']
    for itm in ['ST166Q01HA', 'ST166Q02HA', 'ST166Q03HA', 'ST166Q04HA', 'ST166Q05HA', 'risky_minus_safe']:
        res[itm] = estimate(d, M4A, yvar=itm); print(itm, res[itm], flush=True)
    t = d[d.CNT == 'TUR'].reset_index(drop=True)
    for itm in ['ST166Q01HA', 'ST166Q02HA', 'ST166Q03HA', 'ST166Q04HA', 'ST166Q05HA', 'risky_minus_safe']:
        res['TUR_' + itm] = estimate(t, M4A, yvar=itm, pooled=False, fe=None); print('TUR', itm, res['TUR_' + itm], flush=True)
elif part == 'sep':
    # each other instruction topic entered on its own (M2 + ARS + one topic)
    for o in OTHER:
        res[o] = estimate(d, BASE + ARSC + [o]); print(o, res[o], flush=True)
    # M2 -> M3 change within ARS terciles (country-specific cut points)
    terc = d.groupby('CNT')['ARS'].transform(lambda x: pd.qcut(x.rank(method='first'), 3, labels=False))
    for k in range(3):
        s = d[(terc == k) & (d['ars_miss'] == 0)].reset_index(drop=True)
        res[f'ars_t{k + 1}_M2'] = estimate(s, BASE); res[f'ars_t{k + 1}_M3'] = estimate(s, M3)
        print('tercile', k, res[f'ars_t{k + 1}_M2'], res[f'ars_t{k + 1}_M3'], flush=True)
elif part == 'inter':
    # pooled student-level moderation, M4ars covariates, system FE, Holm across moderators
    mods = ['read', 'ESCS', 'female', 'effort', 'ENTUSE', 'n_other_taught', 'ARS']
    g = d.dropna(subset=['ENTUSE']).reset_index(drop=True)
    w = senate_scale(g, 'W_FSTUWT').to_numpy(); wreps = [senate_scale(g, x).to_numpy() for x in REP_W]
    grp = pd.factorize(g['CNT'])[0]; y = g['click'].to_numpy(float)
    out = {}
    for mod in mods:
        ests, vars_ = [], []
        for pv in PV_READ:
            mv = (g[pv] if mod == 'read' else g[mod]).to_numpy(float)
            mv = mv - np.average(mv, weights=w)
            if mod in ('ESCS', 'effort', 'ENTUSE', 'read'):
                mv = mv / np.sqrt(np.average(mv ** 2, weights=w))  # per SD
            extra = [mv] if mod == 'ENTUSE' else []  # other moderators' main effects are already in M4ars (+PV)
            X = np.column_stack([g['taught'] * mv, g['taught']] + extra + [g[M4A + [pv]].to_numpy(float)])
            b = fit_fe(X, grp, y, w)
            ests.append(b); vars_.append(brr_var(b, [fit_fe(X, grp, y, ww) for ww in wreps]))
        est, se = rubin(ests, vars_)
        out[mod] = {'b': float(est), 'se': float(se), 'p': float(2 * stats.norm.sf(abs(est / se)))}
        print(mod, out[mod], flush=True)
    ps = sorted(out, key=lambda k: out[k]['p']); run = 0
    for i, k in enumerate(ps):
        run = max(run, min(1.0, out[k]['p'] * (len(ps) - i))); out[k]['p_holm'] = float(run)
    res = {'n': int(len(g)), 'moderators': out}
elif part == 'sens':
    # Cinelli & Hazlett (2020) robustness values for pooled M0, M3, M4ars (PV1, BRR t-statistics),
    # with reading and the count of other topics as benchmarks (partial R2 with treatment and outcome)
    def partial_r2(cols, target, others, yv):
        """partial R2 of `target` in a weighted within-country regression of yv on [target]+others"""
        g = d
        w = senate_scale(g, 'W_FSTUWT').to_numpy(); grp = pd.factorize(g['CNT'])[0]
        Z = demean(g[[yv, target] + others].to_numpy(float), grp, w)
        yy, X = Z[:, 0], Z[:, 1:]
        Xw = X * w[:, None]; beta = np.linalg.solve(X.T @ Xw, Xw.T @ yy)
        r = yy - X @ beta
        # FWL: residualise target on others
        O = X[:, 1:]; Ow = O * w[:, None]
        tr = X[:, 0] - O @ np.linalg.solve(O.T @ Ow, Ow.T @ X[:, 0])
        yr = yy - O @ np.linalg.solve(O.T @ Ow, Ow.T @ yy)
        return float(1 - np.sum(w * r ** 2) / np.sum(w * yr ** 2))
    for name, cols in (('M0', []), ('M3', M3), ('M4ars', M4A)):
        e = estimate(d, cols + (['PV1READ'] if cols else []), uses_pv=False)
        dof = e['n'] - len(cols) - 2 - d['CNT'].nunique()
        tval = e['b'] / e['se']; f = tval / np.sqrt(dof)
        rv = 0.5 * (np.sqrt(f ** 4 + 4 * f ** 2) - f ** 2)
        fa = max(0.0, f - stats.t.ppf(0.975, dof) / np.sqrt(dof - 1))
        rva = 0.5 * (np.sqrt(fa ** 4 + 4 * fa ** 2) - fa ** 2)
        res[name] = {'b': e['b'], 'se': e['se'], 't': float(tval), 'partial_r2_treat': float(tval ** 2 / (tval ** 2 + dof)),
                     'RV_q1': float(rv), 'RV_q1_a05': float(rva)}
        print(name, res[name], flush=True)
    base = M4A + ['PV1READ']
    for bm in ('PV1READ', 'n_other_taught'):
        oth = [c for c in base if c != bm and not (bm == 'n_other_taught' and c in OTHER)]
        res['bench_' + bm] = {'r2_with_outcome': partial_r2(None, bm, ['taught'] + oth, 'click'),
                              'r2_with_treatment': partial_r2(None, bm, oth, 'taught')}
        print(bm, res['bench_' + bm], flush=True)
json.dump(res, open(f'results/exp_r1_{part}.json', 'w'), indent=1)
