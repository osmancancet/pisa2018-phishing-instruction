"""Round-2 analyses (exp_r1.py and the earlier scripts are unchanged).

The data preparation and the estimator are those of exp_r1.py: its preamble (everything
before the part dispatch) is executed here on data/analysis_r2.parquet, which adds the
extended balanced acquiescence index (prepare_r2.py).
  count  : instruction breadth. Count of all seven ST158 topics as exposure; common-support
           tables for the phishing item against the other six; coefficient of each other topic in M4.
  schfe  : pooled M0 and M4 with school fixed effects instead of system fixed effects.
  rules2 : share of endorsers removed by each outcome-defined rule, taught vs not taught, and a
           permutation null: click ratings permuted within system x rating-level strata, so that
           click carries no information on instruction beyond the student's general rating level;
           each rule is then applied and the M0 estimate compared with the full sample.
  bal2   : balanced acquiescence from belonging and reading self-concept (12 items); KR-20
           reliabilities of the agreement and balanced indices.
  inter2 : reading x instruction on one sample (ICT subsample) on both scales; average
           marginal effect of instruction from a weighted logit with M4 covariates.
  items2 : item ratings with extreme responding (ERS) added to M4; item ratings >= 4 (LPM).
  subnat : pooled M4 without the Russian regions (Moscow Region, Tatarstan) and without Spain.
  sysvars: system-level means used as meta-regression moderators.
Usage: python exp_r2.py <part>
"""
import json
import sys

_src = open('code/exp_r1.py').read().split("part = sys.argv[1]")[0]
_src = _src.replace("'data/analysis_r1.parquet'", "'data/analysis_r2.parquet'")
_src = _src.replace("'ARS_bal', 'W_FSTUWT']", "'ARS_bal', 'ARS_bal2', 'W_FSTUWT']")
_argv = sys.argv
exec(compile(_src, 'exp_r1_preamble', 'exec'))
sys.argv = _argv

part = sys.argv[1]
res = {}
M3 = BASE + OTHER
M4A = BASE + OTHER + ARSC
ITEMS = [f'ST166Q0{i}HA' for i in range(1, 6)]


def wshare(mask, w):
    return float(np.sum(w * mask) / np.sum(w))


if part == 'count':
    d['n_all7'] = d['taught'] + d[OTHER].sum(axis=1)
    d['n_other6'] = d[OTHER].sum(axis=1)
    w = senate_scale(d, 'W_FSTUWT').to_numpy()
    res['dist_n_all7'] = {int(k): wshare((d['n_all7'] == k).to_numpy(), w) for k in range(8)}
    t1, t0 = d['taught'] == 1, d['taught'] == 0
    w1, w0 = w[t1.to_numpy()], w[t0.to_numpy()]
    res['taught_share_no_other'] = wshare((d.loc[t1, 'n_other6'] == 0).to_numpy(), w1)
    res['taught_share_all_other'] = wshare((d.loc[t1, 'n_other6'] == 6).to_numpy(), w1)
    res['untaught_share_no_other'] = wshare((d.loc[t0, 'n_other6'] == 0).to_numpy(), w0)
    res['untaught_share_all_other'] = wshare((d.loc[t0, 'n_other6'] == 6).to_numpy(), w0)
    res['taught_by_n_other6'] = {int(k): wshare(d.loc[d['n_other6'] == k, 'taught'].to_numpy(), w[(d['n_other6'] == k).to_numpy()])
                                 for k in range(7)}
    res['taught_mean_other6'] = float(np.average(d.loc[t1, 'n_other6'], weights=w1))
    res['untaught_mean_other6'] = float(np.average(d.loc[t0, 'n_other6'], weights=w0))
    res['n_all7_M2ars'] = estimate(d, BASE + ARSC, treat='n_all7'); print('count', res['n_all7_M2ars'], flush=True)
    res['n_all7_M0'] = estimate(d, [], treat='n_all7', uses_pv=False); print('count M0', res['n_all7_M0'], flush=True)
    for o in OTHER:
        res[o + '_M4'] = estimate(d, [c for c in M4A if c != o] + ['taught'], treat=o); print(o, res[o + '_M4'], flush=True)
elif part == 'schfe':
    d['sch'] = d['CNT'].astype(str) + '_' + d['CNTSCHID'].astype('int64').astype(str)
    res['n_schools'] = int(d['sch'].nunique())
    res['M0_schFE'] = estimate(d, [], fe='sch', uses_pv=False); print('M0', res['M0_schFE'], flush=True)
    res['M4_schFE'] = estimate(d, M4A, fe='sch'); print('M4', res['M4_schFE'], flush=True)
elif part == 'rules2':
    it5 = d[ITEMS]
    full5 = it5.notna().all(axis=1)
    w = senate_scale(d, 'W_FSTUWT').to_numpy()

    def rule_masks(c, it):
        allhigh = it.ge(4).all(axis=1) & full5
        alllow = it.le(2).all(axis=1) & full5
        incons = (c >= 4) & (it['ST166Q04HA'] >= 4)
        return {'rule2a_allhigh': allhigh, 'rule2b_alllow': alllow, 'rule3_inconsistent': incons}
    obs = rule_masks(d['ST166Q03HA'], it5)
    endorse = (d['click'] == 1).to_numpy()
    for k, m in obs.items():
        m = m.to_numpy()
        for tv in (1, 0):
            sel = endorse & (d['taught'] == tv).to_numpy()
            res[f'{k}_removed_share_endorsers_t{tv}'] = wshare(m[sel], w[sel])
            sel2 = (~endorse) & (d['taught'] == tv).to_numpy()
            res[f'{k}_removed_share_nonendorsers_t{tv}'] = wshare(m[sel2], w[sel2])

    grp = pd.factorize(d['CNT'])[0]

    def m0(click, keep):
        g = grp[keep]; ww = w[keep]
        A = np.column_stack([click[keep], d['taught'].to_numpy(float)[keep]])
        Z = demean(A, g, ww)
        return float(np.sum(ww * Z[:, 0] * Z[:, 1]) / np.sum(ww * Z[:, 1] ** 2))
    rng = np.random.default_rng(20261125)
    other_mean = d[['ST166Q01HA', 'ST166Q02HA', 'ST166Q04HA', 'ST166Q05HA']].mean(axis=1)
    strata = d['CNT'].astype(str) + '_' + (np.round(other_mean * 2) / 2).astype(str)
    sidx = pd.factorize(strata)[0]
    order = np.argsort(sidx, kind='stable'); bounds = np.flatnonzero(np.diff(sidx[order])) + 1
    blocks = np.split(order, bounds)
    cnt_order = np.argsort(grp, kind='stable'); cnt_blocks = np.split(cnt_order, np.flatnonzero(np.diff(grp[cnt_order])) + 1)
    c_obs = d['ST166Q03HA'].to_numpy(float)
    valid = ~np.isnan(c_obs)
    sims = {'strata': [], 'system': []}
    for name, blks in (('strata', blocks), ('system', cnt_blocks)):
        for r in range(200):
            c = c_obs.copy()
            for b in blks:
                b = b[valid[b]]
                c[b] = c_obs[rng.permutation(b)]
            cl = (c >= 4).astype(float)
            ms = rule_masks(pd.Series(c), it5.assign(ST166Q03HA=c))
            row = {'full': m0(cl, np.ones(len(d), bool))}
            for k, m in ms.items():
                row[k] = m0(cl, ~m.to_numpy())
            sims[name].append(row)
        S = pd.DataFrame(sims[name])
        res['perm_' + name] = {k: {'mean': float(S[k].mean()), 'sd': float(S[k].std()),
                                   'bias_mean': float((S[k] - S['full']).mean()),
                                   'bias_p2_5': float((S[k] - S['full']).quantile(0.025)),
                                   'bias_p97_5': float((S[k] - S['full']).quantile(0.975))} for k in S.columns}
        print(name, res['perm_' + name], flush=True)
    res['obs_M0_full'] = m0(d['click'].to_numpy(float), np.ones(len(d), bool))
    for k, m in obs.items():
        res['obs_M0_' + k] = m0(d['click'].to_numpy(float), ~m.to_numpy())
    res['n_perm'] = 200
elif part == 'bal2':
    d['arsb2_miss'] = d['ARS_bal2'].isna().astype(float)
    res['arsb2_missing_share'] = float(d['arsb2_miss'].mean())
    res['systems_without_bal2'] = sorted(d.loc[d.groupby('CNT')['arsb2_miss'].transform('min') == 1, 'CNT'].unique().tolist())
    ok = d['arsb2_miss'] == 0
    for v in ('click', 'taught', 'ARS'):
        res['corr_bal2_' + v] = float(np.corrcoef(d.loc[ok, 'ARS_bal2'], d.loc[ok, v])[0, 1])
    d['ARS_bal2'] = d['ARS_bal2'].fillna(d.groupby('CNT')['ARS_bal2'].transform('mean')).fillna(d['ARS_bal2'].mean()) * 10
    B2 = ['ARS_bal2', 'arsb2_miss']
    res['M4bal2'] = estimate(d, M3 + B2); print('M4bal2', res['M4bal2'], flush=True)
    res['M4bal2_coef'] = estimate(d, M3 + ['arsb2_miss', 'taught'], treat='ARS_bal2'); print(res['M4bal2_coef'], flush=True)
    # KR-20 of the agreement indicators (38-item index, and the 12 balanced items)
    lk = pd.read_csv('data/likert_items.csv.gz')
    L = lk.drop(columns=['CNT', 'CNTSTUID']); L = L.where(L.isin([1, 2, 3, 4]))
    rev = [c for c in L.columns if c.startswith(('ST034', 'ST036'))]
    A = (L >= 3).astype(float).where(L.notna()); A[rev] = (L[rev] <= 2).astype(float).where(L[rev].notna())

    def kr20(X):
        X = X.dropna()
        k = X.shape[1]; p = X.mean(); tot = X.sum(axis=1)
        return float(k / (k - 1) * (1 - (p * (1 - p)).sum() / tot.var())), int(len(X))
    res['kr20_ars38'], res['kr20_ars38_n'] = kr20(A)
    bal_cols = ['ST034Q01TA', 'ST034Q02TA', 'ST034Q03TA', 'ST034Q04TA', 'ST034Q05TA', 'ST034Q06TA',
                'ST161Q01HA', 'ST161Q02HA', 'ST161Q03HA', 'ST161Q06HA', 'ST161Q07HA', 'ST161Q08HA']
    res['kr20_bal12'], res['kr20_bal12_n'] = kr20(A[bal_cols])
    res['kr20_bal6'], res['kr20_bal6_n'] = kr20(A[bal_cols[:6]])
    print(res, flush=True)
elif part == 'inter2':
    from sklearn.linear_model import LogisticRegression
    g = d.dropna(subset=['ENTUSE']).reset_index(drop=True)
    w = senate_scale(g, 'W_FSTUWT').to_numpy()
    r = g['PV1READ'].to_numpy() - np.average(g['PV1READ'], weights=w)
    r = r / np.sqrt(np.average(r ** 2, weights=w))
    dums = pd.get_dummies(g['CNT'], drop_first=True).to_numpy(float)
    X = np.column_stack([g['taught'] * r, g['taught'], r, g[M4A].to_numpy(float), dums])
    mu, sd = X.mean(0), X.std(0); sd[sd == 0] = 1
    Xs = (X - mu) / sd
    y = g['click'].to_numpy(int)

    def fit(wv):
        m = LogisticRegression(C=1e6, max_iter=3000, tol=1e-6).fit(Xs, y, sample_weight=wv / wv.mean())
        return m.coef_[0][:2] / sd[:2]
    full = fit(w)
    reps = np.array([fit(senate_scale(g, x).to_numpy()) for x in REP_W])
    se = np.sqrt(brr_var(full, reps))
    res['ict_logit_inter_read'] = float(full[0]); res['ict_logit_inter_read_se'] = float(se[0]); res['ict_n'] = int(len(g))
    print(res, flush=True)
    # average marginal effect of instruction, full sample, weighted logit with M4 covariates + PV1 and system dummies
    g = d
    w = senate_scale(g, 'W_FSTUWT').to_numpy()
    dums = pd.get_dummies(g['CNT'], drop_first=True).to_numpy(float)
    X = np.column_stack([g['taught'], g['PV1READ'], g[M4A].to_numpy(float), dums])
    mu, sd = X.mean(0), X.std(0); sd[sd == 0] = 1
    y = g['click'].to_numpy(int)

    def ame(wv):
        m = LogisticRegression(C=1e6, max_iter=3000, tol=1e-6).fit((X - mu) / sd, y, sample_weight=wv / wv.mean())
        X1 = X.copy(); X1[:, 0] = 1; X0 = X.copy(); X0[:, 0] = 0
        p1 = m.predict_proba((X1 - mu) / sd)[:, 1]; p0 = m.predict_proba((X0 - mu) / sd)[:, 1]
        return float(np.average(p1 - p0, weights=wv))
    a_full = ame(w)
    a_reps = np.array([ame(senate_scale(g, x).to_numpy()) for x in REP_W])
    res['logit_M4_ame'] = a_full; res['logit_M4_ame_se'] = float(np.sqrt(brr_var(a_full, a_reps)))
    print(res, flush=True)
elif part == 'items2':
    d['risky_minus_safe'] = d[['ST166Q01HA', 'ST166Q03HA']].mean(axis=1) - d['safe_mean']
    for itm in ITEMS + ['risky_minus_safe']:
        res[itm + '_ers'] = estimate(d, M4A + ['ERS'], yvar=itm); print(itm, res[itm + '_ers'], flush=True)
    for itm in ITEMS:
        d[itm + '_ge4'] = (d[itm] >= 4).astype(float).where(d[itm].notna())
        res[itm + '_ge4'] = estimate(d, M4A, yvar=itm + '_ge4'); print(itm, res[itm + '_ge4'], flush=True)
    w = senate_scale(d, 'W_FSTUWT').to_numpy()
    for itm in ITEMS:
        ok = d[itm].notna().to_numpy()
        res[itm + '_dist'] = {int(v): wshare((d.loc[ok, itm] == v).to_numpy(), w[ok]) for v in range(1, 7)}
elif part == 'subnat':
    for name, drop in (('no_RUS_regions', ['QMR', 'QRT']), ('no_ESP', ['ESP'])):
        s = d[~d['CNT'].isin(drop)].reset_index(drop=True)
        res[name + '_n_systems'] = int(s['CNT'].nunique())
        res[name + '_M4'] = estimate(s, M4A); print(name, res[name + '_M4'], flush=True)
elif part == 'sysvars':
    for cnt, g in d.groupby('CNT'):
        ww = g['W_FSTUWT']
        res[cnt] = {'taught': float(np.average(g['taught'], weights=ww)), 'click': float(np.average(g['click'], weights=ww)),
                    'ars': float(np.average(g['ARS'], weights=ww) / 10), 'read': float(np.average(g['PV1READ'], weights=ww) * 100),
                    'oecd': int(g['OECD'].iloc[0])}
json.dump(res, open(f'results/exp_r2_{part}.json', 'w'), indent=1)
