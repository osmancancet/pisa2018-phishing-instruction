"""Random-effects meta-analysis (DerSimonian-Laird, t-based prediction interval with k-2 df)
of per-system M0, M3 and M4ars estimates (exp_r1_country.json) and of M4ars on the Jerrim-type
restricted sample; BH/Holm; empirical-Bayes shrinkage for Türkiye."""
import json
import numpy as np
from scipy import stats


def meta(est):
    names = sorted(est); b = np.array([est[c][0] for c in names]); s = np.array([est[c][1] for c in names])
    ok = np.isfinite(b) & np.isfinite(s) & (s > 0)
    names = [n for n, k in zip(names, ok) if k]; b = b[ok]; s = s[ok]; k = len(b)
    w = 1 / s ** 2; fe = np.sum(w * b) / np.sum(w); Q = np.sum(w * (b - fe) ** 2)
    tau2 = max(0.0, (Q - (k - 1)) / (np.sum(w) - np.sum(w ** 2) / np.sum(w)))
    wr = 1 / (s ** 2 + tau2); re = np.sum(wr * b) / np.sum(wr); se = np.sqrt(1 / np.sum(wr))
    tq = stats.t.ppf(0.975, k - 2); half = tq * np.sqrt(tau2 + se ** 2)
    p = 2 * stats.norm.sf(np.abs(b / s)); order = np.argsort(p)
    bh = np.empty(k); prev = 1.0
    for r, i in enumerate(order[::-1]):
        prev = min(prev, p[i] * k / (k - r)); bh[i] = prev
    holm = np.empty(k); run = 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, p[i] * (k - r))); holm[i] = run
    out = {'k': k, 'random': float(re), 'random_se': float(se), 'tau': float(np.sqrt(tau2)), 'Q': float(Q),
           'Q_p': float(stats.chi2.sf(Q, k - 1)), 'I2': float(max(0, (Q - (k - 1)) / Q)),
           'pi': [float(re - half), float(re + half)],
           'pos_bh': sorted([n for n, x, q in zip(names, b, bh) if q < .05 and x > 0]),
           'neg_bh': sorted([n for n, x, q in zip(names, b, bh) if q < .05 and x < 0]),
           'n_pos_holm': int(np.sum((holm < .05) & (b > 0))), 'n_pos_raw': int(np.sum((p < .05) & (b > 0))), 'n_neg_raw': int(np.sum((p < .05) & (b < 0)))}
    if 'TUR' in names:
        i = names.index('TUR'); B = tau2 / (tau2 + s[i] ** 2)
        out['TUR'] = {'b': float(b[i]), 'se': float(s[i]), 'p': float(p[i]), 'bh': float(bh[i]), 'holm': float(holm[i]),
                      'rank_desc': int(1 + np.sum(b > b[i])), 'eb': float(re + B * (b[i] - re)),
                      'eb_se': float(np.sqrt(B * s[i] ** 2 + (1 - B) ** 2 * se ** 2))}
    return out


c = json.load(open('results/exp_r1_country.json'))
a2 = json.load(open('results/a2_lpm_bycountry.json'))
j = json.load(open('results/exp_r1_jerrim.json'))['by_country']
res = {'M0': meta({k: (a2[k]['M0_none']['b'], a2[k]['M0_none']['se']) for k in a2}),
       'M3': meta({k: (v['M3']['b'], v['M3']['se']) for k, v in c.items()}),
       'M4ars': meta({k: (v['M4ars']['b'], v['M4ars']['se']) for k, v in c.items()}),
       'M4safe': meta({k: (a2[k]['M4_plus_response_style']['b'], a2[k]['M4_plus_response_style']['se']) for k in a2}),
       'jerrim_M4ars': meta({k: (v['M4ars']['b'], v['M4ars']['se']) for k, v in j.items() if v['n'] >= 500})}
for k, v in res.items():
    print(k, {a: (round(x, 4) if isinstance(x, float) else x) for a, x in v.items()})
json.dump(res, open('results/exp_r1_meta.json', 'w'), indent=1)
