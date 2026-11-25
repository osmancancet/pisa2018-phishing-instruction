"""Meta-regression of per-system M4ars estimates (exp_r1_country.json) on system characteristics
(exp_r2_sysvars.json): share reporting instruction, endorsement rate, mean agreement index, mean
reading score, OECD membership. REML tau^2 with Knapp-Hartung standard errors (t, k - p df).
Also the random-effects mean without the two Russian regions (Moscow Region, Tatarstan)."""
import json
import numpy as np
from scipy import stats, optimize


def reml_mr(b, s, X):
    v = s ** 2; k, p = X.shape

    def fit(t2):
        W = 1 / (v + t2); XtW = X.T * W
        V = np.linalg.inv(XtW @ X); beta = V @ XtW @ b
        return W, V, beta

    def nll(t2):
        W, V, beta = fit(t2); r = b - X @ beta
        return 0.5 * (np.sum(np.log(v + t2)) + np.linalg.slogdet(np.linalg.inv(V))[1] + np.sum(W * r ** 2))
    t2 = optimize.minimize_scalar(nll, bounds=(0, 1), method='bounded', options={'xatol': 1e-12}).x
    W, V, beta = fit(t2); r = b - X @ beta
    q = max(1.0, np.sum(W * r ** 2) / (k - p))  # Knapp-Hartung scale, truncated at 1
    se = np.sqrt(np.diag(V) * q); t = beta / se
    pv = 2 * stats.t.sf(np.abs(t), k - p)
    return {'k': int(k), 'tau': float(np.sqrt(t2)), 'beta': beta.tolist(), 'se': se.tolist(), 'p': pv.tolist()}


c = json.load(open('results/exp_r1_country.json'))
sv = json.load(open('results/exp_r2_sysvars.json'))
names = sorted(n for n in c if n in sv and np.isfinite(c[n]['M4ars']['b']) and c[n]['M4ars']['se'] > 0)
b = np.array([c[n]['M4ars']['b'] for n in names]); s = np.array([c[n]['M4ars']['se'] for n in names])
col = {k: np.array([sv[n][k] for n in names], float) for k in ('taught', 'click', 'ars', 'read', 'oecd')}
one = np.ones(len(names))
out = {'tau_intercept_only': reml_mr(b, s, one[:, None])['tau']}
# moderators in natural units: per 10 percentage points of share taught / endorsing, per 0.1 of agreement index, per 100 reading points
scaled = {'taught': col['taught'] * 10, 'click': col['click'] * 10, 'ars': col['ars'] * 10, 'read': col['read'] / 100, 'oecd': col['oecd']}
for m in scaled:
    out[m] = reml_mr(b, s, np.column_stack([one, scaled[m] - scaled[m].mean()]))
out['taught_click_ars'] = reml_mr(b, s, np.column_stack([one] + [scaled[m] - scaled[m].mean() for m in ('taught', 'click', 'ars')]))
out['corr_b_taught'] = float(np.corrcoef(b, col['taught'])[0, 1])
keep = np.array([n not in ('QMR', 'QRT') for n in names])
out['re_mean_no_RUS_regions'] = reml_mr(b[keep], s[keep], one[keep][:, None])
json.dump(out, open('results/exp_r2_meta.json', 'w'), indent=1)
for k, v in out.items():
    print(k, v)
