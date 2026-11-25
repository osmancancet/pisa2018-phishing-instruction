"""Random-effects meta-analysis of per-system M4ars estimates with the REML estimator of tau^2
and the Hartung-Knapp-Sidik-Jonkman (HKSJ) interval, as a check on the DerSimonian-Laird results."""
import json
import numpy as np
from scipy import stats, optimize


def reml(b, s):
    v = s ** 2; k = len(b)
    def nll(t2):
        w = 1 / (v + t2); mu = np.sum(w * b) / np.sum(w)
        return 0.5 * (np.sum(np.log(v + t2)) + np.log(np.sum(w)) + np.sum(w * (b - mu) ** 2))
    t2 = optimize.minimize_scalar(nll, bounds=(0, 1), method='bounded', options={'xatol': 1e-12}).x
    w = 1 / (v + t2); mu = np.sum(w * b) / np.sum(w)
    q = np.sum(w * (b - mu) ** 2) / (k - 1)
    se_hk = np.sqrt(q / np.sum(w)); tq = stats.t.ppf(0.975, k - 1)
    half_pi = stats.t.ppf(0.975, k - 2) * np.sqrt(t2 + se_hk ** 2)
    return {'k': k, 'tau': float(np.sqrt(t2)), 'random': float(mu), 'hksj_se': float(se_hk),
            'hksj_ci': [float(mu - tq * se_hk), float(mu + tq * se_hk)], 'pi': [float(mu - half_pi), float(mu + half_pi)]}


c = json.load(open('results/exp_r1_country.json'))
est = {k: v['M4ars'] for k, v in c.items() if v.get('M4ars') and np.isfinite(v['M4ars']['b']) and v['M4ars']['se'] > 0}
names = sorted(est)
out = {'M4ars': reml(np.array([est[n]['b'] for n in names]), np.array([est[n]['se'] for n in names]))}
json.dump(out, open('results/exp_r1_meta_reml.json', 'w'), indent=1)
print(out)
