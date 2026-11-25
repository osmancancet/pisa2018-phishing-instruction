"""A1: country-level weighted shares of 'click' and 'taught', with BRR standard errors."""
import json
import numpy as np
from common import load, brr_var, REP_W

d = load()
out = {}
for cnt, g in d.groupby('CNT'):
    row = {'n': int(len(g))}
    for v in ('click', 'taught'):
        full = np.average(g[v], weights=g['W_FSTUWT'])
        reps = [np.average(g[v], weights=g[w]) for w in REP_W]
        row[v] = float(full)
        row[v + '_se'] = float(np.sqrt(brr_var(full, reps)))
    # click rate among taught vs not taught (raw, unadjusted)
    for t in (0, 1):
        gt = g[g.taught == t]
        row[f'click_t{t}'] = float(np.average(gt.click, weights=gt['W_FSTUWT'])) if len(gt) else None
    row['oecd'] = int(g['OECD'].iloc[0])
    out[cnt] = row
# overall averages with equal country weight
vals = np.array([[r['click'], r['taught']] for r in out.values()])
summary = {'n_students': int(len(d)), 'n_systems': len(out),
           'mean_click_systems': float(vals[:, 0].mean()), 'mean_taught_systems': float(vals[:, 1].mean()),
           'corr_click_taught_systems': float(np.corrcoef(vals[:, 0], vals[:, 1])[0, 1])}
json.dump({'countries': out, 'summary': summary}, open('results/a1_descriptives.json', 'w'), indent=1)
print(summary)
print('TUR', out['TUR'])
