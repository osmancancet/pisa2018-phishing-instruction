"""Round-4 analyses (earlier scripts are unchanged).

The data preparation and the estimator are those of exp_r1.py: its preamble is executed here.
  curve_<sample> : specification curve. The pooled instruction coefficient for every combination of
                   covariate set (M0, M1, M2, M3, M4), binary outcome (click rated >= 4, >= 5, 6,
                   strictly above every other reaction) and one sample (full, rule 1 effort, rule 2b
                   all-low, rule 2a all-high, rule 3 click and delete both >= 4, all rules combined).
  overlap        : weighted distribution of the number of other topics reported (0-6) by instruction
                   status, and M4 within the ranges 0-2 and 3-6 other topics.
Usage: python exp_r3.py <part>
"""
import json
import sys

_src = open('code/exp_r1.py').read().split("part = sys.argv[1]")[0]
_argv = sys.argv
exec(compile(_src, 'exp_r1_preamble', 'exec'))
sys.argv = _argv

part = sys.argv[1]
res = {}
M3 = BASE + OTHER
M4A = BASE + OTHER + ARSC
COVS = {'M0': ([], False), 'M1': (BASE, False), 'M2': (BASE, True), 'M3': (M3, True), 'M4': (M4A, True)}
OUTS = ['click', 'click5', 'click6', 'click_top']

it5 = d[[f'ST166Q0{i}HA' for i in range(1, 6)]]
allhigh = it5.ge(4).all(axis=1) & it5.notna().all(axis=1)
alllow = it5.le(2).all(axis=1) & it5.notna().all(axis=1)
extreme_pat = allhigh | alllow
SAMPLES = {'full': d['taught'].notna(), 'effort': d['EFFORT1'] > 4, 'not_alllow': ~alllow,
           'not_allhigh': ~allhigh, 'consistent': d['inconsistent'] == 0,
           'combined': (d['EFFORT1'] > 4) & ~extreme_pat & (d['inconsistent'] == 0)}

if part.startswith('curve_'):
    s = part[len('curve_'):]
    g = d[SAMPLES[s]].reset_index(drop=True)
    for o in OUTS:
        for m, (cols, pv) in COVS.items():
            res[f'{s}|{o}|{m}'] = estimate(g, cols, yvar=o, uses_pv=pv)
            print(s, o, m, res[f'{s}|{o}|{m}'], flush=True)
elif part == 'overlap':
    w = senate_scale(d, 'W_FSTUWT').to_numpy()
    for t in (0, 1):
        sel = (d['taught'] == t).to_numpy()
        tot = w[sel].sum()
        for k in range(7):
            res[f'share_t{t}_n{k}'] = float(w[sel & (d['n_other_taught'] == k).to_numpy()].sum() / tot)
    for name, lo, hi in (('low', 0, 2), ('high', 3, 6)):
        g = d[(d['n_other_taught'] >= lo) & (d['n_other_taught'] <= hi)].reset_index(drop=True)
        res[f'M4_{name}'] = estimate(g, M4A)
        res[f'M4_{name}_share_taught'] = float(np.average(g['taught'], weights=senate_scale(g, 'W_FSTUWT')))
        print(name, res[f'M4_{name}'], flush=True)
json.dump(res, open(f'results/exp_r3_{part}.json', 'w'), indent=1)
