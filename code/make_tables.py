"""Build Markdown tables and figures from results/*.json (never edit numbers by hand)."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R = 'results'
OUT = 'manuscript'
N = json.load(open(f'{R}/numbers.json'))
NAMES = json.load(open('code/system_names.json'))


def pp(x, d=1):
    return f'{100 * x:.{d}f}'


tabs = {}
p = json.load(open(f'{R}/a2_lpm_pooled.json'))
ex = json.load(open(f'{R}/a2_lpm_extra.json'))
bc = json.load(open(f'{R}/a2_lpm_bycountry.json'))
ars = json.load(open(f'{R}/exp_r1_ars.json'))
alt = json.load(open(f'{R}/exp_r1_alt.json'))
sch = json.load(open(f'{R}/exp_r1_school.json'))
jer = json.load(open(f'{R}/exp_r1_jerrim.json'))
tur = json.load(open(f'{R}/exp_r1_tur.json'))
cty = json.load(open(f'{R}/exp_r1_country.json'))
bal = json.load(open(f'{R}/exp_r1_bal.json'))
rules = json.load(open(f'{R}/exp_r1_rules.json'))
turb = json.load(open(f'{R}/exp_r1_turb.json'))
items = json.load(open(f'{R}/exp_r1_items.json'))
r2 = {k: json.load(open(f'{R}/exp_r2_{k}.json')) for k in ('count', 'schfe', 'bal2', 'inter2', 'subnat')}


def cell(e, scale=100, dec=2):
    return (f"{e['b'] * scale:.{dec}f} ({e['se'] * scale:.{dec}f})", f"{e['n']:,}") if e else ('n/a', '')


# Table 1: pooled model sequence, robustness checks and Türkiye
rows = [('**A. Model sequence (outcome: click rated 4 to 6)**', None, None, None),
        ('M0', 'System fixed effects only', p['M0_none'], bc['TUR']['M0_none']),
        ('M1', '+ background', p['M1_background'], None),
        ('M2', '+ reading achievement', p['M2_plus_reading'], None),
        ('M3', '+ other online-information instruction', p['M3_plus_other_instruction'], tur['M3']),
        ('M4', '+ agreement index (main model)', ars['M4ars'], tur['M4ars']),
        ('M4b', 'M4 + extreme-response index', ars['M4ars_ers'], None),
        ('M4c', 'M3 + balanced acquiescence index (belonging items) instead of the agreement index', bal['M4bal'], None),
        ('M4d', 'M3 + balanced acquiescence index (belonging and reading self-concept, 12 items) instead of the agreement index', r2['bal2']['M4bal2'], None),
        ('M4s', 'M4 with school instead of system fixed effects', r2['schfe']['M4_schFE'], None),
        ('M4l', 'M4 as weighted logit, average marginal effect (first plausible value)', {'b': r2['inter2']['logit_M4_ame'], 'se': r2['inter2']['logit_M4_ame_se'], 'n': p['M0_none']['n']}, None),
        ('B', 'M4 + mean rating of the three safe reactions (outcome-based sensitivity specification)', ars['M4ars_safe'], turb['TUR_M4ars_safe']),
        ('**B. Alternative outcomes and exposure (M4 covariates)**', None, None, None),
        ('O1', 'Click rated 5 or 6', alt['click5_M4ars'], None),
        ('O2', 'Click rated 6', alt['click6_M4ars'], None),
        ('O3', 'Click rated strictly higher than every other reaction', alt['click_top_M4ars'], None),
        ('O4', 'METASPAM (SD units x 100; higher = better ranking)', alt['metaspam_sd_M4ars'], None),
        ('S1', 'School share taught (leave-one-out, per 10 pp)', {'b': sch['school_M4ars']['b'] / 10, 'se': sch['school_M4ars']['se'] / 10, 'n': sch['school_M4ars']['n']}, None),
        ('X7', 'Exposure: number of the seven instruction topics reported (per topic), M2 + agreement index', r2['count']['n_all7_M2ars'], None),
        ('**C. Sample restrictions (M4 unless stated)**', None, None, None),
        ('E1', 'Test effort above 4 (rule 1 only)', rules['effort_M4ars'], None),
        ('E2', 'No all-low pattern on the scenario (rule 2b only)', rules['not_alllow_M4ars'], None),
        ('E3', 'No all-high pattern on the scenario (rule 2a only; removes endorsers only)', rules['not_allhigh_M4ars'], None),
        ('E4', 'Click and delete not both 4 or above (rule 3 only; removes endorsers only)', rules['consistent_M4ars'], None),
        ('E5', 'Top tenth of each system\'s agreement index removed (with ties at the cut-off and imputed index values)', json.load(open(f'{R}/exp_r1_excl.json'))['ars_top10_dropped_M4ars'], None),
        ('E6', 'Moscow Region and Tatarstan removed (overlap with the Russian Federation)', r2['subnat']['no_RUS_regions_M4'], None),
        ('J0', 'Restricted sample (Jerrim-type), M0', jer['all_M0'], jer['by_country']['TUR']['M0']),
        ('J4', 'Restricted sample (Jerrim-type), M4', jer['all_M4ars'], jer['by_country']['TUR']['M4ars']),
        ('J0o', 'Restricted sample, OECD members only, M0', jer['oecd_M0'], None),
        ('J4o', 'Restricted sample, OECD members only, M4', jer['oecd_M4ars'], None),
        ('T1', 'M4 + programme fixed effects', None, tur['M4ars_progFE']),
        ('T2', 'M4 + school fixed effects', None, tur['M4ars_schoolFE'])]
t = ['| Model | Specification | All systems: b (SE) | n | Türkiye: b (SE) | n |', '|' + '|'.join('-' * w for w in (8, 44, 16, 10, 16, 8)) + '|']
for code, spec, a, tr in rows:
    if spec is None:
        t.append(f'| | {code} | | | | |'); continue
    fa, na = cell(a); ft, nt = cell(tr)
    if a is None:
        fa, na = '', ''
    t.append(f"| {code} | {spec} | {fa} | {na} | {ft} | {nt} |")
tabs['table1'] = '\n'.join(t)

# Table 2: item-level ratings, raw means and within-system differences (M4 covariates)
a6 = json.load(open(f'{R}/a6_items_turkey.json'))
lab = [('answer_email', 'ST166Q01HA', 'Answer the email and ask for more information (risky)'),
       ('check_sender', 'ST166Q02HA', "Check the sender's email address"),
       ('click_link', 'ST166Q03HA', 'Click the link to fill out the form (risky)'),
       ('delete_email', 'ST166Q04HA', 'Delete the email without clicking the link'),
       ('check_operator_site', 'ST166Q05HA', "Check the operator's website")]
t = ['| Reaction | All: taught | All: not | All: adj. diff. (SE) | TR: taught | TR: not | TR: adj. diff. (SE) |',
     '|' + '|'.join('-' * w for w in (34, 9, 9, 16, 9, 9, 16)) + '|']
for k, code, l in lab:
    a = a6['items']['pooled'][k]; b = a6['items']['TUR'][k]; ia = items[code]; it = items['TUR_' + code]
    t.append(f"| {l} | {a['mean_taught']:.2f} | {a['mean_not']:.2f} | {ia['b']:.3f} ({ia['se']:.3f}) | {b['mean_taught']:.2f} | {b['mean_not']:.2f} | {it['b']:.3f} ({it['se']:.3f}) |")
ra = items['risky_minus_safe']; rt = items['TUR_risky_minus_safe']
t.append(f"| Risky minus safe (mean of 2 risky minus mean of 3 safe) | | | {ra['b']:.3f} ({ra['se']:.3f}) | | | {rt['b']:.3f} ({rt['se']:.3f}) |")
tabs['table2'] = '\n'.join(t)

# Table 3: prediction (thresholds tuned on held-out training systems, seed 0)
from scipy import stats as _st
f = pd.read_csv(f'{R}/exp_r1_predict_folds.csv')
names = {'logit': 'Logistic regression', 'lgbm': 'LightGBM'}
t = ['| Model | AUC | PR-AUC | Brier | Accuracy | Precision | Recall | F1 | MCC |', '|' + '|'.join('-' * w for w in (20, 12, 12, 12, 12, 12, 12, 12, 12)) + '|']
for m in ('logit', 'lgbm'):
    g = f[f.model == m]
    cells = []
    for kk in ('auc', 'prauc', 'brier', 'acc', 'prec', 'rec', 'f1', 'mcc'):
        x = g[kk].to_numpy(); h = _st.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
        cells.append(f'{x.mean():.3f} ({h:.3f})')
    t.append(f"| {names[m]} | " + ' | '.join(cells) + ' |')
tabs['table3'] = '\n'.join(t)

# Table A1: per-system descriptives and M4 coefficient
a1 = json.load(open(f'{R}/a1_descriptives.json'))['countries']
t = ['| System | n | Endorse link % (SE) | Taught % (SE) | M4 b, pp (SE) |', '|---|---|---|---|---|']
for cnt in sorted(a1, key=lambda c: NAMES.get(c, c)):
    v = a1[cnt]
    m4 = cty.get(cnt, {}).get('M4ars')
    fm = f"{100 * m4['b']:.1f} ({100 * m4['se']:.1f})" if m4 and np.isfinite(m4['b']) else 'n/a'
    t.append(f"| {NAMES.get(cnt, cnt)} | {v['n']:,} | {pp(v['click'])} ({pp(v['click_se'])}) | {pp(v['taught'])} ({pp(v['taught_se'])}) | {fm} |")
tabs['tableA1'] = '\n'.join(t)
json.dump(tabs, open(f'{R}/tables.json', 'w'), indent=1)

# Figure 1: system-level taught vs endorse
fig, ax = plt.subplots(figsize=(6.5, 4.6))
xs = np.array([a1[c]['taught'] for c in a1]) * 100; ys = np.array([a1[c]['click'] for c in a1]) * 100
ax.scatter(xs, ys, s=18, color='#4a6fa5', alpha=0.8)
ax.scatter([a1['TUR']['taught'] * 100], [a1['TUR']['click'] * 100], s=50, color='#c0392b', zorder=3)
ax.annotate('Türkiye', (a1['TUR']['taught'] * 100, a1['TUR']['click'] * 100), xytext=(6, -12), textcoords='offset points', fontsize=9, color='#c0392b')
for c in (N['click_min_cnt'], N['click_max_cnt'], N['taught_max_cnt'], N['taught_min_cnt']):
    ax.annotate(NAMES.get(c, c), (a1[c]['taught'] * 100, a1[c]['click'] * 100), xytext=(5, 4), textcoords='offset points', fontsize=8)
ax.set_xlabel('Students taught to detect phishing or spam emails (%)')
ax.set_ylabel('Students rating "click the link" as appropriate (%)')
ax.set_xlim(15, 82); ax.set_ylim(5, 65)
ax.grid(alpha=0.25)
fig.tight_layout(); fig.savefig(f'{OUT}/fig1_systems.png', dpi=300)

# Figure 2: forest plot of per-system M4 coefficients (two panels), filled = BH q < .05
meta = json.load(open(f'{R}/exp_r1_meta.json'))['M4ars']
sig = set(meta['pos_bh']) | set(meta['neg_bh'])
items_f = sorted(((NAMES.get(c, c), v['M4ars']['b'], v['M4ars']['se'], c) for c, v in cty.items()
                  if np.isfinite(v['M4ars']['b'])), key=lambda x: -x[1])
half = (len(items_f) + 1) // 2
fig, axes = plt.subplots(1, 2, figsize=(6.5, 7.6), sharex=True)
for ax, chunk in zip(axes, (items_f[:half], items_f[half:])):
    for i, (nm, b, se_, c) in enumerate(chunk):
        col = '#c0392b' if c == 'TUR' else '#333333'
        ax.errorbar(100 * b, i, xerr=196 * se_, fmt='o', ms=3.2, color=col, ecolor=col, elinewidth=0.8,
                    mfc=col if c in sig else 'white')
    ax.axvline(0, color='grey', lw=0.8)
    ax.axvline(100 * meta['random'], color='#4a6fa5', lw=0.8, ls='--')
    ax.set_yticks(range(len(chunk))); ax.set_yticklabels([x[0] for x in chunk], fontsize=6.5)
    ax.set_ylim(half, -1)
    ax.tick_params(axis='x', labelsize=7)
fig.supxlabel('Conditional association (M4), percentage points, with 95% CI', fontsize=8)
fig.tight_layout(); fig.savefig(f'{OUT}/fig2_forest.png', dpi=300)

# Figure 3: SHAP mean |value| top 15, bars coloured by direction of association with the feature value
LAB = {'PV1READ': 'Reading achievement (PV1)', 'PV1SCIE': 'Science achievement (PV1)', 'PV1MATH': 'Mathematics achievement (PV1)',
       'METASUM': 'Metacognition: summarising', 'UNDREM': 'Metacognition: understanding and remembering',
       'HOMESCH': 'ICT use outside school for schoolwork', 'PISADIFF': 'Perceived difficulty of the PISA test',
       'COMPETE': 'Competitiveness', 'ISCEDL': 'ISCED level of current programme', 'RESILIENCE': 'Resilience',
       'WORKMAST': 'Work mastery', 'PERCOMP': 'Perceived competitiveness at school', 'DISCRIM': 'Discriminating school climate',
       'ENTUSE': 'ICT use outside school for leisure', 'MASTGOAL': 'Mastery-goal orientation', 'ESCS': 'Economic, social and cultural status',
       'ST158Q05_y': 'Taught: using search-result descriptions', 'EUDMO': 'Eudaemonia (meaning in life)', 'GRADE': 'Grade relative to modal grade',
       'EFFORT1': 'Test effort', 'AGE': 'Age', 'JOYREAD': 'Joy of reading', 'SCREADCOMP': 'Self-concept: reading competence',
       'SCREADDIFF': 'Self-concept: reading difficulty', 'GFOFAIL': 'General fear of failure', 'ATTLNACT': 'Attitude towards school: learning activities',
       'BEINGBULLIED': 'Exposure to bullying', 'COMPICT': 'Perceived ICT competence', 'AUTICT': 'Perceived autonomy related to ICT use',
       'INTICT': 'Interest in ICT', 'SOIAICT': 'ICT as a topic in social interaction', 'USESCH': 'ICT use at school', 'TMINS': 'Learning time (minutes per week)'}
s = json.load(open(f'{R}/a5_shap.json'))
top = list(s['all']['mean_abs'])[:15][::-1]
dirc = s['all'].get('direction_corr_top20', {})
cols = ['#c0392b' if dirc.get(k, 0) > 0 else '#4a6fa5' for k in top]
fig, ax = plt.subplots(figsize=(6.5, 4.8))
ax.barh(range(len(top)), [s['all']['mean_abs'][k] for k in top], color=cols)
ax.set_yticks(range(len(top))); ax.set_yticklabels([LAB.get(k, k) for k in top], fontsize=7.5)
ax.set_xlabel('Mean |SHAP value| (log-odds), country-balanced sample', fontsize=8)
fig.tight_layout(); fig.savefig(f'{OUT}/fig3_shap.png', dpi=300)
print('tables and figures written')

# Fig. 5 (round 4): specification curve over covariate set x outcome x sample
import glob, os
_cf = sorted(glob.glob(f'{R}/exp_r3_curve_*.json'))
if len(_cf) == 6:
    c = {}
    for f in _cf:
        c.update(json.load(open(f)))
    OUTSET = {'not_allhigh', 'consistent', 'combined'}
    rows = sorted(c.items(), key=lambda kv: kv[1]['b'])
    b = np.array([v['b'] for _, v in rows]) * 100
    se = np.array([v['se'] for _, v in rows]) * 100
    samp = [k.split('|')[0] for k, _ in rows]
    col = ['#c0392b' if s in OUTSET else '#1f4e79' for s in samp]
    x = np.arange(len(rows))
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.2, 6.6), sharex=True, gridspec_kw={'height_ratios': [1.15, 1.6]})
    a1.axhline(0, color='#555', lw=0.7)
    a1.vlines(x, b - 1.96 * se, b + 1.96 * se, color=col, lw=0.6, alpha=0.6)
    a1.scatter(x, b, c=col, s=7, zorder=3)
    a1.set_ylabel('Coefficient on phishing\ninstruction (pp, 95% CI)', fontsize=8)
    a1.tick_params(labelsize=7)
    from matplotlib.lines import Line2D
    a1.legend(handles=[Line2D([], [], marker='o', ls='', color='#1f4e79', ms=4, label='Full sample or rule defined outside the outcome'),
                       Line2D([], [], marker='o', ls='', color='#c0392b', ms=4, label='Rule defined on the scenario ratings')],
              fontsize=7, loc='upper left', frameon=False)
    lab_rows = [('cov', 'M0', 'M0: system FE'), ('cov', 'M1', 'M1: + background'), ('cov', 'M2', 'M2: + reading'),
                ('cov', 'M3', 'M3: + other topics'), ('cov', 'M4', 'M4: + agreement index'),
                ('out', 'click', 'Click rated 4-6'), ('out', 'click5', 'Click rated 5-6'), ('out', 'click6', 'Click rated 6'),
                ('out', 'click_top', 'Click above all others'),
                ('smp', 'full', 'Full sample'), ('smp', 'effort', 'Rule 1: effort'), ('smp', 'not_alllow', 'Rule 2b: all low'),
                ('smp', 'not_allhigh', 'Rule 2a: all high'), ('smp', 'consistent', 'Rule 3: click and delete'),
                ('smp', 'combined', 'All rules combined')]
    pos = {'smp': 0, 'out': 1, 'cov': 2}
    for i, (kind, val, lab) in enumerate(lab_rows):
        y = len(lab_rows) - 1 - i
        hit = [j for j, (k, _) in enumerate(rows) if k.split('|')[pos[kind]] == val]
        a2.scatter(hit, [y] * len(hit), c=[col[j] for j in hit], s=4, marker='|')
    a2.set_yticks(range(len(lab_rows))); a2.set_yticklabels([l for *_, l in lab_rows][::-1], fontsize=7)
    for yy in (5.5, 9.5):
        a2.axhline(yy, color='#bbb', lw=0.5)
    a2.set_xlabel('Specifications ordered by estimate', fontsize=8); a2.tick_params(axis='x', labelsize=7)
    fig.tight_layout(); fig.savefig(f'{OUT}/fig5_speccurve.png', dpi=300)
