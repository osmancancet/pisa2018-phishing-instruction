"""Fill {{key|fmt}} placeholders from results/numbers.json and tables, then build the manuscript.

Formats: int, pct0, pct1, pp1, pp2, negpp1 (abs value x100), neg2 (abs value), f1, f2, f3, name, ordinal, raw.
"""
import json
import re
import subprocess
import sys

N = json.load(open('results/numbers.json'))
T = json.load(open('results/tables.json'))
NAMES = json.load(open('code/system_names.json'))
ORD = {1: 'first', 2: 'second', 3: 'third', 4: 'fourth', 5: 'fifth'}


def fmt(v, f):
    if f == 'int':
        return f'{int(round(v)):,}'
    if f == 'pct0':
        return f'{100 * v:.0f}'
    if f == 'pct2':
        return f'{100 * v:.2f}'
    if f == 'pct1':
        return f'{100 * v:.1f}'
    if f == 'pp1':
        return f'{100 * v:.1f}'
    if f == 'pp2':
        return f'{100 * v:.2f}'
    if f == 'negpp1':
        return f'{abs(100 * v):.1f}'
    if f == 'neg2':
        return f'{abs(v):.2f}'
    if f in ('f1', 'f2', 'f3'):
        return f'{v:.{f[1]}f}'
    if f == 'name':
        return NAMES.get(v, v)
    if f == 'ordinal':
        return ORD.get(int(v), f'{int(v)}th')
    if f == 'raw':
        return str(v)
    raise ValueError(f)


def fill(text):
    text = re.sub(r'\{\{TABLE:(\w+)\}\}', lambda m: T[m.group(1)], text)
    return re.sub(r'\{\{(\w+)\|(\w+)\}\}', lambda m: fmt(N[m.group(1)], m.group(2)), text)


parts = ['00_front', '00_abstract', '01_introduction', '02_background', '03_methods', '04_results', '05_discussion', '06_conclusion', '07_declarations']
body = '\n\n'.join(open(f'manuscript/sections/{p}.md').read() for p in parts)
body += '\n\n# Appendix {-}\n\nTable: Table A1. Students in the descriptive sample (n), weighted shares (%) with BRR standard errors, and the system-specific M4 coefficient (percentage points; regression sample, which is slightly smaller because of missing covariates). Viet Nam has no M4 estimate because its reading plausible values are not reported.\n\n{{TABLE:tableA1}}\n'
body += '\n\n# References {-}\n\n::: {#refs}\n:::\n'
out = fill(body)
left = re.findall(r'\{\{[^}]*\}\}', out)
assert not left, left
open('manuscript/main.md', 'w').write(out)
print('rendered', len(out.split()), 'words')
