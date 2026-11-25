"""Additional data (merged by row position; all files come from the same .sav in file order).

(1) Exact integer school and student IDs (the first extraction wrote 6 significant digits,
    which collapsed IDs; nothing in the original analyses used them).
(2) External acquiescence (ARS) and extreme-response (ERS) indices from 38 four-point
    agree/disagree items in nine scales unrelated to phishing (ST034, ST036, ST160, ST161,
    ST181, ST182, ST183, ST185, ST188). ST034 and ST036 are coded 1 = strongly agree, the
    others 1 = strongly disagree. ARS = share of agree/strongly-agree answers;
    ERS = share of the two extreme categories. Requires >= 10 valid items.
"""
import numpy as np
import pandas as pd

a = pd.read_parquet('data/analysis.parquet')
ids = pd.read_csv('data/ids.csv.gz')
lk = pd.read_csv('data/likert_items.csv.gz')
assert len(a) == len(ids) == len(lk)
assert (a.CNT.values == ids.CNT.values).all() and (a.CNT.values == lk.CNT.values).all()
assert np.allclose(a.CNTSTUID.values, ids.CNTSTUID.values, rtol=1e-5)
a['CNTSCHID'] = ids.CNTSCHID.values
a['CNTSTUID'] = ids.CNTSTUID.values
items = [c for c in lk.columns if c.startswith('ST')]
rev = [c for c in items if c.startswith(('ST034', 'ST036'))]
L = lk[items].where(lk[items].isin([1, 2, 3, 4]))
agree = L.copy()
agree[[c for c in items if c not in rev]] = (L[[c for c in items if c not in rev]] >= 3).where(L[[c for c in items if c not in rev]].notna())
agree[rev] = (L[rev] <= 2).where(L[rev].notna())
extreme = L.isin([1, 4]).where(L.notna())
nvalid = L.notna().sum(axis=1)
a['ars_items'] = nvalid.values
a['ARS'] = np.where(nvalid >= 10, agree.astype(float).mean(axis=1), np.nan)
a['ERS'] = np.where(nvalid >= 10, extreme.astype(float).mean(axis=1), np.nan)
# balanced acquiescence from the sense-of-belonging scale: three positively and three negatively worded items
# (ST034Q01/Q04/Q06 positive; Q02/Q03/Q05 negative). Agreeing with both sets cannot be explained by belonging.
pos = ['ST034Q01TA', 'ST034Q04TA', 'ST034Q06TA']; neg = ['ST034Q02TA', 'ST034Q03TA', 'ST034Q05TA']
ag_pos = agree[pos].astype(float).mean(axis=1, skipna=True).where(L[pos].notna().sum(axis=1) >= 2)
ag_neg = agree[neg].astype(float).mean(axis=1, skipna=True).where(L[neg].notna().sum(axis=1) >= 2)
a['ARS_bal'] = ((ag_pos + ag_neg) / 2).values
a.to_parquet('data/analysis_r1.parquet')
print(len(items), 'items;', 'ARS missing share', float(np.isnan(a.ARS).mean()))
print(a.groupby('CNT').CNTSCHID.nunique().describe())
print(a[['ARS', 'ERS']].describe())
print('countries with all-missing ARS:', a.groupby('CNT').ARS.apply(lambda s: s.isna().mean()).sort_values().tail(8).round(3).to_dict())
