"""Round-2 additions to the analysis file (merged by row position, like prepare_r1.py).

ARS_bal2: balanced acquiescence from two scales with oppositely worded items:
sense of belonging (ST034 Q01/Q04/Q06 positive, Q02/Q03/Q05 negative) and reading
self-concept (ST161 Q01-Q03 positive, Q06-Q08 negative). Mean agreement over the four
item groups (each needs >= 2 valid items). Agreeing with both directions of the same
scale cannot be explained by the trait the scale measures.
"""
import numpy as np
import pandas as pd

a = pd.read_parquet('data/analysis_r1.parquet')
lk = pd.read_csv('data/likert_items.csv.gz')
assert len(a) == len(lk) and (a.CNT.values == lk.CNT.values).all()
L = lk.drop(columns=['CNT', 'CNTSTUID'])
L = L.where(L.isin([1, 2, 3, 4]))
groups = {'bel_pos': (['ST034Q01TA', 'ST034Q04TA', 'ST034Q06TA'], 'le2'),
          'bel_neg': (['ST034Q02TA', 'ST034Q03TA', 'ST034Q05TA'], 'le2'),
          'read_pos': (['ST161Q01HA', 'ST161Q02HA', 'ST161Q03HA'], 'ge3'),
          'read_neg': (['ST161Q06HA', 'ST161Q07HA', 'ST161Q08HA'], 'ge3')}
parts = {}
for g, (cols, rule) in groups.items():
    ag = (L[cols] <= 2) if rule == 'le2' else (L[cols] >= 3)
    ag = ag.astype(float).where(L[cols].notna())
    parts[g] = ag.mean(axis=1).where(L[cols].notna().sum(axis=1) >= 2)
P = pd.DataFrame(parts)
a['ARS_bal2'] = P.mean(axis=1).where(P.notna().all(axis=1)).values
a['ARS_bal_read'] = ((P['read_pos'] + P['read_neg']) / 2).values
a.to_parquet('data/analysis_r2.parquet')
print('ARS_bal2 missing share', float(a['ARS_bal2'].isna().mean()))
print(a.groupby('CNT')['ARS_bal2'].apply(lambda s: s.isna().mean()).sort_values().tail(5).round(3).to_dict())
