"""Build the analysis file from the PISA 2018 student questionnaire subset.

Outcome: ST166Q03HA ("Click on the link to fill out the form as soon as possible"),
rated 1-6; ratings 4-6 = judged appropriate (Jerrim, 2023 coding).
Treatment: ST158Q07HA ("Taught at school: How to detect phishing or spam emails"), 1 = Yes.
"""
import sys
import numpy as np
import pandas as pd

SRC = sys.argv[1] if len(sys.argv) > 1 else 'data/pisa2018_subset.csv.gz'
OUT = sys.argv[2] if len(sys.argv) > 2 else 'data/analysis.parquet'

df = pd.read_csv(SRC, low_memory=False)
print('raw', df.shape)
for c in df.columns:
    if df[c].dtype == object and c not in ('CNT',):
        df[c] = pd.to_numeric(df[c], errors='coerce')

df['click'] = np.where(df['ST166Q03HA'].isna(), np.nan, (df['ST166Q03HA'] >= 4).astype(float))
df['taught'] = df['ST158Q07HA'].map({1: 1.0, 2: 0.0})
other = [f'ST158Q0{i}HA' for i in range(1, 7)]
for c in other:
    df[c.replace('HA', '_y')] = df[c].map({1: 1.0, 2: 0.0})
df['n_other_taught'] = df[[c.replace('HA', '_y') for c in other]].sum(axis=1, min_count=6)
df['female'] = df['ST004D01T'].map({1: 1.0, 2: 0.0})
df['read_pv_mean'] = df[[f'PV{i}READ' for i in range(1, 11)]].mean(axis=1)
df.to_parquet(OUT, index=False)
print('saved', df.shape)
