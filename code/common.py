"""Shared helpers: weights, BRR variance (Fay k=0.5, 80 replicates), Rubin's rules."""
import numpy as np
import pandas as pd

N_REP = 80
FAY = 0.5
REP_W = [f'W_FSTURWT{i}' for i in range(1, N_REP + 1)]
PV_READ = [f'PV{i}READ' for i in range(1, 11)]


def brr_var(theta_full, theta_reps):
    theta_reps = np.asarray(theta_reps)
    return np.sum((theta_reps - theta_full) ** 2, axis=0) / (N_REP * (1 - FAY) ** 2)


def rubin(estimates, variances):
    """Combine M plausible-value estimates (Rubin, 1987)."""
    est = np.asarray(estimates)
    var = np.asarray(variances)
    m = len(est)
    qbar = est.mean(axis=0)
    ubar = var.mean(axis=0)
    b = est.var(axis=0, ddof=1) if m > 1 else 0.0
    total = ubar + (1 + 1 / m) * b
    return qbar, np.sqrt(total)


def senate_scale(df, wcol, by='CNT', total=5000.0):
    """Rescale a weight so each country sums to `total` (equal country contribution)."""
    s = df.groupby(by)[wcol].transform('sum')
    return df[wcol] / s * total


def load(path='data/analysis.parquet'):
    d = pd.read_parquet(path)
    d = d.dropna(subset=['click', 'taught']).reset_index(drop=True)
    return d
