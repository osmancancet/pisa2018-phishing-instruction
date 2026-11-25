"""A3: how well do student characteristics predict endorsing the phishing link?

Leave-country-out (GroupKFold by system, 5 folds) and within-country stratified
5-fold CV. Models: majority class, logistic regression, LightGBM (10 seeds).
Items of the phishing scenario (ST166) and METASPAM are excluded as features.
Senate-scaled weights in fitting and evaluation. Decision threshold chosen to
maximise F1 on an inner 20% validation split of the training fold.
"""
import json
import sys
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold, StratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             matthews_corrcoef, roc_auc_score, average_precision_score,
                             brier_score_loss, confusion_matrix, precision_recall_curve)
from common import load, senate_scale

EXCLUDE_PREFIX = ('ST166', 'W_', 'PV', 'CNT', 'ST158Q0', 'LANG', 'COBN', 'OCOD', 'VER')
EXCLUDE = {'click', 'click_ips', 'METASPAM', 'OECD', 'ADMINMODE', 'WVARSTRR', 'test', 'SENWT',
           'ST004D01T', 'PROGN', 'ISCEDD', 'ISCEDO', 'BMMJ1', 'BFMJ2', 'BSMJ', 'read_pv_mean',
           'EFFORT2', 'n_other_taught', 'safe_mean', 'inconsistent',
           # parent-questionnaire, financial-literacy and well-being options (few systems) and duplicates
           'CURSUPP', 'EMOSUPP', 'PQSCHOOL', 'PASCHPOL', 'PRESUPP', 'JOYREADP', 'ATTIMMP', 'INTCULTP',
           'GCAWAREP', 'FLCONFIN', 'FLCONICT', 'FLSCHOOL', 'FLFAMILY', 'BODYIMA', 'SOCONPA', 'STUBMI',
           'MISCED_D', 'FISCED_D', 'HISCED_D', 'PAREDINT', 'MMINS', 'LMINS', 'SMINS', 'FCFMLRTY'}


def features(d):
    cols = [c for c in d.columns if not c.startswith(EXCLUDE_PREFIX) and c not in EXCLUDE]
    cols += [f'ST158Q0{i}_y' for i in range(1, 7)] + ['PV1READ', 'PV1MATH', 'PV1SCIE']
    cols = [c for c in dict.fromkeys(cols) if c in d.columns and d[c].dtype != object]
    return cols


def best_threshold(y, p, w):
    prec, rec, thr = precision_recall_curve(y, p, sample_weight=w)
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-12, None)
    return float(thr[np.nanargmax(f1[:-1])])


def metrics(y, p, thr, w):
    yhat = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat, sample_weight=w).ravel()
    return dict(acc=accuracy_score(y, yhat, sample_weight=w), prec=precision_score(y, yhat, sample_weight=w, zero_division=0),
                rec=recall_score(y, yhat, sample_weight=w), f1=f1_score(y, yhat, sample_weight=w),
                mcc=matthews_corrcoef(y, yhat, sample_weight=w), auc=roc_auc_score(y, p, sample_weight=w),
                prauc=average_precision_score(y, p, sample_weight=w), brier=brier_score_loss(y, p, sample_weight=w),
                tn=tn, fp=fp, fn=fn, tp=tp, thr=thr)


def lgbm(seed):
    return lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=63, min_child_samples=200,
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                              random_state=seed, n_jobs=4, verbose=-1)


def logit():
    return make_pipeline(SimpleImputer(strategy='median', add_indicator=True), StandardScaler(),
                         LogisticRegression(max_iter=2000, C=1.0))


def run(scheme, seeds):
    d = load()
    d = d[d['CNT'] != 'VNM'].reset_index(drop=True)  # no reading plausible values
    X = d[features(d)].astype(float)
    y = d['click'].astype(int).to_numpy()
    w = senate_scale(d, 'W_FSTUWT').to_numpy()
    if scheme == 'loco':
        splits = list(GroupKFold(n_splits=5).split(X, y, groups=d['CNT']))
    else:
        strat = d['CNT'] + '_' + d['click'].astype(int).astype(str)
        splits = list(StratifiedKFold(5, shuffle=True, random_state=0).split(X, strat))
    rows = []
    for k, (tr, te) in enumerate(splits):
        itr, ival = train_test_split(tr, test_size=0.2, random_state=k, stratify=y[tr])
        # majority baseline: predicts training prevalence
        prev = np.average(y[tr], weights=w[tr])
        rows.append(dict(model='majority', fold=k, seed=0, **metrics(y[te], np.full(len(te), prev), 0.5, w[te])))
        lr = logit().fit(X.iloc[itr], y[itr], logisticregression__sample_weight=w[itr])
        thr = best_threshold(y[ival], lr.predict_proba(X.iloc[ival])[:, 1], w[ival])
        rows.append(dict(model='logit', fold=k, seed=0, **metrics(y[te], lr.predict_proba(X.iloc[te])[:, 1], thr, w[te])))
        for s in seeds:
            m = lgbm(s).fit(X.iloc[itr], y[itr], sample_weight=w[itr])
            thr = best_threshold(y[ival], m.predict_proba(X.iloc[ival])[:, 1], w[ival])
            rows.append(dict(model='lgbm', fold=k, seed=s, **metrics(y[te], m.predict_proba(X.iloc[te])[:, 1], thr, w[te])))
            print(scheme, k, s, rows[-1]['auc'], rows[-1]['mcc'], flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(f'results/a3_{scheme}_folds.csv', index=False)
    json.dump({'features': features(d), 'n': int(len(d))}, open(f'results/a3_{scheme}_meta.json', 'w'), indent=1)


if __name__ == '__main__':
    run(sys.argv[1], seeds=list(range(int(sys.argv[2]) if len(sys.argv) > 2 else 3)))
