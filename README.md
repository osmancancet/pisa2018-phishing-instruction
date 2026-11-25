# Phishing instruction and adolescents' judgements of a phishing email (PISA 2018)

Code, per-run results and manuscript source for the paper by O. C. Çetlenbik and B. Öktelik.

## Data
Public PISA 2018 student questionnaire file (SPSS, `CY07_MSU_STU_QQQ.sav`) from the OECD PISA 2018 database
(https://www.oecd.org/pisa/data/2018database/). The data are not redistributed here.

## Run order
| Step | Script | Output |
|---|---|---|
| 1 | `extract_subset.py` (reads the .sav in 40,000-row chunks) | `data/pisa2018_subset.csv.gz` |
| 2 | `prepare.py` | `data/analysis.parquet` |
| 3 | `a1_descriptives.py` | `results/a1_descriptives.json` |
| 4 | `a2_lpm.py pooled`, `a2_lpm.py country`, `a2_lpm.py extra` | `results/a2_lpm_*.json` |
| 5 | `a3_predict.py loco 3` | `results/a3_loco_folds.csv`, `results/a3_loco_meta.json` |
| 6 | (`a4_heterogeneity.py 0`, causal forest; not reported, kept for the record) | `results/a4_cf_seed0.json` |
| 7 | `a5_shap.py` | `results/a5_shap.json` |
| 8 | `a6_items_turkey.py` (needs step 4) | `results/a6_items_turkey.json` |
| 9 | `extract_likert.py`, `extract_ids.py`, `prepare_r1.py` (38 agreement items for the response-style indices; exact school and student IDs) | `data/analysis_r1.parquet` |
| 10 | `exp_r1.py <part>` for part in ars, alt, school, excl, tur, items, sep, inter, sens, jerrim, country, bal, rules, turb, sens2, logit | `results/exp_r1_<part>.json` |
| 11 | `exp_r1_meta.py`, `exp_r1_meta_reml.py`, `exp_r1_predict.py`, `exp_r1_readgap.py` | meta-analyses (DerSimonian-Laird and REML with Hartung-Knapp-Sidik-Jonkman intervals), leave-systems-out prediction with system-grouped thresholds and saved predictions |
| 11b | `prepare_r2.py`, `exp_r2.py <part>` for part in sysvars, count, schfe, rules2, bal2, inter2, items2, subnat; `exp_r2_meta.py` | round-2 checks: breadth of instruction, pooled school fixed effects, outcome-defined rules under a permutation null, 12-item balanced index, same-sample logit, item checks, sub-national units, meta-regression (`results/exp_r2_*.json`) |
| 11c | `exp_r3.py <part>` for part in curve_full, curve_effort, curve_not_alllow, curve_not_allhigh, curve_consistent, curve_combined, overlap; `make_dag.py` | round-4 checks: specification curve (5 covariate sets x 4 outcomes x 6 samples, `results/exp_r3_curve_*.json`), overlap in instruction breadth (`results/exp_r3_overlap.json`), Appendix Fig. A1 |
| 12 | `make_numbers.py`, `make_tables.py` (Tables 1 to 3, A1; Figs 1 to 3), `render.py`, `build_docx.py`, `make_submission.py` | `results/numbers.json`, `results/tables.json`, figures, `manuscript/main.docx`, highlights and cover letter |

`run_all.sh` runs the whole sequence and writes SHA-256 digests of the code to `results/code_sha256.txt`.
The .sav file is expected at `data/STU/CY07_MSU_STU_QQQ.sav`.

Note: steps 1 to 8 are the first set of analyses and are unchanged. Steps 9 to 11 add the response-style indices, alternative outcomes, exclusion rules and system-grouped prediction.
The manuscript reports the prediction results of step 11 (one seed, thresholds tuned on held-out systems); step 5 (three seeds, thresholds tuned on held-out students) is kept for comparison.
`a4_heterogeneity.py` (causal forest) is kept in the repository but no longer reported: it was underpowered for student-level heterogeneity, which is now tested with pooled interaction models (`exp_r1.py inter`).
The first extraction wrote IDs with six significant digits; `extract_ids.py` restores the exact integer IDs (rows are aligned by file order, which `prepare_r1.py` checks).

Every number in the manuscript is inserted from `results/numbers.json` by `render.py`; none is typed by hand.

## Environment
Analysis: Python 3.13 with the pinned versions in `requirements.txt`. Extraction from the .sav file (steps 1 and 9) was run separately on Python 3.10 with pyreadstat 1.3.8; pandoc 3.1 and LibreOffice for the Word and PDF build.
Seeds: LightGBM seeds 0-2 per fold (step 5), seed 0 for the causal forest and SHAP model.
Runtime on a 4-core machine with 15 GB RAM: step 4 about 1.5 h (80 replicate weights x 10 plausible values), step 5 about 2 h, step 10 about 1.5 h with four parts in parallel, step 11 about 30 min.

## Licence
MIT.
