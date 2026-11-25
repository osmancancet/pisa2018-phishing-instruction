#!/usr/bin/env bash
# Full pipeline from the PISA 2018 student file at data/STU/CY07_MSU_STU_QQQ.sav (run from the repository root).
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=code
python code/extract_subset.py && python code/extract_likert.py && python code/extract_ids.py
python code/prepare.py && python code/prepare_r1.py
python code/a1_descriptives.py
for p in pooled country extra; do python code/a2_lpm.py $p; done
python code/a3_predict.py loco 3
python code/a5_shap.py
python code/a6_items_turkey.py
for p in ars alt school excl tur items sep inter sens jerrim country bal rules turb sens2 logit; do python code/exp_r1.py $p; done
python code/exp_r1_meta.py && python code/exp_r1_meta_reml.py
python code/exp_r1_predict.py && python code/exp_r1_readgap.py
python code/prepare_r2.py
for p in sysvars count schfe rules2 bal2 inter2 items2 subnat; do python code/exp_r2.py $p; done
python code/exp_r2_meta.py
python code/make_numbers.py && python code/make_tables.py && python code/render.py && python code/build_docx.py && python code/make_submission.py
(cd manuscript && soffice --headless --convert-to pdf main.docx)
sha256sum code/*.py code/*.sh code/*.lua > results/code_sha256.txt
