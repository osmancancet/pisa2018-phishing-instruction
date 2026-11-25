"""Extract the analysis subset from the PISA 2018 student questionnaire file."""
import re
import pyreadstat

SAV = 'data/STU/CY07_MSU_STU_QQQ.sav'
_, meta = pyreadstat.read_sav(SAV, metadataonly=True)
cols = meta.column_names
keep = [c for c in cols if re.match(
    r'^(CNT|CNTRYID|CNTSCHID|CNTSTUID|OECD|ADMINMODE|ST004D01T|ST158Q0\dHA|ST166Q0\dHA|'
    r'EFFORT\d|GRADE|AGE|IMMIG|REPEAT|W_FSTUWT|W_FSTURWT\d+|SENWT|PV\d+(READ|MATH|SCIE))$', c)]
item = re.compile(r'^(ST|IC|EC|PA|WB|FL|TC|SC)\d{3}')
derived = [c for c in cols if not item.match(c) and c not in keep
           and not c.startswith(('W_', 'PV', 'UNIT', 'BOOK', 'OPTION', 'CYC', 'NatCen',
                                 'STRATUM', 'SUBNATIO', 'REGION', 'VER_'))]
keep += derived
print(len(keep), derived, flush=True)
import gzip
n = 0
with gzip.open('data/pisa2018_subset.csv.gz', 'wt') as fh:
    for i, (df, _) in enumerate(pyreadstat.read_file_in_chunks(
            pyreadstat.read_sav, SAV, chunksize=40000, usecols=keep)):
        df.to_csv(fh, index=False, header=(i == 0), float_format='%.6g')
        n += len(df)
        print('chunk', i, n, flush=True)
print(n, 'done', flush=True)
