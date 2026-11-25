"""Extract raw 4-point agree/disagree items used for the external acquiescence index."""
import gzip
import pyreadstat

SAV = 'data/STU/CY07_MSU_STU_QQQ.sav'
_, meta = pyreadstat.read_sav(SAV, metadataonly=True)
BLOCKS = ('ST034', 'ST036', 'ST160', 'ST161', 'ST181', 'ST182', 'ST183', 'ST185', 'ST188')
cols = ['CNT', 'CNTSTUID'] + [c for c in meta.column_names if c.startswith(BLOCKS)]
print(cols, flush=True)
with gzip.open('data/likert_items.csv.gz', 'wt') as fh:
    for i, (df, _) in enumerate(pyreadstat.read_file_in_chunks(pyreadstat.read_sav, SAV, chunksize=60000, usecols=cols)):
        df.to_csv(fh, index=False, header=(i == 0), float_format='%.6g')
        print('chunk', i, flush=True)
print('done', flush=True)
