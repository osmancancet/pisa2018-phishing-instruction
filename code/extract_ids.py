"""Extract exact integer school and student IDs (the earlier CSV used 6 significant digits)."""
import gzip
import pyreadstat
SAV = 'data/STU/CY07_MSU_STU_QQQ.sav'
with gzip.open('data/ids.csv.gz', 'wt') as fh:
    for i, (df, _) in enumerate(pyreadstat.read_file_in_chunks(pyreadstat.read_sav, SAV, chunksize=100000, usecols=['CNT', 'CNTSCHID', 'CNTSTUID'])):
        df['CNTSCHID'] = df['CNTSCHID'].astype('int64'); df['CNTSTUID'] = df['CNTSTUID'].astype('int64')
        df.to_csv(fh, index=False, header=(i == 0))
        print('chunk', i, flush=True)
