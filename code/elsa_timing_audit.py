from pathlib import Path
import sys,json
sys.path.insert(0,str(Path('work/analysis_runtime').resolve()))
sys.path.append(r'C:\Users\SAMSUNG\Documents\Codex\2026-09-29\c-users-samsung-desktop-ai-hrs\work\step17_runtime')
import pandas as pd,numpy as np,pyreadstat
B=Path(r'C:\Users\SAMSUNG\Desktop\AI\数据库论文\数据');W=Path('work/jnha_revision_1113')
d,_=pyreadstat.read_sav(str(B/'【数据】Harmonized ELSA/【数据】Harmonized ELSA/h_elsa_g3.sav'),usecols=['idauniq','r8iwindm','r8iwindy','r8iwnrsm','r8iwnrsy'],apply_value_formats=False)
keys=pd.read_pickle(W/'elsa_reconstructed_private.pkl');keys=keys[keys.analysis_eligible].idauniq;d=d[d.idauniq.isin(keys)].copy();assert len(d)==1536
months=(d.r8iwnrsy-d.r8iwindy)*12+(d.r8iwnrsm-d.r8iwindm);valid=months.notna()
out={'n':len(d),'interview_year_counts':d.r8iwindy.value_counts().sort_index().to_dict(),'nurse_year_counts':d.r8iwnrsy.value_counts().sort_index().to_dict(),'valid_month_intervals_n':int(valid.sum()),'interval_months_median':float(months.median()),'interval_months_Q1':float(months.quantile(.25)),'interval_months_Q3':float(months.quantile(.75)),'interval_months_min':float(months.min()),'interval_months_max':float(months.max()),'negative_intervals_n':int(months.lt(0).sum())}
(W/'elsa_timing_audit.json').write_text(json.dumps(out,indent=2),encoding='utf-8');print(json.dumps(out),flush=True)
