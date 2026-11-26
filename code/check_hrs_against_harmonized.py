from pathlib import Path
import sys,zipfile,json
sys.path.insert(0,str(Path('work/analysis_runtime').resolve()))
import pandas as pd,numpy as np
B=Path(r'C:\Users\SAMSUNG\Desktop\AI\数据库论文\数据');W=Path('work/jnha_revision_1113')
with zipfile.ZipFile(B/'HRS原始数据/HRS数据/H_HRS_d_stata.zip') as z:
 with z.open('H_HRS_d.dta') as f:h=pd.read_stata(f,columns=['hhidpn','r13orient','r13lgrip','r13rgrip','r13gripsum'],convert_categoricals=False)
c=pd.read_csv(W/'hrs_corrected/HRS2016_analysis_ready.csv');c=c[c.analysis_eligible];d=c.merge(h,on='hhidpn',validate='one_to_one');out=[]
for raw,official in [('cog_orientation_correct','r13orient')]:
 both=d[raw].notna()&d[official].notna();diff=d.loc[both,raw]-d.loc[both,official];out.append({'component':raw,'official':official,'both_observed_n':int(both.sum()),'official_missing_n':int(d[official].isna().sum()),'max_absolute_difference':float(diff.abs().max()),'different_n':int(diff.ne(0).sum())})
out.append({'component':'maximum grip','max_absolute_difference':float(np.nanmax(np.abs(d.vitality_raw-d[['r13lgrip','r13rgrip']].max(axis=1))))})
(W/'hrs_official_component_check.json').write_text(json.dumps(out,indent=2),encoding='utf-8');print(json.dumps(out),flush=True)
