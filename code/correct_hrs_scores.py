from pathlib import Path
import sys,importlib.util,contextlib,json
sys.path.insert(0,str(Path('work/analysis_runtime').resolve()))
import pandas as pd,numpy as np
B=Path(r'C:\Users\SAMSUNG\Desktop\AI\数据库论文\数据');W=Path('work/jnha_revision_1113');O=W/'hrs_corrected';O.mkdir(exist_ok=True)
src=B/'代码与results/03 数据处理、主要分析、敏感性分析、绘图制表代码/数据处理/hrs2016_processing.py'
spec=importlib.util.spec_from_file_location('h_correct',src);mod=importlib.util.module_from_spec(spec);exec(src.read_text(encoding='utf-8').replace('OUTPUT_DIR.mkdir(parents=True, exist_ok=True)',''),mod.__dict__);mod.OUTPUT_DIR=O;mod.WORK_DIR=O
with (W/'corrected_hrs_processing_log.txt').open('w',encoding='utf-8') as log,contextlib.redirect_stdout(log):mod.main()
old=pd.read_csv(B/'第10步_三队列统一建模数据_2026-10-02/three_cohort_modeling_combined.csv',dtype={'participant_id':'string','survey_psu':'string'})
h=pd.read_csv(O/'HRS2016_analysis_ready.csv',dtype={'hhidpn':'string'});h=h[h.analysis_eligible].copy()
def key(s):return s.astype('string').str.replace(r'\.0$','',regex=True).str.lstrip('0')
old['key']=key(old.participant_id);h['key']=key(h.hhidpn)
assert set(h.key)==set(old.loc[old.cohort.eq('HRS2016'),'key'])
rows=[];ref=h.set_index('key')
for field in ['ic_total_z','ic_mean_z','ic_cognition_z','ic_psychology_z','ic_locomotion_z','ic_sensory_z','ic_vitality_z','ic_five_domains_complete']:
 mask=old.cohort.eq('HRS2016');before=old.loc[mask,field].to_numpy();after=old.loc[mask,'key'].map(ref[field]).to_numpy();old.loc[mask,field]=after
 if field!='ic_five_domains_complete':rows.append({'field':field,'max_absolute_change':float(np.nanmax(np.abs(after-before))),'correlation':float(np.corrcoef(before[np.isfinite(before)&np.isfinite(after)],after[np.isfinite(before)&np.isfinite(after)])[0,1])})
mask=old.cohort.eq('HRS2016');old.loc[mask,'grip_max_both_hands_kg']=old.loc[mask,'key'].map(ref.vitality_raw)
old.drop(columns='key').to_csv(W/'corrected_modeling_combined_private.csv',index=False)
(W/'hrs_score_changes.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
print('CORRECTED HRS BUILT',len(h),'sample keys unchanged',flush=True)
print(json.dumps(rows),flush=True)
