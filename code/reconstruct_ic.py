from pathlib import Path
import sys,zipfile,json,importlib.util
import numpy as np,pandas as pd
sys.path.insert(0,str(Path('work/analysis_runtime').resolve()))
sys.path.append(r'C:\Users\SAMSUNG\Documents\Codex\2026-09-29\c-users-samsung-desktop-ai-hrs\work\step17_runtime')
import pyreadstat
B=Path(r'C:\Users\SAMSUNG\Desktop\AI\数据库论文\数据');W=Path('work/jnha_revision_1113');W.mkdir(exist_ok=True)
src=B/'代码与results/03 数据处理、主要分析、敏感性分析、绘图制表代码/数据处理/process_harmonized_charls_elsa.py'
spec=importlib.util.spec_from_file_location('hproc',src);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);mod.OUT=W
# Reconstruct the raw sensory inputs used by the harmonized CHARLS processing.
with zipfile.ZipFile(B/'2015 CHARLS Wave 3/2015 CHARLS Wave 3/CHARLS2015r.zip') as z:
 tmp=W/'charls_health_private.dta';tmp.write_bytes(z.read('Health_Status_and_Functioning.dta'))
raw,meta=pyreadstat.read_dta(str(tmp),usecols=['ID','proxy','da033','da034','da039'],apply_value_formats=False)
vision=(6-raw[['da033','da034']].where(raw[['da033','da034']].isin([1,2,3,4,5]))).mean(axis=1)
hearing=(6-raw.da039).where(raw.da039.isin([1,2,3,4,5]));raw['sensory_raw']=pd.concat([vision,hearing],axis=1).mean(axis=1).where(vision.notna()&hearing.notna())
raw[['ID','proxy','sensory_raw']].to_csv(W/'CHARLS2015_analysis_ready.csv',index=False)
(W/'charls_sensory_labels.json').write_text(json.dumps({k:meta.column_names_to_labels[k] for k in ['da033','da034','da039']},ensure_ascii=False,indent=2),encoding='utf-8')
c=mod.charls();print('CHARLS reconstructed',len(c),int(c.analysis_eligible.sum()),flush=True)
e=mod.elsa();print('ELSA reconstructed',len(e),int(e.analysis_eligible.sum()),flush=True)
h=pd.read_csv(B/'HRS处理数据/HRS2016_analysis_ready.csv',dtype={'hhidpn':'string'})
base=pd.read_csv(B/'第10步_三队列统一建模数据_2026-10-02/three_cohort_modeling_combined.csv',dtype={'participant_id':'string'})
def key(s):return s.astype('string').str.replace(r'\.0$','',regex=True).str.lstrip('0')
audit=[];summaries=[];stats=[];comp=[];groups=[]
for co,d,idcol in [('HRS2016',h,'hhidpn'),('CHARLS2015',c,'ID'),('ELSA_W8W9',e,'idauniq')]:
 d['key']=key(d[idcol]);bb=base[base.cohort.eq(co)].copy();bb['key']=key(bb.participant_id);sel=d[d.analysis_eligible].copy();joined=bb.merge(sel,on='key',suffixes=('_base','_reconstructed'),validate='one_to_one');assert len(joined)==len(bb),(co,len(joined),len(bb))
 columns=['ic_total_z','ic_mean_z']+[('ic_' if co=='HRS2016' else '')+v+'_z' for v in ['cognition','psychology','locomotion','sensory','vitality']]
 for col in columns:
  target=col if col.startswith('ic_') else 'ic_'+col
  a=bb.set_index('key')[target].sort_index();z=sel.set_index('key')[col].sort_index().reindex(a.index);diff=np.max(np.abs(a-z));assert np.array_equal(a.isna(),z.isna()) and diff<1e-7,(co,col,diff);audit.append({'cohort':co,'score':target,'matched_n':len(a),'max_abs_difference':float(diff)})
 dz=bb[['ic_cognition_z','ic_psychology_z','ic_locomotion_z','ic_sensory_z','ic_vitality_z']];pattern=dz.isna().apply(lambda row:'None' if not row.any() else '+'.join(row.index[row]),axis=1)
 for pattern_name,n in pattern.value_counts().items():comp.append({'cohort':co,'missing_domain_pattern':pattern_name,'n':int(n)})
 for group,g in bb.groupby('bp_group',sort=True):groups.append({'cohort':co,'group':group,'unweighted_n':len(g),'weighted_percent':100*g.survey_weight.sum()/bb.survey_weight.sum(),'weighted_mean_age':float(np.average(g.age_years,weights=g.survey_weight)),'weighted_mean_sbp':float(np.average(g.sbp_mean,weights=g.survey_weight)),'weighted_mean_dbp':float(np.average(g.dbp_mean,weights=g.survey_weight)),'low_ic_n':int(g.ic_total_z.le(-1).sum()),'weighted_low_ic_percent':float(100*np.average(g.ic_total_z.le(-1),weights=g.survey_weight))})
 if co=='HRS2016':
  ref=d.analysis_eligible;rawcols=['psychology_raw','locomotion_raw','sensory_raw','vitality_raw'];ddof=1;ref_total=int(ref.sum());composite_ref_n=ref_total
 else:ref=d.base_eligible_before_ic;rawcols=['psychology_capacity_raw','locomotion_capacity_raw','sensory_capacity_raw','grip_max_both_hands_kg'];ddof=0;ref_total=int(ref.sum());composite_ref_n=int((ref&d.ic_minimum_rule).sum())
 summaries.append({'cohort':co,'reference_base_n':ref_total,'composite_reference_n':composite_ref_n,'final_n':len(bb),'sd_ddof':ddof,'unweighted_standardization':True})
 for col in rawcols:
  val=d.loc[ref,col];stats.append({'cohort':co,'raw_score':col,'reference_valid_n':int(val.notna().sum()),'mean':float(val.mean()),'sd':float(val.std(ddof=ddof)),'observed_min':float(val.min()),'observed_max':float(val.max())})
 if co!='HRS2016':
  for col in [f'r{3 if co=="CHARLS2015" else 8}imrc',f'r{3 if co=="CHARLS2015" else 8}dlrc',f'r{3 if co=="CHARLS2015" else 8}orient']:
   val=d.loc[ref,col];stats.append({'cohort':co,'raw_score':col,'reference_valid_n':int(val.notna().sum()),'mean':float(val.mean()),'sd':float(val.std(ddof=ddof)),'observed_min':float(val.min()),'observed_max':float(val.max())})
pd.DataFrame(audit).to_csv(W/'score_reproduction_audit.csv',index=False,encoding='utf-8-sig');pd.DataFrame(comp).to_csv(W/'domain_missing_patterns.csv',index=False,encoding='utf-8-sig');pd.DataFrame(groups).to_csv(W/'phenotype_denominators.csv',index=False,encoding='utf-8-sig');pd.DataFrame(stats).to_csv(W/'standardization_parameters.csv',index=False,encoding='utf-8-sig');(W/'standardization_reference.json').write_text(json.dumps(summaries,indent=2),encoding='utf-8')
# Restrict detailed local reconstructions to work; only aggregate evidence is delivered.
c.to_pickle(W/'charls_reconstructed_private.pkl');e.to_pickle(W/'elsa_reconstructed_private.pkl');print('ALL SCORES REPRODUCED; AGGREGATE TABLES WRITTEN',flush=True)
