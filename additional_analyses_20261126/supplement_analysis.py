from pathlib import Path
import os,sys,ast,json,hashlib
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
# Install dependencies in your analysis environment before running.
import numpy as np,pandas as pd,scipy,pyreadstat,patsy
from scipy import stats,optimize
from patsy import dmatrix

ROOT=Path(__file__).resolve().parents[1]
OLD=Path(os.environ['JNHA_PRIOR_WORK'])
DATA=Path(os.environ['JNHA_COHORT_DATA_ROOT'])
OUT=Path(os.environ['JNHA_OUTPUT_DIR'])
ARCHIVE=Path(os.environ['JNHA_ARCHIVE_ROOT'])
OUT.mkdir(parents=True,exist_ok=True)
config={'status':'exploratory post-review sensitivity analyses','primary_time_window_months':6,'secondary_time_window_months':12,'cohorts_refitted':['CHARLS2015','ELSA_W8W9'],'HRS':'archived aggregate estimates only; no HRS participant-level fitting','uniform_score':'all five domains standardized using final eligible sample; grip standardized within sex; available-domain mean standardized again within cohort; ddof=1','missing_covariates':'reuse archived 20 completed covariate datasets, no outcome imputation','domain_MI_scope':'available outcome records with existing covariate imputations; does not refit domain-specific imputation models','python':sys.version,'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__,'patsy':patsy.__version__,'pyreadstat':pyreadstat.__version__}
(OUT/'analysis_specification.json').write_text(json.dumps(config,indent=2),encoding='utf8')

# Import only pure numerical function definitions, never execute archived data-loading statements.
source=(ARCHIVE/'code/sensitivity_corrected.py').read_text(encoding='utf-8-sig')
tree=ast.parse(source)
definitions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['rubin','meta','cluster_variance']]
exec(compile(ast.fix_missing_locations(ast.Module(body=definitions,type_ignores=[])),'archived_numerical_functions','exec'))
assert len(definitions)==3

def read_permitted(path):
    frames=[]
    for chunk in pd.read_csv(path,dtype={'participant_id':'string','survey_psu':'string'},chunksize=2000):
        permitted=chunk.loc[chunk.cohort.isin(['CHARLS2015','ELSA_W8W9'])].copy()
        if len(permitted):frames.append(permitted)
    return pd.concat(frames,ignore_index=True)

b=read_permitted(OLD/'corrected_modeling_combined_private.csv')
im=read_permitted(OLD/'corrected_imputations_private.csv.gz')
assert set(b.cohort)=={'CHARLS2015','ELSA_W8W9'}
assert len(b)==2366 and len(im)==2366*20
def key(s):return s.astype('string').str.replace(r'\.0$','',regex=True).str.strip()
b.participant_id=key(b.participant_id);im.participant_id=key(im.participant_id)
TARGETS=[x for x in im if x not in ['cohort','imputation','participant_id']]
DOM=['ic_cognition_z','ic_psychology_z','ic_locomotion_z','ic_sensory_z','ic_vitality_z']
COV='age_per10 + female + C(education_level, Treatment(reference=1)) + partnered + current_smoker + current_alcohol + diabetes + heart_disease + stroke + chronic_lung_disease + cancer'
FORM="C(bp_group, Treatment(reference='A')) + "+COV

timing,_=pyreadstat.read_sav(str(DATA/'【数据】Harmonized ELSA/【数据】Harmonized ELSA/h_elsa_g3.sav'),usecols=['idauniq','r8iwindm','r8iwindy','r8iwnrsm','r8iwnrsy'],apply_value_formats=False)
timing['participant_id']=key(timing.idauniq)
timing['interval_months']=(timing.r8iwnrsy-timing.r8iwindy)*12+timing.r8iwnrsm-timing.r8iwindm
assert not timing.participant_id.duplicated().any()
intervals=timing.set_index('participant_id').interval_months
mask=b.cohort.eq('ELSA_W8W9');b.loc[mask,'interval_months']=b.loc[mask,'participant_id'].map(intervals)
assert b.loc[mask,'interval_months'].notna().all()

def fit(d,outcome,extra=''):
    xd=dmatrix(FORM+extra,d,return_type='dataframe',eval_env=-1)
    x=xd.to_numpy();y=d[outcome].to_numpy(float);w=d.survey_weight.to_numpy(float);w=w/w.mean()
    assert len(x)==len(d) and np.linalg.matrix_rank(x)==x.shape[1]
    bread=np.linalg.pinv(x.T@(w[:,None]*x),rcond=1e-12)
    beta=bread@(x.T@(w*y));scores=x*(w*(y-x@beta))[:,None]
    j=next(j for j,s in enumerate(xd.columns) if '[T.B]' in s)
    kind='hc1' if d.cohort.iloc[0]=='CHARLS2015' else 'reverse_codes'
    influence=scores@bread[:,j]
    if kind=='reverse_codes':
        # Survey subpopulation analysis retains all original design units with
        # zero contributions for records outside the fitted domain.
        indexed=pd.Series(influence,index=d.participant_id)
        full=GLOBAL_DESIGN.participant_id.map(indexed).fillna(0).to_numpy()
        variance,design=cluster_variance(full,GLOBAL_DESIGN,kind,x.shape[1])
    else:
        variance,design=cluster_variance(influence,d,kind,x.shape[1])
    return beta[j],variance,design

rows=[];counts=[];paired=[];scaling=[]
for co,base in b.groupby('cohort',sort=False):
    base=base.reset_index(drop=True)
    GLOBAL_DESIGN=base
    scaled=base[DOM].copy()
    for col in DOM:
        if col=='ic_vitality_z':
            for sex in sorted(base.female.dropna().unique()):
                select=base.female.eq(sex);v=base.loc[select,col];scaled.loc[select,col]=(v-v.mean())/v.std(ddof=1)
        else:
            v=base[col];scaled[col]=(v-v.mean())/v.std(ddof=1)
        scaling.append({'cohort':co,'domain':col,'observed_n':int(base[col].notna().sum()),'old_mean':float(base[col].mean()),'old_sd':float(base[col].std(ddof=1)),'rescaled_mean':float(scaled[col].mean()),'rescaled_sd':float(scaled[col].std(ddof=1))})
    raw=scaled.mean(axis=1);base['uniform_ic']=(raw-raw.mean())/raw.std(ddof=1)
    assert base['uniform_ic'].notna().all()
    specs=[('benchmark_original','ic_total_z',None,''),('uniform_final_reference_equal_domain_scaling','uniform_ic',None,'')]
    if co=='ELSA_W8W9':
        specs += [('ELSA_interval_le6','ic_total_z',6,''),('ELSA_interval_le12','ic_total_z',12,''),('ELSA_interval_adjusted','ic_total_z',None,' + I(interval_months / 12)'),('ELSA_vitality_interval_le6','ic_vitality_z',6,''),('ELSA_vitality_interval_le12','ic_vitality_z',12,'')]
        for group,g in base.groupby('bp_group'):
            counts.append({'cohort':co,'analysis':'interval_distribution','bp_group':group,'n':len(g),'median_months':g.interval_months.median(),'Q1_months':g.interval_months.quantile(.25),'Q3_months':g.interval_months.quantile(.75)})
    specs += [('domain_MI_'+col,col,None,'') for col in DOM]
    ci=im.loc[im.cohort.eq(co)]
    for label,outcome,cut,extra in specs:
        select=base[outcome].notna()
        if cut is not None:select &= base.interval_months.between(0,cut)
        selected=base.loc[select].copy()
        counts.extend({'cohort':co,'analysis':label,'bp_group':g,'n':int(selected.bp_group.eq(g).sum())} for g in 'ABCDE')
        qs=[];us=[];design=None
        for m in range(1,21):
            d=selected.drop(columns=TARGETS).merge(ci.loc[ci.imputation.eq(m),['participant_id']+TARGETS],on='participant_id',validate='one_to_one')
            d['age_per10']=(d.age_years-65)/10
            q,u,design=fit(d,outcome,extra);qs.append(q);us.append(u)
        result={'cohort':co,'analysis':label,'outcome':outcome,'n':len(selected),'n_A':int(selected.bp_group.eq('A').sum()),'n_B':int(selected.bp_group.eq('B').sum()),'clusters':design['clusters'],'strata':design['strata'],**rubin(qs,us)}
        finite=rubin(qs,us,design['df']);result.update({'finite_df':finite['df'],'finite_lower':finite['lower'],'finite_upper':finite['upper']})
        rows.append(result)
        print('FITTED',co,label,'n',len(selected),'B',result['n_B'],flush=True)

r=pd.DataFrame(rows);r.to_csv(OUT/'cohort_sensitivity_estimates.csv',index=False,encoding='utf-8-sig')
pd.DataFrame(counts).to_csv(OUT/'sample_counts_and_timing.csv',index=False,encoding='utf-8-sig')
pd.DataFrame(scaling).to_csv(OUT/'domain_scaling_diagnostics.csv',index=False,encoding='utf-8-sig')
arch=pd.read_csv(ARCHIVE/'aggregate/core_refreshed_cohort.csv')
arch=arch.loc[arch.model.eq('categorical')&arch.effect.str.contains('[T.B]',regex=False)]
for co in ['CHARLS2015','ELSA_W8W9']:
    new=r.loc[r.cohort.eq(co)&r.analysis.eq('benchmark_original')].iloc[0]
    old=arch.loc[arch.cohort.eq(co)].iloc[0]
    assert abs(new.estimate-old.estimate)<1e-8 and abs(new.se-old.se)<1e-8,(co,'benchmark mismatch')
pooled=[]
for label in ['ELSA_interval_le6','ELSA_interval_le12','ELSA_interval_adjusted']:
    e=r.loc[r.analysis.eq(label)].iloc[0]
    others=arch.loc[arch.cohort.ne('ELSA_W8W9')]
    q=list(others.estimate)+[e.estimate];v=list(others.se**2)+[e.se**2]
    pooled.append({'analysis':label,'note':'only ELSA refitted; HRS and CHARLS original archived estimates retained',**meta(q,v)})
pd.DataFrame(pooled).to_csv(OUT/'ELSA_timing_three_cohort_pooling.csv',index=False,encoding='utf-8-sig')
config['benchmark_reproduction']='PASS: CHARLS and ELSA estimates and SEs agree within 1e-8'
config['uniform_three_cohort_pooling']='NOT PERFORMED: uniform HRS estimate unavailable'
config['ELSA_subpopulation_variance']='All 1377 household clusters in nine strata retained; zero score contributions for records outside fitted subpopulation.'
config['source_hashes']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [OLD/'corrected_modeling_combined_private.csv',OLD/'corrected_imputations_private.csv.gz']}
(OUT/'analysis_specification.json').write_text(json.dumps(config,indent=2),encoding='utf8')
print('COMPLETE; benchmarks passed; aggregate-only outputs',flush=True)
