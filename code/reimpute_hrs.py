from pathlib import Path
import os,sys,json
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
sys.path.insert(0,str(Path('work/analysis_runtime').resolve()))
import pandas as pd,numpy as np,statsmodels
from statsmodels.imputation.mice import MICEData
B=Path(r'C:\Users\SAMSUNG\Desktop\AI\数据库论文\数据');W=Path('work/jnha_revision_1113')
src=B/'代码与results/03 数据处理、主要分析、敏感性分析、绘图制表代码/主要分析/run_step12_multiple_imputation.py';code=src.read_text(encoding='utf-8');params=code[code.index('M = 20'):code.index('def zscore')];defs=code[code.index('def zscore'):code.index('def fit_one')].replace('        rebuilt = prepare(rebuilt)','        # Auxiliary design columns are not needed for the covariate export.')
exec(params+defs)
base=pd.read_csv(W/'corrected_modeling_combined_private.csv',dtype={'participant_id':'string','survey_psu':'string'});h=base[base.cohort.eq('HRS2016')].reset_index(drop=True)
_,diag,long=run_imputation('HRS2016',h)
old=pd.read_csv(B/'第12步_多重插补与Rubin合并_2026-10-02/02_imputed_covariates_long.csv.gz',dtype={'participant_id':'string'});other=old[~old.cohort.eq('HRS2016')]
all_imputed=pd.concat([long,other],ignore_index=True);assert len(all_imputed)==4650*20
all_imputed.to_csv(W/'corrected_imputations_private.csv.gz',index=False);diag.to_csv(W/'corrected_HRS_imputation_diagnostics.csv',index=False,encoding='utf-8-sig');(W/'reimputation_audit.json').write_text(json.dumps({'m':M,'burn_in':BURN_IN,'between_samples':BETWEEN_SAMPLES,'seed':SEEDS['HRS2016'],'method':'MICE Gaussian perturbation, predictive mean matching, 20 nearest donors','statsmodels':statsmodels.__version__,'only_HRS_reimputed':True,'CHARLS_ELSA_imputations_retained':True},indent=2),encoding='utf-8');print('HRS REIMPUTATION COMPLETE',len(long),flush=True)
