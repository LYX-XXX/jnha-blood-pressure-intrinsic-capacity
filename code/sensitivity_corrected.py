from pathlib import Path
import sys,os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
sys.path.insert(0,str(Path('work/analysis_runtime').resolve()))
sys.path.append(r'C:\Users\SAMSUNG\Documents\Codex\2026-09-29\c-users-samsung-desktop-ai-hrs\work\step17_runtime')
import numpy as np,pandas as pd,scipy,patsy,pyreadstat
from scipy import stats,optimize
from patsy import dmatrix,build_design_matrices
import json,zipfile,hashlib,itertools
B=Path(r'C:\Users\SAMSUNG\Desktop\AI\数据库论文\数据');W=Path('work/jnha_revision_1113/corrected_analysis');W.mkdir(exist_ok=True)
BASE=Path('work/jnha_revision_1113/corrected_modeling_combined_private.csv');IMP=Path('work/jnha_revision_1113/corrected_imputations_private.csv.gz')
b=pd.read_csv(BASE,dtype={'participant_id':'string','survey_psu':'string'});im=pd.read_csv(IMP,dtype={'participant_id':'string'})
def key(x):return x.astype('string').str.replace(r'\.0$','',regex=True).str.strip()
b.participant_id=key(b.participant_id);im.participant_id=key(im.participant_id)
assert len(b)==4650 and not b.duplicated(['cohort','participant_id']).any();assert not im.duplicated(['cohort','participant_id','imputation']).any()
# Join actual community IDs; no inferred strata.
z=zipfile.ZipFile(B/'2015 CHARLS Wave 3/2015 CHARLS Wave 3/Sample_Infor.zip');temp=W/'charls_sample_metadata.dta';temp.write_bytes(z.read('Sample_Infor.dta'))
cm,cmmeta=pyreadstat.read_dta(str(temp),usecols=['ID','communityID'],apply_value_formats=False);cm['participant_id']=key(cm.ID);cm['community_id']=key(cm.communityID)
assert not cm.participant_id.duplicated().any()
char=b.cohort.eq('CHARLS2015');joined=b.loc[char,['participant_id']].merge(cm[['participant_id','community_id']],on='participant_id',how='left',validate='one_to_one');assert joined.community_id.notna().all()
b.loc[char,'community_id']=joined.community_id.to_numpy()
# Record immutable input hashes and software before fitting.
config={'python':sys.version,'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__,'patsy':patsy.__version__,'pyreadstat':pyreadstat.__version__,'base_sha256':hashlib.sha256(BASE.read_bytes()).hexdigest(),'imputation_sha256':hashlib.sha256(IMP.read_bytes()).hexdigest(),'new_analyses':'exploratory; conditional associational BP-distribution contrasts'}
(W/'configuration.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
COV='age_per10 + female + C(education_level, Treatment(reference=1)) + partnered + current_smoker + current_alcohol + diabetes + heart_disease + stroke + chronic_lung_disease + cancer'
SB="cr(sbp_mean, knots=(120,140,160), lower_bound=75, upper_bound=220, constraints='center')";DB="cr(dbp_mean, knots=(60,75,90), lower_bound=35, upper_bound=130, constraints='center')";INT='I(((sbp_mean-140)/10)*((dbp_mean-80)/10))'
MODELS={'categorical':"C(bp_group, Treatment(reference='A')) + "+COV,'original_spline':SB+' + '+DB+' + '+INT+' + '+COV,'additive_no_interaction':SB+' + '+DB+' + '+COV,'tensor_spline':SB+' * '+DB+' + '+COV}
TARGETS=[x for x in im if x not in ['cohort','imputation','participant_id']]

# Scalar influence variance with explicit singleton policies.
def cluster_variance(v,d,kind,p):
 n=len(d)
 if kind=='hc1':return float(v@v*n/(n-p)),{'clusters':n,'strata':0,'singletons':0,'df':n-p}
 if kind=='hc3':raise ValueError('HC3 must use leverage-adjusted influence')
 if kind=='community':cl=d.community_id.astype(str);st=np.zeros(n,dtype=int)
 elif kind=='region':cl=d.survey_psu.astype(str);st=np.zeros(n,dtype=int)
 elif kind=='reverse_codes':cl=d.survey_stratum.astype(str);st=d.survey_psu.astype(str)
 elif kind=='unstratified':cl=d.survey_stratum.astype(str)+'|'+d.survey_psu.astype(str);st=np.zeros(n,dtype=int)
 else:cl=d.survey_psu.astype(str);st=d.survey_stratum.astype(str)
 f=pd.DataFrame({'v':v,'s':np.asarray(st),'c':np.asarray(cl)}).groupby(['s','c'],sort=False,observed=True).v.sum().reset_index()
 groups=[g.v.to_numpy() for _,g in f.groupby('s',sort=False,observed=True)];non=[];sing=[]
 for u in groups:
  if len(u)>1:non.append(float(len(u)/(len(u)-1)*np.sum((u-u.mean())**2)))
  else:sing.append(float(u[0]))
 count=len(sing);grand=float(f.v.mean())
 if kind=='singleton_average' and not non:return np.nan,{'clusters':len(f),'strata':len(groups),'singletons':count,'df':len(f)-len(groups),'status':'not_estimable_no_multipsu_strata'}
 total=sum(non)
 for u in sing:
  if kind=='singleton_center':total+=(u-grand)**2
  elif kind=='singleton_average':total+=float(np.mean(non))
  else:total+=u*u
 if kind in ['community','region']:total*= (n-1)/(n-p) # cluster finite-sample correction, J/(J-1) already included
 return total,{'clusters':len(f),'strata':len(groups),'singletons':count,'df':len(f)-len(groups)}

def rubin(q,u,dfcomplete=np.inf):
 q=np.array(q);u=np.array(u);m=len(q);est=q.mean();within=u.mean();between=q.var(ddof=1);tot=within+(1+1/m)*between
 se=np.sqrt(max(0,tot));r=(1+1/m)*between/within if within>0 else np.inf
 olddf=(m-1)*(1+1/r)**2 if r>0 and np.isfinite(r) else np.inf
 # Archived scalar Rubin inference; optionally Barnard–Rubin small complete-data df.
 if np.isfinite(dfcomplete) and dfcomplete>0:
  lam=((1+1/m)*between)/tot if tot>0 else 0
  obsdf=((dfcomplete+1)/(dfcomplete+3))*dfcomplete*(1-lam)
  df=1/(1/olddf+1/obsdf) if obsdf>0 else np.nan
 else:df=olddf
 crit=stats.t.ppf(.975,df) if np.isfinite(df) else stats.norm.ppf(.975)
 return {'estimate':est,'se':se,'lower':est-crit*se,'upper':est+crit*se,'within':within,'between':between,'df':df,'p':float(2*(stats.t.sf(abs(est/se),df) if np.isfinite(df) else stats.norm.sf(abs(est/se)))) if se>0 else np.nan}

def meta(y,v):
 y=np.array(y);v=np.array(v);k=len(y);w=1/v;fe=np.sum(w*y)/sum(w);fes=np.sqrt(1/sum(w));Q=np.sum(w*(y-fe)**2)
 def obj(t):
  w=1/(v+t);mu=sum(w*y)/sum(w);return .5*(np.log(v+t).sum()+np.log(sum(w))+sum(w*(y-mu)**2))
 opt=optimize.minimize_scalar(obj,bounds=(0,max(1,float(np.var(y))*10)),method='bounded',options={'xatol':1e-13});tau=0 if obj(0)<=opt.fun else float(opt.x)
 wr=1/(v+tau);re=sum(wr*y)/sum(wr);q=sum(wr*(y-re)**2)/(k-1);res=np.sqrt(max(1,q)/sum(wr));c=stats.t.ppf(.975,k-1)
 return {'k':k,'fixed':fe,'fixed_lower':fe-1.959964*fes,'fixed_upper':fe+1.959964*fes,'I2':max(0,(Q-(k-1))/Q)*100 if Q>0 else 0,'tau2':tau,'random':re,'random_lower':re-c*res,'random_upper':re+c*res,'random_se':res,'random_normal_lower':re-1.959964/np.sqrt(sum(wr)),'random_normal_upper':re+1.959964/np.sqrt(sum(wr)),'prediction_lower':re-stats.t.ppf(.975,max(1,k-2))*np.sqrt(tau+res**2),'prediction_upper':re+stats.t.ppf(.975,max(1,k-2))*np.sqrt(tau+res**2)}

results=[];detail=[];design=[];contrasts=[];paired=[];fits=[];diagnostics=[]
for co,base in b.groupby('cohort',sort=False):
 base=base.reset_index(drop=True);ci=im.loc[im.cohort.eq(co)];samples={};influences={}
 sbq=tuple(float(x) for x in np.quantile(base.sbp_mean,[.1,.5,.9]));dbq=tuple(float(x) for x in np.quantile(base.dbp_mean,[.1,.5,.9]));forms=dict(MODELS)
 forms['quantile_knots']=f"cr(sbp_mean, knots={sbq}, lower_bound=75, upper_bound=220, constraints='center') + cr(dbp_mean, knots={dbq}, lower_bound=35, upper_bound=130, constraints='center') + {INT} + {COV}"
 for model,formula in forms.items():
  qs=[];us=[];ucs=[];points=[];pointus=[];qdiff=[];udiff=[];kindvalues={}
  for mi in range(1,21):
   d=base.drop(columns=TARGETS).merge(ci.loc[ci.imputation.eq(mi),['participant_id']+TARGETS],on='participant_id',validate='one_to_one');d['age_per10']=(d.age_years-65)/10
   xd=dmatrix(formula,d,return_type='dataframe',eval_env=-1);x=xd.to_numpy();y=d.ic_total_z.to_numpy();w=d.survey_weight.to_numpy();w=w/w.mean();n,p=x.shape
   assert n==len(base) and np.linalg.matrix_rank(x)==p
   bread=np.linalg.pinv(x.T@(w[:,None]*x),rcond=1e-12);beta=bread@(x.T@(w*y));res=y-x@beta;score=x*(w*res)[:,None]
   # Common-covariate standardization: same nuisance distribution, observed group-specific BP distributions.
   names=list(xd.columns);bpcols=[i for i,k in enumerate(names) if 'sbp_mean' in k or 'dbp_mean' in k or 'bp_group' in k]
   A=d.bp_group.eq('A').to_numpy();G=d.bp_group.eq('B').to_numpy();C=np.zeros(p);C[bpcols]=np.average(x[G][:,bpcols],weights=w[G],axis=0)-np.average(x[A][:,bpcols],weights=w[A],axis=0)
   q=float(C@beta);vreg=score@(bread@C);f=x[:,bpcols]@beta[bpcols];muB=np.average(f[G],weights=w[G]);muA=np.average(f[A],weights=w[A]);vdirect=G*w/w[G].sum()*(f-muB)-A*w/w[A].sum()*(f-muA);vinf=vreg+vdirect
   basekind='hc1' if co=='CHARLS2015' else ('reverse_codes' if co=='ELSA_W8W9' else 'archived');u,di=cluster_variance(vinf,d,basekind,p);uc,_=cluster_variance(vreg,d,basekind,p)
   qs.append(q);us.append(u);ucs.append(uc)
   if mi==1:
    design.append({'cohort':co,'model':model,'n':n,'p':p,**di,'n_A':int(A.sum()),'n_B':int(G.sum()),'condition_number':float(np.linalg.cond(x.T@(w[:,None]*x)))})
   if model=='categorical':influences[mi]=(q,vinf)
   else:
    refq,refv=influences[mi];dv=vinf-refv;du,_=cluster_variance(dv,d,basekind,p);qdiff.append(q-refq);udiff.append(du)
   if model!='categorical':
    pred=d.iloc[[0,0]].copy();pred['sbp_mean']=[120,140];pred['dbp_mean']=[55,80]
    xp=np.asarray(build_design_matrices([xd.design_info],pred)[0]);cp=xp[0]-xp[1];pq=float(cp@beta);pu,_=cluster_variance(score@(bread@cp),d,basekind,p);points.append(pq);pointus.append(pu)
   if model=='categorical':
    h=np.sum((x@bread)*x,axis=1)*w
    variants=['hc1','hc3','community'] if co=='CHARLS2015' else ['archived','singleton_center','singleton_average','unstratified']+(['region','reverse_codes'] if co=='ELSA_W8W9' else [])
    for kind in variants:
     if kind=='hc3':vu=float(np.sum((vinf/(1-h))**2));di2={'df':n-p,'clusters':n,'strata':0,'singletons':0}
     else:vu,di2=cluster_variance(vinf,d,kind,p)
     kindvalues.setdefault(kind,{'q':[],'u':[],'di':di2});kindvalues[kind]['q'].append(q);kindvalues[kind]['u'].append(vu)
   if mi==1:fits.append({'cohort':co,'model':model,'n':n,'p':p,'weighted_R2':1-sum(w*res**2)/sum(w*(y-np.average(y,weights=w))**2),'SBP_knots':str(sbq) if model=='quantile_knots' else '(120,140,160)','DBP_knots':str(dbq) if model=='quantile_knots' else '(60,75,90)'})
  results.append({'cohort':co,'model':model,'contrast':'standardized_B_minus_A',**rubin(qs,us)})
  results.append({'cohort':co,'model':model,'contrast':'fixed_BP_distribution_B_minus_A',**rubin(qs,ucs)})
  if points:contrasts.append({'cohort':co,'model':model,'contrast':'120_55_vs_140_80',**rubin(points,pointus)})
  if qdiff:paired.append({'cohort':co,'model':model,'contrast':'standardized_minus_categorical',**rubin(qdiff,udiff)})
  if model=='categorical':
   for kind,vals in kindvalues.items():
    di=vals['di']
    if not np.isfinite(vals['u']).all():diagnostics.append({'cohort':co,'kind':kind,'status':'not_estimable_no_multipsu_strata',**di});continue
    for rule in ['archived_Rubin','finite_cluster_df']:
     if rule=='finite_cluster_df' and di['df']<=0:
      diagnostics.append({'cohort':co,'kind':kind,'status':'zero_complete_data_design_df',**di});continue
     detail.append({'cohort':co,'kind':kind,'interval_rule':rule,**di,**rubin(vals['q'],vals['u'],di['df'] if rule=='finite_cluster_df' else np.inf)})
  print('FITTED',co,model,flush=True)
# HRS scores deliberately corrected. Old-estimate reproduction is inapplicable.
r=pd.DataFrame(results);v=pd.DataFrame(detail);pt=pd.DataFrame(contrasts)
expected={'HRS2016':0,'CHARLS2015':0,'ELSA_W8W9':0}
metas=[]
for (model,con),g in r.groupby(['model','contrast']):metas.append({'model':model,'contrast':con,**meta(g.estimate,g.se**2)})
# Full-cohort variance scenarios: only the stated cohort's covariance changes, point model fixed.
scenarios={'corrected_ELSA':{},'archived':{'ELSA_W8W9':'archived'},'HRS_center':{'HRS2016':'singleton_center'},'HRS_average':{'HRS2016':'singleton_average'},'unstratified_codes':{'HRS2016':'unstratified','ELSA_W8W9':'unstratified'},'CHARLS_HC3':{'CHARLS2015':'hc3'},'CHARLS_community':{'CHARLS2015':'community'},'ELSA_region':{'ELSA_W8W9':'region'},'ELSA_geographic_strata':{'ELSA_W8W9':'reverse_codes'}}
vm=[]
for scenario,changes in scenarios.items():
 selected=[]
 for co in expected:
  kind=changes.get(co,'hc1' if co=='CHARLS2015' else ('reverse_codes' if co=='ELSA_W8W9' else 'archived'));selected.append(v[v.cohort.eq(co)&v.kind.eq(kind)&v.interval_rule.eq('archived_Rubin')].iloc[0])
 g=pd.DataFrame(selected);vm.append({'scenario':scenario,**meta(g.estimate,g.se**2)})
basev=v[v.interval_rule.eq('archived_Rubin')&((v.cohort.eq('CHARLS2015')&v.kind.eq('hc1'))|(v.cohort.eq('HRS2016')&v.kind.eq('archived'))|(v.cohort.eq('ELSA_W8W9')&v.kind.eq('reverse_codes')))]
loo=[]
for omit in expected:
 g=basev[~basev.cohort.eq(omit)];loo.append({'omitted':omit,**meta(g.estimate,g.se**2)})
for name,df in [('model_contrasts',r),('model_point_contrasts',pt),('model_paired_differences',pd.DataFrame(paired)),('model_meta',pd.DataFrame(metas)),('variance_cohort',v),('variance_meta',pd.DataFrame(vm)),('leave_one_cohort_out',pd.DataFrame(loo)),('design_audit',pd.DataFrame(design)),('fit_audit',pd.DataFrame(fits)),('non_estimable_designs',pd.DataFrame(diagnostics))]:df.to_csv(W/(name+'.csv'),index=False,encoding='utf-8-sig')
print('ALL CHECKS PASSED. Aggregate outputs written.',flush=True)
