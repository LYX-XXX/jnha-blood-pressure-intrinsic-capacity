from pathlib import Path
exec(Path('work/jnha_revision_1113/sensitivity_corrected.py').read_text(encoding='utf-8').split('results=[];detail=[]')[0])
forms={'categorical':MODELS['categorical'],'linear':f'I((sbp_mean-140)/10) * I((dbp_mean-80)/10) + {COV}','PP':f'I(pulse_pressure/10) + {COV}','MAP':f'I(mean_arterial_pressure/10) + {COV}'}
rows=[]
for co,base in b.groupby('cohort',sort=False):
 base=base.reset_index(drop=True);ci=im[im.cohort.eq(co)];kind='hc1' if co=='CHARLS2015' else ('reverse_codes' if co=='ELSA_W8W9' else 'archived')
 for model,formula in forms.items():
  values={}
  for mi in range(1,21):
   d=base.drop(columns=TARGETS).merge(ci.loc[ci.imputation.eq(mi),['participant_id']+TARGETS],on='participant_id',validate='one_to_one');d['age_per10']=(d.age_years-65)/10
   xd=dmatrix(formula,d,return_type='dataframe',eval_env=-1);x=xd.to_numpy();w=d.survey_weight.to_numpy();w=w/w.mean();y=d.ic_total_z.to_numpy();bread=np.linalg.pinv(x.T@(w[:,None]*x),rcond=1e-12);beta=bread@(x.T@(w*y));score=x*(w*(y-x@beta))[:,None]
   for j,name in enumerate(xd.columns):
    if not ('bp_group' in name or name.startswith('I(')):continue
    u,_=cluster_variance(score@bread[:,j],d,kind,x.shape[1]);values.setdefault(name,([],[]));values[name][0].append(beta[j]);values[name][1].append(u)
  for name,(q,u) in values.items():rows.append({'cohort':co,'model':model,'effect':name,**rubin(q,u)})
 print('REFRESH',co,flush=True)
r=pd.DataFrame(rows);r.to_csv(W/'core_refreshed_cohort.csv',index=False,encoding='utf-8-sig');met=[]
for (model,effect),g in r.groupby(['model','effect']):met.append({'model':model,'effect':effect,**meta(g.estimate,g.se**2)})
pd.DataFrame(met).to_csv(W/'core_refreshed_meta.csv',index=False,encoding='utf-8-sig')
# Complete-case domain models use the same observed sample and covariates.
domains=['ic_cognition_z','ic_psychology_z','ic_locomotion_z','ic_sensory_z','ic_vitality_z'];dr=[]
for co,base in b.groupby('cohort',sort=False):
 kind='hc1' if co=='CHARLS2015' else ('reverse_codes' if co=='ELSA_W8W9' else 'archived')
 for outcome in domains:
  d=base.copy();d['age_per10']=(d.age_years-65)/10;d=d.dropna(subset=[outcome,'age_per10','female','education_level','partnered','current_smoker','current_alcohol','diabetes','heart_disease','stroke','chronic_lung_disease','cancer']).reset_index(drop=True)
  for model in ['categorical','linear']:
   xd=dmatrix(forms[model],d,return_type='dataframe',eval_env=-1);x=xd.to_numpy();w=d.survey_weight.to_numpy();w=w/w.mean();y=d[outcome].to_numpy();bread=np.linalg.pinv(x.T@(w[:,None]*x),rcond=1e-12);beta=bread@(x.T@(w*y));score=x*(w*(y-x@beta))[:,None]
   name=next(c for c in xd.columns if ('[T.B]' in c if model=='categorical' else ':' in c));j=list(xd.columns).index(name);u,_=cluster_variance(score@bread[:,j],d,kind,x.shape[1]);se=np.sqrt(u);z=beta[j]/se
   dr.append({'cohort':co,'outcome':outcome,'effect':'B_vs_A' if model=='categorical' else 'SBP_x_DBP','n':len(d),'estimate':beta[j],'se':se,'lower':beta[j]-1.959964*se,'upper':beta[j]+1.959964*se,'p':2*stats.norm.sf(abs(z))})
dc=pd.DataFrame(dr);dc.to_csv(W/'domain_refreshed_cohort.csv',index=False,encoding='utf-8-sig');dm=[]
for (outcome,effect),g in dc.groupby(['outcome','effect']):
 m=meta(g.estimate,g.se**2);m['fixed_p']=2*stats.norm.sf(abs(m['fixed']/np.sqrt(1/sum(1/g.se**2))));m['random_p']=2*stats.t.sf(abs(m['random']/m['random_se']),2);dm.append({'outcome':outcome,'effect':effect,**m})
dm=pd.DataFrame(dm)
for typ in ['fixed','random']:
 a=dm[typ+'_p'].to_numpy();order=np.argsort(a);adj=np.minimum.accumulate((a[order]*len(a)/np.arange(1,len(a)+1))[::-1])[::-1];out=np.empty(len(a));out[order]=np.minimum(1,adj);dm[typ+'_BH_p']=out
dm.to_csv(W/'domain_refreshed_meta.csv',index=False,encoding='utf-8-sig');print('CORE AND DOMAIN REFRESH COMPLETE',flush=True)
