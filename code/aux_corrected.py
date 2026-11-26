from pathlib import Path
exec(Path('work/jnha_revision_1113/sensitivity_corrected.py').read_text(encoding='utf-8').split('results=[];detail=[]')[0])
rows=[]
dom=['ic_cognition_z','ic_psychology_z','ic_locomotion_z','ic_sensory_z','ic_vitality_z']
for co,base in b.groupby('cohort',sort=False):
 base=base.reset_index(drop=True);complete=base[dom].notna().all(axis=1)
 for label,cols in [('five_domains',dom),('excluding_vitality',dom[:-1])]:
  raw=base[cols].mean(axis=1).where(complete);base[label]=(raw-raw.mean())/raw.std(ddof=0)
 ci=im[im.cohort.eq(co)];kind='hc1' if co=='CHARLS2015' else ('reverse_codes' if co=='ELSA_W8W9' else 'archived')
 for outcome in ['five_domains','excluding_vitality','low_IC']:
  qs=[];us=[]
  for mi in range(1,21):
   d=base.drop(columns=TARGETS).merge(ci.loc[ci.imputation.eq(mi),['participant_id']+TARGETS],on='participant_id',validate='one_to_one');d['age_per10']=(d.age_years-65)/10
   if outcome!='low_IC':d=d.dropna(subset=[outcome]).reset_index(drop=True)
   xd=dmatrix(MODELS['categorical'],d,return_type='dataframe',eval_env=-1);x=xd.to_numpy();w=d.survey_weight.to_numpy();w=w/w.mean();j=next(j for j,k in enumerate(xd.columns) if '[T.B]' in k)
   if outcome=='low_IC':
    y=(d.ic_total_z.to_numpy()<=-1).astype(float);beta=np.zeros(x.shape[1]);beta[0]=np.log(np.average(y,weights=w))
    for it in range(100):
     mu=np.exp(np.clip(x@beta,-30,30));bread=np.linalg.pinv(x.T@((w*mu)[:,None]*x),rcond=1e-12);delta=bread@(x.T@(w*(y-mu)));beta+=delta
     if np.max(np.abs(delta))<1e-10:break
    mu=np.exp(x@beta);bread=np.linalg.pinv(x.T@((w*mu)[:,None]*x),rcond=1e-12);score=x*(w*(y-mu))[:,None]
   else:
    y=d[outcome].to_numpy();bread=np.linalg.pinv(x.T@(w[:,None]*x),rcond=1e-12);beta=bread@(x.T@(w*y));score=x*(w*(y-x@beta))[:,None]
   u,_=cluster_variance(score@bread[:,j],d,kind,x.shape[1]);qs.append(beta[j]);us.append(u)
  rows.append({'cohort':co,'outcome':outcome,'n':len(d),**rubin(qs,us)})
r=pd.DataFrame(rows);r.to_csv(W/'aux_refreshed_cohort.csv',index=False,encoding='utf-8-sig');met=[]
for outcome,g in r.groupby('outcome'):met.append({'outcome':outcome,**meta(g.estimate,g.se**2)})
m=pd.DataFrame(met);m.to_csv(W/'aux_refreshed_meta.csv',index=False,encoding='utf-8-sig');v=m.set_index('outcome');print('ATTENUATION',100*(1-abs(v.loc['excluding_vitality','fixed']/v.loc['five_domains','fixed'])),flush=True)
