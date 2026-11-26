from pathlib import Path
exec(Path('work/jnha_revision_1113/sensitivity_corrected.py').read_text(encoding='utf-8').split('results=[];detail=[]')[0])
# Refit retained score/cutoff sensitivities using the corrected scores and final design roles.
cwalk,_=pyreadstat.read_dta(str(B/'【数据】Harmonized CHARLS/【数据】Harmonized CHARLS/Harmonized_CHARLS_D_Data.dta'),usecols=['ID','r3wspeed'],apply_value_formats=False);cwalk['participant_id']=key(cwalk.ID)
ewalk,_=pyreadstat.read_sav(str(B/'【数据】Harmonized ELSA/【数据】Harmonized ELSA/h_elsa_g3.sav'),usecols=['idauniq','r8wspeed'],apply_value_formats=False);ewalk['participant_id']=key(ewalk.idauniq)
hwalk=pd.read_csv('work/jnha_revision_1113/hrs_corrected/HRS2016_analysis_ready.csv',dtype={'hhidpn':'string'});hwalk['participant_id']=key(hwalk.hhidpn)
walk={'HRS2016':hwalk.set_index('participant_id').r13wspeed,'CHARLS2015':cwalk.set_index('participant_id').r3wspeed,'ELSA_W8W9':ewalk.set_index('participant_id').r8wspeed}
dom=['ic_cognition_z','ic_psychology_z','ic_locomotion_z','ic_sensory_z','ic_vitality_z'];rows=[];loads=[]
def std(s,mask):
 r=s[mask&s.notna()];return (s-r.mean())/r.std(ddof=0)
for co,base in b.groupby('cohort',sort=False):
 base=base.reset_index(drop=True);complete=base[dom].notna().all(axis=1);ranked=base.loc[complete,dom].rank(method='average');ranked=stats.norm.ppf((ranked-.5)/len(ranked));z=pd.Series(np.nan,index=base.index);z.loc[complete]=ranked.mean(axis=1);base['rank_normal']=std(z,complete)
 x=base.loc[complete,dom].to_numpy();x=(x-x.mean(axis=0))/x.std(axis=0,ddof=0);vals,vec=np.linalg.eigh(np.corrcoef(x,rowvar=False));v=vec[:,np.argmax(vals)];v=v if v.sum()>0 else -v;s=pd.Series(np.nan,index=base.index);s.loc[complete]=x@v;base['PCA']=std(s,complete)
 for name,value in zip(dom,v):loads.append({'cohort':co,'domain':name,'loading':value,'eigenvalue':vals.max()})
 time=base.participant_id.map(walk[co]);time=time.where(time.between(1,60));zt=std(-time,time.notna());xx=base[[v for v in dom if v!='ic_locomotion_z']].copy();xx['objective']=zt;ok=xx.notna().all(axis=1);base['objective_locomotion']=std(xx.mean(axis=1).where(ok),ok);base['percentile']=base.ic_total_z.rank(method='average',pct=True)*100
 ci=im[im.cohort.eq(co)];kind='hc1' if co=='CHARLS2015' else ('reverse_codes' if co=='ELSA_W8W9' else 'archived')
 specs=[('percentile','percentile',140,60,''),('rank_normal','rank_normal',140,60,''),('PCA','PCA',140,60,''),('objective_locomotion','objective_locomotion',140,60,''),('SBP130','ic_total_z',130,60,''),('SBP150','ic_total_z',150,60,''),('DBP55','ic_total_z',140,55,''),('DBP65','ic_total_z',140,65,''),('BMI_adjusted','ic_total_z',140,60,' + C(bmi_cat, Treatment(reference=1))'),('activity_adjusted','ic_total_z',140,60,' + moderate_or_vigorous_weekly')]
 for label,outcome,sbp_cut,dbp_cut,extra in specs:
  qs=[];us=[]
  for mi in range(1,21):
   d=base.drop(columns=TARGETS).merge(ci.loc[ci.imputation.eq(mi),['participant_id']+TARGETS],on='participant_id',validate='one_to_one');d['age_per10']=(d.age_years-65)/10;d['bmi_cat']=pd.cut(d.bmi,[-np.inf,18.5,25,30,np.inf],labels=[0,1,2,3],right=False).astype(int)
   if sbp_cut!=140 or dbp_cut!=60:d['bp_group']=np.select([d.dbp_mean.ge(90),d.sbp_mean.lt(sbp_cut)&d.dbp_mean.lt(dbp_cut),d.sbp_mean.lt(sbp_cut)&d.dbp_mean.ge(dbp_cut),d.sbp_mean.ge(sbp_cut)&d.dbp_mean.lt(dbp_cut)],['E','B','A','D'],default='C')
   d=d.dropna(subset=[outcome]).reset_index(drop=True);xd=dmatrix(MODELS['categorical']+extra,d,return_type='dataframe',eval_env=-1);x=xd.to_numpy();w=d.survey_weight.to_numpy();w=w/w.mean();y=d[outcome].to_numpy();bread=np.linalg.pinv(x.T@(w[:,None]*x),rcond=1e-12);beta=bread@(x.T@(w*y));score=x*(w*(y-x@beta))[:,None];j=next(j for j,k in enumerate(xd.columns) if '[T.B]' in k);u,_=cluster_variance(score@bread[:,j],d,kind,x.shape[1]);qs.append(beta[j]);us.append(u)
  rows.append({'cohort':co,'analysis':label,'n':len(d),**rubin(qs,us)})
 print('EXTRA',co,flush=True)
r=pd.DataFrame(rows);r.to_csv(W/'extra_refreshed_cohort.csv',index=False,encoding='utf-8-sig');pd.DataFrame(loads).to_csv(W/'PCA_loadings.csv',index=False,encoding='utf-8-sig');met=[]
for label,g in r.groupby('analysis'):met.append({'analysis':label,**meta(g.estimate,g.se**2)})
pd.DataFrame(met).to_csv(W/'extra_refreshed_meta.csv',index=False,encoding='utf-8-sig');print('EXTRA ANALYSES COMPLETE',flush=True)
