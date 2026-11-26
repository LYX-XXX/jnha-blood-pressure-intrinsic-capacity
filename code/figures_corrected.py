from pathlib import Path
exec(Path('work/jnha_revision_1113/sensitivity_corrected.py').read_text(encoding='utf-8').split('results=[];detail=[]')[0])
import importlib.util,shutil
O=Path('work/jnha_revision_1113/corrected_figures');O.mkdir(exist_ok=True)
src=Path('work')
def load(name,file):
 spec=importlib.util.spec_from_file_location(name,file);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
forest=load('forest',src/'jnha_plot_forest.py');surface=load('surface',src/'jnha_plot_surface.py')
r=pd.read_csv(W/'core_refreshed_cohort.csv');m=pd.read_csv(W/'core_refreshed_meta.csv');keep=(r.effect.str.contains(r'\[T.B\]',regex=True))|((r.model=='linear')&r.effect.str.contains(':'));r=r[keep].copy();r['term']=np.where(r.model.eq('categorical'),'bp_B','sbp_dbp_interaction');r=r.rename(columns={'estimate':'beta_pooled','se':'standard_error_pooled','lower':'ci95_lower','upper':'ci95_upper'});r['n_min']=r.cohort.map(b.groupby('cohort').size())
keep=m.effect.str.contains(r'\[T.B\]',regex=True)|((m.model=='linear')&m.effect.str.contains(':'));m=m[keep].copy();m['term']=np.where(m.model.eq('categorical'),'bp_B','sbp_dbp_interaction')
def actual_Q(term):
 g=r[r.term.eq(term)];w=1/g.standard_error_pooled**2;mu=np.sum(w*g.beta_pooled)/sum(w);return float(np.sum(w*(g.beta_pooled-mu)**2))
m['Q']=m.term.map(actual_Q);m['Q_p_value']=stats.chi2.sf(m.Q,2)
m=m.rename(columns={'fixed':'fixed_estimate','fixed_lower':'fixed_ci95_lower','fixed_upper':'fixed_ci95_upper','random':'random_estimate','random_lower':'random_ci95_lower_modified_KH','random_upper':'random_ci95_upper_modified_KH','I2':'I2_percent','tau2':'tau2_REML'})
r.to_csv(W/'forest_inputs.csv',index=False);m.to_csv(W/'forest_meta.csv',index=False)
forest.plot_figure(r,m,O)
# Keep the original grid and display support; refresh ELSA pointwise variances.
paths=json.loads(Path('work/evidence_paths.json').read_text(encoding='utf-8'));grid=pd.read_csv(paths['spline']);
for co in ['HRS2016','ELSA_W8W9']:
 base=b[b.cohort.eq(co)].reset_index(drop=True);ci=im[im.cohort.eq(co)];g=grid[grid.cohort.eq(co)].copy();qs=[];us=[]
 for mi in range(1,21):
  d=base.drop(columns=TARGETS).merge(ci.loc[ci.imputation.eq(mi),['participant_id']+TARGETS],on='participant_id',validate='one_to_one');d['age_per10']=(d.age_years-65)/10
  xd=dmatrix(MODELS['original_spline'],d,return_type='dataframe',eval_env=-1);x=xd.to_numpy();w=d.survey_weight.to_numpy();w=w/w.mean();y=d.ic_total_z.to_numpy();bread=np.linalg.pinv(x.T@(w[:,None]*x),rcond=1e-12);beta=bread@(x.T@(w*y));score=x*(w*(y-x@beta))[:,None]
  group=pd.DataFrame(score).assign(st=d.survey_psu.astype(str) if co=='ELSA_W8W9' else d.survey_stratum.astype(str),cl=d.survey_stratum.astype(str) if co=='ELSA_W8W9' else d.survey_psu.astype(str)).groupby(['st','cl']).sum();cov=np.zeros((x.shape[1],x.shape[1]))
  for _,gg in group.groupby(level=0):
   u=gg.to_numpy()
   if len(u)>1:
    u=u-u.mean(axis=0);cov+=len(u)/(len(u)-1)*(u.T@u)
   else:cov+=u.T@u
  cov=bread@cov@bread
  pred=pd.concat([d.iloc[[0]]]*len(g),ignore_index=True);pred['sbp_mean']=g.sbp_mean.to_numpy();pred['dbp_mean']=g.dbp_mean.to_numpy();xp=np.asarray(build_design_matrices([xd.design_info],pred)[0]);ref=d.iloc[[0]].copy();ref['sbp_mean']=140;ref['dbp_mean']=80;xp-=np.asarray(build_design_matrices([xd.design_info],ref)[0])[0]
  qs.append(xp@beta);us.append(np.einsum('ij,jk,ik->i',xp,cov,xp))
 q=np.array(qs);u=np.array(us);est=q.mean(axis=0);bt=q.var(axis=0,ddof=1);wi=u.mean(axis=0);var=wi+1.05*bt;rr=np.divide(1.05*bt,wi,out=np.zeros_like(bt),where=wi>0);df=np.divide(19*(1+rr)**2,rr**2,out=np.full_like(rr,np.inf),where=rr>0);crit=stats.t.ppf(.975,df)
 # HRS point estimates change after correcting scores.
 grid.loc[g.index,'estimate']=est;grid.loc[g.index,'ci95_lower']=est-crit*np.sqrt(var);grid.loc[g.index,'ci95_upper']=est+crit*np.sqrt(var);grid.to_csv(W/'spline_grid_refreshed.csv.gz',index=False)
surface.plot_figure(grid,O)
shutil.copyfile(O/'figure3_meta_forest.tiff','outputs/JNHA_Corrected_Figure_2.tiff');shutil.copyfile(O/'figure2_joint_surface_and_slices.tiff','outputs/JNHA_Corrected_Figure_3.tiff');shutil.copyfile('work/latest_image1.tiff','outputs/JNHA_Corrected_Figure_1.tiff')
print('UPDATED FIGURES WRITTEN',flush=True)
