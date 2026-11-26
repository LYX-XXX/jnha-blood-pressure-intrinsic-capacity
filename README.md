# Blood pressure phenotypes and intrinsic capacity in treated older adults

Supporting aggregate results and archived analysis scripts for:

Low diastolic blood pressure phenotype and lower intrinsic capacity among treated older adults: evidence from three national ageing studies.

## Status and provenance

These files were recovered from the corrected 13 November 2026 evidence archive, not the older uncompressed JNHA_Statistical_Evidence folder. The archive corrects HRS intrinsic-capacity scoring and repeats HRS covariate imputation. It retains the existing CHARLS and ELSA covariate imputations. See audit/hrs_score_changes.json and audit/reimputation_audit.json.

The HRS B-versus-A estimate is -0.1697272894. Fixed-effect pooling is -0.2056150764 (95% CI -0.3040531983 to -0.1071769545); REML pooling with modified Knapp-Hartung inference is -0.2079901072 (95% CI -0.4419301102 to 0.0259498958). These round to the main manuscript values. The fixed-effect result was independently recalculated from the archived cohort estimates and standard errors. This does not constitute a rerun from participant-level data.

## Contents

- aggregate/: cohort and pooled estimates, model contrasts, design diagnostics, and model prediction grids. Rows describe estimates, scenarios or prediction-grid coordinates, not participants.
- audit/: score correction, imputation, timing, missing-domain patterns and environment metadata.
- code/: selected original analysis and score-correction scripts, preserved as archived.
- verification.json: scope and numerical checks performed during recovery.
- SHA256SUMS.txt: checksums for this repository snapshot.

core_refreshed_* contain the categorical and linear primary models. domain_refreshed_* contain complete-case domain analyses. extra_refreshed_* and aux_refreshed_* contain scoring, cutoff and additional adjustment analyses. model_* contain spline specifications and point/group/paired contrasts. variance_* and leave_one_cohort_out contain design and pooling sensitivity analyses. forest_* are plotting inputs. The archived outputs include additional analyses not selected for the current manuscript, including a pooled D-group row; that row is not endorsed as a main-paper estimate because the manuscript deliberately does not pool sparse group D.

## Data access

No participant-level cohort data, participant identifiers, completed imputation microdata or cohort download archives are redistributed. Authorized users must obtain HRS, CHARLS and ELSA data directly from their custodians:

- HRS: https://hrsdata.isr.umich.edu/
- CHARLS: https://charls.charlsdata.com/
- ELSA: https://www.elsa-project.ac.uk/data-and-documentation

Cohort data retain their own access and use conditions. No license to redistribute third-party cohort data is granted here.

## Reanalysis limitations

The code is an archived analysis snapshot, not a portable one-command pipeline. It retains local source and work paths and expects authorized original or corrected modeling files and 20-imputation files that are not included. Some correction and plotting scripts depend on external local processing scripts or nature-figure helpers; those dependencies must be supplied and paths configured by an authorized analyst. Read the scripts before running them. No analysis scripts were executed during recovery.

For the corrected analysis stage, core_corrected.py, aux_corrected.py and extra_corrected.py read definitions from sensitivity_corrected.py using the original work/jnha_revision_1113 layout. sensitivity_corrected.py also runs exploratory model comparisons. Score correction and HRS reimputation precede this stage. Do not substitute the old uncorrected modeling file merely to satisfy a missing path.

Reported runtime versions are in audit/configuration.json; the HRS reimputation audit records statsmodels 0.15.0. The manuscript is a cross-sectional associational study; these results do not establish effects of changing antihypertensive treatment.

## Verification and supporting dependency snapshots

Run `python verify_aggregate.py` from any working directory. It uses only Python standard-library modules and included aggregate tables. Verification passed for nine fixed-effect estimates and confidence intervals, 20 BH-adjusted p values and 12 paired contrast point estimates. It does not validate random-effects fitting or rerun participant-level models.

`code/dependencies/` contains four historical processing and imputation modules referenced by the corrected scripts. Their presence makes the source chain inspectable; they retain original filesystem assumptions and must not be substituted for the corrected outputs. `figures_corrected.py` also expects historical plotting helpers and an evidence-path mapping not included here, so it is retained for inspection rather than advertised as a self-contained plotting command.
