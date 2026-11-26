# Additional exploratory sensitivities, 26 November 2026

These analyses refit CHARLS and ELSA only. No HRS participant-level models were fitted. The original two cohort models and SEs reproduced the corrected archive to absolute tolerance 1e-8.

## Findings and interpretation

ELSA intervals <=6 months retained n=987 (A=458, B=153), IC difference -0.340 (95% CI -0.543 to -0.138). Intervals <=12 months retained n=996 (A=462, B=157), difference -0.324 (-0.525 to -0.123). The six-month window was designated the primary additional timing sensitivity before fitting; all these checks remain post-review exploratory analyses, not preregistered analyses. The two windows overlap substantially.

Replacing only the ELSA estimate in three-cohort pooling retained negative fixed-effect estimates, with modified Knapp-Hartung random-effects intervals including zero. HRS and CHARLS estimates in this timing pooling are archived aggregates.

Uniform final-sample domain-level rescaling gave CHARLS -0.124 (-0.462 to 0.214), ELSA -0.309 (-0.496 to -0.122). No uniform-score HRS estimate or three-cohort uniform-score pooling is provided.

ELSA vitality intervals in both short-window samples include zero. The domain-MI sensitivity also shows a negative locomotion estimate in ELSA, so these checks do not establish vitality-specific association.

## Methods

Model 2 and the original 20 completed covariate datasets were retained. For ELSA outcome-availability and timing subpopulations, all 1377 original household clusters in nine strata were retained with zero score contributions outside the fitted subpopulation. This is survey domain estimation; participants outside the domain were not included in coefficient estimation. CHARLS retains HC1 covariance.

Uniform scores restandardize each non-vitality domain within the final eligible cohort, restandardize vitality within sex, average available domains using original eligibility, and restandardize the composite, using sample SDs. Domain outcomes were not imputed. Existing imputations were reused rather than refitted with individual domain outcomes. Domain B-versus-A p values are BH-adjusted across five tests separately within each of the two cohorts; this differs from the original 10-test pooled family. Main manuscript domain estimates are not replaced by these exploratory checks.

See analysis_specification.json for software and hashes. New analyses ran in Python 3.11; reproduction of the original estimates checks numerical consistency across environments.

## Running the script

Set JNHA_PRIOR_WORK to your authorized corrected modeling/imputation input directory, JNHA_COHORT_DATA_ROOT to the local cohort-data directory, JNHA_OUTPUT_DIR to a writable aggregate-output directory, and JNHA_ARCHIVE_ROOT to this repository root. Required libraries are numpy, pandas, scipy, patsy and pyreadstat. Input names and required ELSA fields are explicit in the script. Only CHARLS and ELSA rows are retained for individual-level analysis. Comply with your cohort agreements; no participant data should be committed to this repository.

The remaining HRS uniform-score analysis must be performed by an analyst whose workflow complies with HRS terms. Its absence is not evidence that the new score is equivalent in HRS.
