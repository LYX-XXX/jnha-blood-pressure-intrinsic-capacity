from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "work" / "mi_runtime"))

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.imputation.mice import MICEData

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_step11_weighted_models import MODEL_COVARS, prepare, survey_wls


OUT = Path(__file__).resolve().parent
SOURCE = OUT / "three_cohort_modeling_combined.csv"
CC_RESULTS = OUT / "step11_weighted_models" / "03_primary_complete_case_models.csv"
STEP = OUT / "step12_multiple_imputation"
STEP.mkdir(parents=True, exist_ok=True)

M = 20
BURN_IN = 10
BETWEEN_SAMPLES = 2
SEEDS = {"HRS2016": 20261021, "CHARLS2015": 20261022, "ELSA_W8W9": 20261023}

TARGETS = [
    "education_level", "current_smoker", "current_alcohol", "diabetes", "heart_disease",
    "stroke", "chronic_lung_disease", "cancer", "moderate_or_vigorous_weekly", "bmi",
]
BINARY_TARGETS = [
    "current_smoker", "current_alcohol", "diabetes", "heart_disease", "stroke",
    "chronic_lung_disease", "cancer", "moderate_or_vigorous_weekly",
]
IMPUTATION_COLUMNS = [
    "education_level", "current_smoker", "current_alcohol", "diabetes", "heart_disease",
    "stroke", "chronic_lung_disease", "cancer", "moderate_or_vigorous_weekly", "bmi",
    "age_z", "female", "partnered", "sbp_z", "dbp_z", "ic_total_z", "log_weight_z", "stratum_z",
]


def zscore(x: pd.Series) -> pd.Series:
    values = pd.to_numeric(x, errors="coerce")
    sd = values.std(ddof=0)
    return (values - values.mean()) / sd if np.isfinite(sd) and sd > 0 else pd.Series(0.0, index=x.index)


def imputation_frame(d: pd.DataFrame) -> pd.DataFrame:
    frame = d[TARGETS].apply(pd.to_numeric, errors="coerce").copy()
    frame["age_z"] = zscore(d["age_years"])
    frame["female"] = pd.to_numeric(d["female"], errors="coerce")
    frame["partnered"] = pd.to_numeric(d["partnered"], errors="coerce")
    frame["sbp_z"] = zscore(d["sbp_mean"])
    frame["dbp_z"] = zscore(d["dbp_mean"])
    frame["ic_total_z"] = pd.to_numeric(d["ic_total_z"], errors="coerce")
    frame["log_weight_z"] = zscore(np.log(pd.to_numeric(d["survey_weight"], errors="coerce")))
    frame["stratum_z"] = zscore(d["survey_stratum"])
    return frame[IMPUTATION_COLUMNS]


def run_imputation(cohort: str, original: pd.DataFrame) -> tuple[list[pd.DataFrame], pd.DataFrame, pd.DataFrame]:
    np.random.seed(SEEDS[cohort])
    model_data = imputation_frame(original)
    # statsmodels 0.14 checks for the traditional object-backed string index;
    # pandas 3 may otherwise supply its newer dedicated string dtype.
    model_data.columns = pd.Index([str(c) for c in model_data.columns], dtype=object)
    missing_masks = {v: model_data[v].isna().copy() for v in TARGETS}
    mice = MICEData(model_data, perturbation_method="gaussian", k_pmm=20)
    mice.update_all(BURN_IN)
    completed = []
    diagnostic_rows = []
    long_rows = []
    for imp in range(1, M + 1):
        mice.update_all(BETWEEN_SAMPLES)
        sample = mice.data.copy()
        rebuilt = original.copy()
        for variable in TARGETS:
            rebuilt[variable] = sample[variable].to_numpy()
        # PMM should retain donor-supported values. Fail loudly if a coding invariant is violated.
        if not rebuilt["education_level"].isin([1, 2, 3]).all():
            raise ValueError(f"{cohort} imputation {imp}: invalid education category")
        for variable in BINARY_TARGETS:
            if not rebuilt[variable].isin([0, 1]).all():
                raise ValueError(f"{cohort} imputation {imp}: invalid binary {variable}")
        if not rebuilt["bmi"].between(10, 80).all():
            raise ValueError(f"{cohort} imputation {imp}: invalid BMI")
        rebuilt = prepare(rebuilt)
        completed.append(rebuilt)

        covariate_export = rebuilt[["participant_id", *TARGETS]].copy()
        covariate_export.insert(0, "imputation", imp)
        covariate_export.insert(0, "cohort", cohort)
        long_rows.append(covariate_export)
        for variable in TARGETS:
            missing = missing_masks[variable]
            observed = model_data.loc[~missing, variable]
            imputed = sample.loc[missing, variable]
            diagnostic_rows.append({
                "cohort": cohort,
                "imputation": imp,
                "variable": variable,
                "original_missing_n": int(missing.sum()),
                "original_missing_percent": float(missing.mean() * 100),
                "observed_mean": float(observed.mean()) if len(observed) else np.nan,
                "observed_sd": float(observed.std(ddof=1)) if len(observed) > 1 else np.nan,
                "imputed_mean": float(imputed.mean()) if len(imputed) else np.nan,
                "imputed_sd": float(imputed.std(ddof=1)) if len(imputed) > 1 else np.nan,
                "imputed_min": float(imputed.min()) if len(imputed) else np.nan,
                "imputed_max": float(imputed.max()) if len(imputed) else np.nan,
            })
    return completed, pd.DataFrame(diagnostic_rows), pd.concat(long_rows, ignore_index=True)


def fit_one(
    data: pd.DataFrame,
    cohort: str,
    family: str,
    adjustment: str,
    exposure_terms: list[str],
    extra_covars: list[str] | None = None,
    restriction: pd.Series | None = None,
) -> pd.DataFrame:
    sub = data.loc[restriction].copy() if restriction is not None else data
    terms = [*exposure_terms, *MODEL_COVARS[adjustment], *(extra_covars or [])]
    result, info = survey_wls(sub, "ic_total_z", terms, cohort)
    result["cohort"] = cohort
    result["model_family"] = family
    result["adjustment"] = adjustment
    for key, value in info.items():
        result[key] = value
    return result


def fit_all_models(data: pd.DataFrame, cohort: str, imp: int) -> pd.DataFrame:
    category = ["bp_B", "bp_C", "bp_D", "bp_E"]
    continuous = ["sbp_per10", "dbp_per10", "sbp_dbp_interaction"]
    stores = []
    for adjustment in ["M0", "M1", "M2"]:
        stores.append(fit_one(data, cohort, "BP_category_A_reference", adjustment, category))
        stores.append(fit_one(data, cohort, "SBP_DBP_continuous_interaction", adjustment, continuous))
    stores.append(fit_one(data, cohort, "Pulse_pressure", "M2", ["pp_per10"]))
    stores.append(fit_one(data, cohort, "Mean_arterial_pressure", "M2", ["map_per10"]))
    stores.append(fit_one(
        data, cohort, "M2_plus_BMI_category", "M2", continuous,
        ["bmi_underweight_cat", "bmi_overweight_cat", "bmi_obesity_cat"],
    ))
    stores.append(fit_one(
        data, cohort, "M2_plus_activity", "M2", continuous,
        ["moderate_or_vigorous_weekly"],
    ))
    stores.append(fit_one(
        data, cohort, "Five_domains_complete", "M2", continuous,
        restriction=data["ic_five_domains_complete"].eq(True),
    ))
    result = pd.concat(stores, ignore_index=True)
    result["imputation"] = imp
    return result


def rubin_pool(estimates: pd.DataFrame) -> pd.DataFrame:
    keys = ["cohort", "model_family", "adjustment", "term", "term_label"]
    rows = []
    for key, group in estimates.groupby(keys, dropna=False, sort=False):
        q = group["beta"].to_numpy(float)
        u = np.square(group["standard_error"].to_numpy(float))
        m = len(q)
        qbar = q.mean()
        ubar = u.mean()
        b = q.var(ddof=1) if m > 1 else 0.0
        total = ubar + (1 + 1 / m) * b
        se = math.sqrt(max(total, 0))
        if b > 0 and ubar > 0:
            r = (1 + 1 / m) * b / ubar
            df = (m - 1) * (1 + 1 / r) ** 2
        else:
            r, df = 0.0, np.inf
        statistic = qbar / se if se > 0 else np.nan
        p = 2 * stats.t.sf(abs(statistic), df) if np.isfinite(df) else 2 * stats.norm.sf(abs(statistic))
        critical = stats.t.ppf(0.975, df) if np.isfinite(df) else stats.norm.ppf(0.975)
        lambda_mi = ((1 + 1 / m) * b / total) if total > 0 else 0.0
        rows.append({
            **dict(zip(keys, key)),
            "m": m,
            "beta_pooled": qbar,
            "standard_error_pooled": se,
            "ci95_lower": qbar - critical * se,
            "ci95_upper": qbar + critical * se,
            "test_statistic": statistic,
            "degrees_freedom": df,
            "p_value": p,
            "within_variance": ubar,
            "between_variance": b,
            "relative_increase_variance": r,
            "fraction_missing_information": lambda_mi,
            "n_min": int(group["n"].min()),
            "n_max": int(group["n"].max()),
            "variance_method": group["variance_method"].iloc[0],
        })
    return pd.DataFrame(rows)


def comparison_table(pooled: pd.DataFrame) -> pd.DataFrame:
    complete = pd.read_csv(CC_RESULTS)
    cc = complete[[
        "cohort", "model_family", "adjustment", "term", "beta", "standard_error", "ci95_lower", "ci95_upper", "p_value", "n",
    ]].rename(columns={
        "beta": "beta_complete_case", "standard_error": "se_complete_case",
        "ci95_lower": "cc_ci95_lower", "ci95_upper": "cc_ci95_upper", "p_value": "cc_p_value", "n": "cc_n",
    })
    out = pooled.merge(cc, on=["cohort", "model_family", "adjustment", "term"], how="left", validate="one_to_one")
    out["beta_difference_MI_minus_CC"] = out["beta_pooled"] - out["beta_complete_case"]
    return out


def main() -> None:
    source = pd.read_csv(SOURCE, dtype={"participant_id": "string", "survey_psu": "string"}, low_memory=False)
    all_estimates = []
    all_diagnostics = []
    all_long = []
    for cohort, original in source.groupby("cohort", sort=False):
        original = original.reset_index(drop=True)
        completed, diagnostics, long_data = run_imputation(cohort, original)
        all_diagnostics.append(diagnostics)
        all_long.append(long_data)
        for imp, completed_data in enumerate(completed, start=1):
            all_estimates.append(fit_all_models(completed_data, cohort, imp))
        print(f"{cohort}: completed {M} imputations")

    diagnostics = pd.concat(all_diagnostics, ignore_index=True)
    long_data = pd.concat(all_long, ignore_index=True)
    estimates = pd.concat(all_estimates, ignore_index=True)
    pooled = rubin_pool(estimates)
    comparison = comparison_table(pooled)

    diagnostics.to_csv(STEP / "01_imputation_diagnostics.csv", index=False, encoding="utf-8-sig")
    long_data.to_csv(STEP / "02_imputed_covariates_long.csv.gz", index=False, compression="gzip")
    estimates.to_csv(STEP / "03_imputation_specific_estimates.csv.gz", index=False, compression="gzip")
    pooled.to_csv(STEP / "04_Rubin_pooled_models.csv", index=False, encoding="utf-8-sig")
    comparison.to_csv(STEP / "05_MI_vs_complete_case.csv", index=False, encoding="utf-8-sig")

    key_terms = ["bp_B", "bp_C", "bp_D", "bp_E", "sbp_per10", "dbp_per10", "sbp_dbp_interaction", "pp_per10", "map_per10"]
    key = pooled.loc[
        pooled["adjustment"].eq("M2")
        & pooled["term"].isin(key_terms)
        & pooled["model_family"].isin([
            "BP_category_A_reference", "SBP_DBP_continuous_interaction", "Pulse_pressure", "Mean_arterial_pressure",
        ])
    ].copy()
    key.to_csv(STEP / "06_key_pooled_M2_results.csv", index=False, encoding="utf-8-sig")

    audit = {
        "created": "2026-10-02",
        "method": "statsmodels MICEData with Gaussian parameter perturbation and predictive mean matching",
        "m": M,
        "burn_in_cycles": BURN_IN,
        "cycles_between_saved_imputations": BETWEEN_SAMPLES,
        "k_pmm": 20,
        "seeds": SEEDS,
        "targets": TARGETS,
        "predictors": IMPUTATION_COLUMNS,
        "pooling": "Rubin rules with t-based confidence intervals",
        "wealth": "not imputed and not included in main models; optional future sensitivity only",
        "rows_long": len(long_data),
        "invalid_binary_values": int(sum((~long_data[v].isin([0, 1])).sum() for v in BINARY_TARGETS)),
        "invalid_education_values": int((~long_data["education_level"].isin([1, 2, 3])).sum()),
        "invalid_bmi_values": int((~long_data["bmi"].between(10, 80)).sum()),
    }
    (STEP / "07_step12_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    print(key[["cohort", "model_family", "term", "beta_pooled", "ci95_lower", "ci95_upper", "p_value", "fraction_missing_information"]].to_string(index=False))


if __name__ == "__main__":
    main()
