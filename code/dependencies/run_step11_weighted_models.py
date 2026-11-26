from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(__file__).resolve().parent / "three_cohort_modeling_combined.csv"
STEP = Path(__file__).resolve().parent / "step11_weighted_models"
STEP.mkdir(parents=True, exist_ok=True)

MODEL_COVARS = {
    "M0": ["age_per10", "female"],
    "M1": ["age_per10", "female", "edu_2", "edu_3", "partnered", "current_smoker", "current_alcohol"],
    "M2": [
        "age_per10", "female", "edu_2", "edu_3", "partnered", "current_smoker", "current_alcohol",
        "diabetes", "heart_disease", "stroke", "chronic_lung_disease", "cancer",
    ],
}

LABELS = {
    "intercept": "截距",
    "bp_B": "B组 vs A组",
    "bp_C": "C组 vs A组",
    "bp_D": "D组 vs A组",
    "bp_E": "E组 vs A组",
    "sbp_per10": "SBP每升高10 mmHg（DBP=80时）",
    "dbp_per10": "DBP每升高10 mmHg（SBP=140时）",
    "sbp_dbp_interaction": "SBP×DBP交互（每10×10 mmHg）",
    "pp_per10": "脉压每升高10 mmHg",
    "map_per10": "MAP每升高10 mmHg",
    "bmi_underweight_cat": "低体重 vs 正常BMI",
    "bmi_overweight_cat": "超重 vs 正常BMI",
    "bmi_obesity_cat": "肥胖 vs 正常BMI",
    "moderate_or_vigorous_weekly": "每周中高强度活动",
}


def normal_p(z: float) -> float:
    return math.erfc(abs(z) / math.sqrt(2.0)) if np.isfinite(z) else np.nan


def prepare(data: pd.DataFrame) -> pd.DataFrame:
    d = data.copy()
    d["age_per10"] = (d["age_years"] - 65.0) / 10.0
    d["edu_2"] = d["education_level"].eq(2).astype(float).where(d["education_level"].notna())
    d["edu_3"] = d["education_level"].eq(3).astype(float).where(d["education_level"].notna())
    for group in "BCDE":
        d[f"bp_{group}"] = d["bp_group"].eq(group).astype(float).where(d["bp_group"].notna())
    d["sbp_per10"] = (d["sbp_mean"] - 140.0) / 10.0
    d["dbp_per10"] = (d["dbp_mean"] - 80.0) / 10.0
    d["sbp_dbp_interaction"] = d["sbp_per10"] * d["dbp_per10"]
    d["pp_per10"] = (d["pulse_pressure"] - 50.0) / 10.0
    d["map_per10"] = (d["mean_arterial_pressure"] - (80.0 + 50.0 / 3.0)) / 10.0
    d["bmi_underweight_cat"] = d["bmi_category_4"].eq("underweight").astype(float).where(d["bmi_category_4"].notna())
    d["bmi_overweight_cat"] = d["bmi_category_4"].eq("overweight").astype(float).where(d["bmi_category_4"].notna())
    d["bmi_obesity_cat"] = d["bmi_category_4"].eq("obesity").astype(float).where(d["bmi_category_4"].notna())
    return d


def survey_wls(
    data: pd.DataFrame,
    outcome: str,
    terms: list[str],
    cohort: str,
) -> tuple[pd.DataFrame, dict]:
    needed = [outcome, "survey_weight", *terms]
    if cohort != "CHARLS2015":
        needed += ["survey_stratum", "survey_psu"]
    use = data[needed].notna().all(axis=1) & data["survey_weight"].gt(0)
    sub = data.loc[use].copy()
    names = ["intercept", *terms]
    x = np.column_stack([np.ones(len(sub)), *[sub[t].to_numpy(float) for t in terms]])
    y = sub[outcome].to_numpy(float)
    w = sub["survey_weight"].to_numpy(float)
    w = w / np.mean(w)
    xtwx = x.T @ (w[:, None] * x)
    bread = np.linalg.pinv(xtwx, rcond=1e-12)
    beta = bread @ (x.T @ (w * y))
    resid = y - x @ beta
    scores = x * (w * resid)[:, None]

    singleton_strata = 0
    if cohort == "CHARLS2015":
        meat = scores.T @ scores
        if len(sub) > len(names):
            meat *= len(sub) / (len(sub) - len(names))
        clusters = len(sub)
        strata = 0
        variance_method = "weighted_HC1"
    else:
        meat = np.zeros((len(names), len(names)), dtype=float)
        strata = int(sub["survey_stratum"].nunique())
        clusters = int(sub[["survey_stratum", "survey_psu"]].drop_duplicates().shape[0])
        temp = pd.DataFrame(scores, columns=names, index=sub.index)
        temp["_stratum"] = sub["survey_stratum"].to_numpy()
        temp["_psu"] = sub["survey_psu"].to_numpy()
        cluster_scores = temp.groupby(["_stratum", "_psu"], sort=False)[names].sum().reset_index()
        for _, h in cluster_scores.groupby("_stratum", sort=False):
            u = h[names].to_numpy(float)
            j = len(u)
            if j >= 2:
                centered = u - u.mean(axis=0, keepdims=True)
                meat += (j / (j - 1.0)) * (centered.T @ centered)
            else:
                singleton_strata += 1
                meat += u.T @ u
        variance_method = "stratum_centered_PSU_linearization"
    cov = bread @ meat @ bread
    se = np.sqrt(np.clip(np.diag(cov), 0, np.inf))
    fitted = x @ beta
    ybar = np.average(y, weights=w)
    sse = np.sum(w * (y - fitted) ** 2)
    sst = np.sum(w * (y - ybar) ** 2)
    r2 = 1 - sse / sst if sst > 0 else np.nan
    rows = []
    for name, b, s in zip(names, beta, se):
        z = b / s if s > 0 else np.nan
        rows.append({
            "term": name,
            "term_label": LABELS.get(name, name),
            "beta": b,
            "standard_error": s,
            "ci95_lower": b - 1.959964 * s,
            "ci95_upper": b + 1.959964 * s,
            "z_value": z,
            "p_value": normal_p(z),
        })
    info = {
        "n": len(sub), "clusters": clusters, "strata": strata, "singleton_strata": singleton_strata,
        "weighted_r_squared": r2, "condition_number": float(np.linalg.cond(xtwx)),
        "variance_method": variance_method,
    }
    return pd.DataFrame(rows), info


def add_model(
    store: list[pd.DataFrame],
    data: pd.DataFrame,
    cohort: str,
    outcome: str,
    family: str,
    adjustment: str,
    exposure_terms: list[str],
    extra_covars: list[str] | None = None,
    restriction: pd.Series | None = None,
) -> None:
    sub = data.loc[restriction].copy() if restriction is not None else data
    terms = [*exposure_terms, *MODEL_COVARS[adjustment], *(extra_covars or [])]
    results, info = survey_wls(sub, outcome, terms, cohort)
    for key, value in {
        "cohort": cohort, "outcome": outcome, "model_family": family, "adjustment": adjustment,
        **info,
    }.items():
        results[key] = value
    results["reference"] = "A组" if family.startswith("BP_category") else "continuous"
    store.append(results)


def weighted_stats(values: pd.Series, weights: pd.Series) -> tuple[float, float, int]:
    ok = values.notna() & weights.notna() & weights.gt(0)
    x = values.loc[ok].to_numpy(float)
    w = weights.loc[ok].to_numpy(float)
    if len(x) == 0:
        return np.nan, np.nan, 0
    mean = np.average(x, weights=w)
    var = np.average((x - mean) ** 2, weights=w)
    return mean, math.sqrt(max(var, 0)), len(x)


def baseline_table(data: pd.DataFrame) -> pd.DataFrame:
    continuous = ["age_years", "sbp_mean", "dbp_mean", "pulse_pressure", "mean_arterial_pressure", "ic_total_z", "bmi"]
    binary = [
        "female", "partnered", "current_smoker", "current_alcohol", "diabetes", "heart_disease",
        "stroke", "chronic_lung_disease", "cancer", "moderate_or_vigorous_weekly", "bmi_underweight",
    ]
    rows = []
    for cohort, cohort_data in data.groupby("cohort", sort=False):
        groups = [("Overall", cohort_data), *[(g, cohort_data[cohort_data["bp_group"].eq(g)]) for g in "ABCDE"]]
        for group, sub in groups:
            for variable in continuous:
                mean, sd, n_valid = weighted_stats(sub[variable], sub["survey_weight"])
                rows.append({
                    "cohort": cohort, "bp_group": group, "variable": variable, "variable_type": "continuous",
                    "unweighted_n": len(sub), "valid_n": n_valid, "weighted_mean_or_proportion": mean,
                    "weighted_sd": sd, "weighted_percent": np.nan,
                })
            for variable in binary:
                mean, sd, n_valid = weighted_stats(sub[variable], sub["survey_weight"])
                rows.append({
                    "cohort": cohort, "bp_group": group, "variable": variable, "variable_type": "binary",
                    "unweighted_n": len(sub), "valid_n": n_valid, "weighted_mean_or_proportion": mean,
                    "weighted_sd": sd, "weighted_percent": mean * 100 if np.isfinite(mean) else np.nan,
                })
    return pd.DataFrame(rows)


def phenotype_overlap(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cohort, d in data.groupby("cohort", sort=False):
        w = d["survey_weight"]
        for label, mask in {
            "DBP<60": d["dbp_mean"].lt(60),
            "PP>=60": d["pulse_pressure"].ge(60),
            "DBP<60_and_PP>=60": d["dbp_mean"].lt(60) & d["pulse_pressure"].ge(60),
        }.items():
            rows.append({
                "cohort": cohort, "indicator": label, "unweighted_n": int(mask.sum()),
                "weighted_percent": float(100 * w[mask].sum() / w.sum()),
            })
        corr = np.corrcoef(d[["sbp_mean", "dbp_mean", "pulse_pressure", "mean_arterial_pressure"]].dropna().to_numpy().T)
        names = ["sbp_mean", "dbp_mean", "pulse_pressure", "mean_arterial_pressure"]
        for i, left in enumerate(names):
            for j in range(i + 1, len(names)):
                rows.append({
                    "cohort": cohort, "indicator": f"correlation_{left}_with_{names[j]}",
                    "unweighted_n": len(d), "weighted_percent": corr[i, j],
                })
    return pd.DataFrame(rows)


def main() -> None:
    raw = pd.read_csv(SOURCE, low_memory=False)
    data = prepare(raw)
    baseline = baseline_table(data)
    baseline.to_csv(STEP / "01_weighted_baseline_long.csv", index=False, encoding="utf-8-sig")
    phenotype_overlap(data).to_csv(STEP / "02_bp_overlap_and_correlations.csv", index=False, encoding="utf-8-sig")

    primary: list[pd.DataFrame] = []
    secondary: list[pd.DataFrame] = []
    sensitivities: list[pd.DataFrame] = []
    category = ["bp_B", "bp_C", "bp_D", "bp_E"]
    continuous = ["sbp_per10", "dbp_per10", "sbp_dbp_interaction"]
    for cohort, cohort_data in data.groupby("cohort", sort=False):
        for adjustment in ["M0", "M1", "M2"]:
            add_model(primary, cohort_data, cohort, "ic_total_z", "BP_category_A_reference", adjustment, category)
            add_model(primary, cohort_data, cohort, "ic_total_z", "SBP_DBP_continuous_interaction", adjustment, continuous)
            add_model(primary, cohort_data, cohort, "ic_total_z", "Pulse_pressure", adjustment, ["pp_per10"])
            add_model(primary, cohort_data, cohort, "ic_total_z", "Mean_arterial_pressure", adjustment, ["map_per10"])
        for outcome in ["ic_cognition_z", "ic_psychology_z", "ic_locomotion_z", "ic_sensory_z", "ic_vitality_z"]:
            add_model(secondary, cohort_data, cohort, outcome, "BP_category_A_reference", "M2", category)
            add_model(secondary, cohort_data, cohort, outcome, "SBP_DBP_continuous_interaction", "M2", continuous)
        add_model(
            sensitivities, cohort_data, cohort, "ic_total_z", "M2_plus_BMI_category", "M2", continuous,
            ["bmi_underweight_cat", "bmi_overweight_cat", "bmi_obesity_cat"],
        )
        add_model(
            sensitivities, cohort_data, cohort, "ic_total_z", "M2_plus_activity", "M2", continuous,
            ["moderate_or_vigorous_weekly"],
        )
        add_model(
            sensitivities, cohort_data, cohort, "ic_total_z", "Five_domains_complete", "M2", continuous,
            restriction=cohort_data["ic_five_domains_complete"].eq(True),
        )

    primary_df = pd.concat(primary, ignore_index=True)
    secondary_df = pd.concat(secondary, ignore_index=True)
    sensitivity_df = pd.concat(sensitivities, ignore_index=True)
    primary_df.to_csv(STEP / "03_primary_complete_case_models.csv", index=False, encoding="utf-8-sig")
    secondary_df.to_csv(STEP / "04_secondary_domain_models.csv", index=False, encoding="utf-8-sig")
    sensitivity_df.to_csv(STEP / "05_sensitivity_models.csv", index=False, encoding="utf-8-sig")

    audit = {
        "source": str(SOURCE),
        "created": "2026-10-02",
        "rows_by_cohort": raw.groupby("cohort").size().to_dict(),
        "primary_model_rows": len(primary_df),
        "secondary_model_rows": len(secondary_df),
        "sensitivity_model_rows": len(sensitivity_df),
        "variance": {
            "HRS2016": "stratum-centered PSU linearization",
            "ELSA_W8W9": "stratum-centered PSU linearization",
            "CHARLS2015": "weighted HC1 because official stratum/PSU unavailable",
        },
        "wealth": "not in main models; reserved for an optional within-cohort rank sensitivity analysis",
        "missing_covariates": "complete-case estimates in this step; multiple imputation remains a separate next step",
    }
    (STEP / "06_step11_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    focus = primary_df[
        primary_df["adjustment"].eq("M2")
        & primary_df["term"].isin(["bp_B", "bp_C", "bp_D", "bp_E", "sbp_per10", "dbp_per10", "sbp_dbp_interaction"])
    ]
    print(focus[["cohort", "model_family", "term", "beta", "ci95_lower", "ci95_upper", "p_value", "n"]].to_string(index=False))


if __name__ == "__main__":
    main()
