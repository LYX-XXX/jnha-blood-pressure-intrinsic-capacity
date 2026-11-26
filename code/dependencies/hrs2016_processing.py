from __future__ import annotations

import io
import json
import re
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


SOURCE_DIR = Path(
    r"C:\Users\SAMSUNG\Desktop\AI\数据库论文\数据\HRS原始数据\HRS数据"
)
WORK_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = WORK_DIR.parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

HARMONIZED_ZIP = SOURCE_DIR / "H_HRS_d_stata.zip"
RAW_DATA_ZIP = SOURCE_DIR / "h16da.zip"
RAW_SCRIPT_ZIP = SOURCE_DIR / "h16sps.zip"
TRACKER_DTA = Path(r"C:\Users\SAMSUNG\Desktop\AI\数据库论文\数据\HRS原始数据\trk2022v1\trk2022tr_r.dta")


def parse_sps_layout(text: str) -> dict[str, tuple[int, int, bool]]:
    layout: dict[str, tuple[int, int, bool]] = {}
    match = re.search(r"DATA LIST FILE=.*?/(.*?)\n\s*\.", text, flags=re.I | re.S)
    if not match:
        raise ValueError("SPSS DATA LIST block not found")
    for line in match.group(1).splitlines():
        m = re.match(r"\s*([A-Za-z0-9_]+)\s+(\d+)\s*-\s*(\d+)(?:\s+\(A\))?", line)
        if m:
            name, start, end = m.group(1), int(m.group(2)), int(m.group(3))
            layout[name] = (start - 1, end, "(A)" in line)
    return layout


def read_raw_module(module: str, columns: list[str]) -> pd.DataFrame:
    script_name = f"H16{module}.sps"
    data_name = f"H16{module}.da"
    with zipfile.ZipFile(RAW_SCRIPT_ZIP) as scripts:
        script = scripts.read(script_name).decode("latin1").replace("\r", "")
    layout = parse_sps_layout(script)
    missing = [c for c in columns if c not in layout]
    if missing:
        raise KeyError(f"{module}: fields not found: {missing}")
    with zipfile.ZipFile(RAW_DATA_ZIP) as data:
        lines = data.read(data_name).decode("latin1").splitlines()
    result: dict[str, list[object]] = {c: [] for c in columns}
    for line in lines:
        for column in columns:
            start, end, is_text = layout[column]
            value = line[start:end].strip()
            if is_text:
                result[column].append(value or pd.NA)
            else:
                result[column].append(float(value) if value else np.nan)
    return pd.DataFrame(result)


def add_key(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["hhidpn"] = pd.to_numeric(
        frame["HHID"].astype("string").str.zfill(6)
        + frame["PN"].astype("string").str.zfill(3),
        errors="coerce",
    ).astype("Int64")
    return frame.drop(columns=["HHID", "PN"])


def valid_binary(series: pd.Series) -> pd.Series:
    return series.where(series.isin([1, 5]))


def standardize(series: pd.Series, mask: pd.Series) -> pd.Series:
    mean = series.loc[mask].mean()
    sd = series.loc[mask].std(ddof=1)
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(np.nan, index=series.index)
    return (series - mean) / sd


def effective_n(weights: pd.Series) -> float:
    weights = weights.dropna()
    weights = weights[weights > 0]
    if weights.empty:
        return float("nan")
    return float(weights.sum() ** 2 / (weights.pow(2).sum()))


def main() -> None:
    raw_a = add_key(
        read_raw_module(
            "A_R", ["HHID", "PN", "PA002", "PA009", "PA019"]
        )
    )
    raw_c = add_key(read_raw_module("C_R", ["HHID", "PN", "PC005"]))
    recall_i = [f"PD182M{i}" for i in range(1, 19)]
    recall_d = [f"PD183M{i}" for i in range(1, 20)]
    cesd = [f"PD{i}" for i in range(110, 118)]
    raw_d = add_key(
        read_raw_module(
            "D_R",
            ["HHID", "PN", *recall_i, *recall_d, "PD151", "PD152", "PD153", "PD154", *cesd],
        )
    )
    raw_g = add_key(
        read_raw_module("G_R", ["HHID", "PN", "PG001", "PG003", "PG005"])
    )

    harmonized_columns = [
        "hhidpn",
        "inw13pm",
        "r13nwtresp",
        "r13rxhibp",
        "r13systo1",
        "r13systo2",
        "r13systo3",
        "r13diasto1",
        "r13diasto2",
        "r13diasto3",
        "r13orient",
        "r13sight",
        "r13hearing",
        "r13gripsum",
        "r13lgrip",
        "r13rgrip",
        "r13wspeed",
        "r13mbmi",
        "r13bpcomp",
    ]
    with zipfile.ZipFile(HARMONIZED_ZIP) as archive:
        with archive.open("H_HRS_d.dta") as stream:
            harmonized = pd.read_stata(
                stream,
                columns=harmonized_columns,
                convert_categoricals=False,
                preserve_dtypes=False,
            )
    harmonized["hhidpn"] = pd.to_numeric(harmonized["hhidpn"], errors="coerce").astype(
        "Int64"
    )
    tracker = pd.read_stata(TRACKER_DTA, convert_categoricals=False, columns=["HHID", "PN", "SEX"])
    tracker["hhidpn"] = pd.to_numeric(tracker["HHID"] + tracker["PN"], errors="coerce").astype("Int64")
    tracker = tracker[["hhidpn", "SEX"]].rename(columns={"SEX": "tracker_sex"})

    data = raw_a.merge(raw_c, on="hhidpn", how="left", validate="one_to_one")
    data = data.merge(raw_d, on="hhidpn", how="left", validate="one_to_one")
    data = data.merge(raw_g, on="hhidpn", how="left", validate="one_to_one")
    data = data.merge(harmonized, on="hhidpn", how="left", validate="one_to_one")
    data = data.merge(tracker, on="hhidpn", how="left", validate="one_to_one")

    data = data.rename(
        columns={
            "PA019": "age_years",
            "PA009": "interview_self_proxy",
            "PC005": "doctor_diagnosed_hypertension",
            "r13rxhibp": "taking_antihypertensive",
            "r13nwtresp": "physical_measure_weight",
        }
    )

    # Blood pressure cleaning and visit-level exposure construction.
    valid_pairs = []
    for reading in (1, 2, 3):
        sbp = data[f"r13systo{reading}"]
        dbp = data[f"r13diasto{reading}"]
        valid = (
            sbp.between(70, 250, inclusive="both")
            & dbp.between(30, 150, inclusive="both")
            & (sbp > dbp)
        )
        data[f"bp{reading}_valid"] = valid
        valid_pairs.append(valid)
    data["bp_valid_readings"] = pd.concat(valid_pairs, axis=1).sum(axis=1)

    sbp_values = pd.DataFrame(
        {
            f"v{i}": data[f"r13systo{i}"].where(data[f"bp{i}_valid"])
            for i in (1, 2, 3)
        }
    )
    dbp_values = pd.DataFrame(
        {
            f"v{i}": data[f"r13diasto{i}"].where(data[f"bp{i}_valid"])
            for i in (1, 2, 3)
        }
    )
    both_23 = data["bp2_valid"] & data["bp3_valid"]
    data["sbp_mean"] = sbp_values.mean(axis=1, skipna=True)
    data["dbp_mean"] = dbp_values.mean(axis=1, skipna=True)
    data.loc[both_23, "sbp_mean"] = sbp_values.loc[both_23, ["v2", "v3"]].mean(axis=1)
    data.loc[both_23, "dbp_mean"] = dbp_values.loc[both_23, ["v2", "v3"]].mean(axis=1)
    data.loc[data["bp_valid_readings"] < 2, ["sbp_mean", "dbp_mean"]] = np.nan
    data["pulse_pressure"] = data["sbp_mean"] - data["dbp_mean"]
    data["mean_arterial_pressure"] = data["dbp_mean"] + data["pulse_pressure"] / 3

    data["bp_phenotype"] = pd.Series(pd.NA, index=data.index, dtype="string")
    valid_bp = data["sbp_mean"].notna() & data["dbp_mean"].notna()
    data.loc[valid_bp & (data["dbp_mean"] >= 90), "bp_phenotype"] = (
        "E DBP>=90"
    )
    data.loc[
        valid_bp & (data["dbp_mean"] < 60) & (data["sbp_mean"] < 140),
        "bp_phenotype",
    ] = "B SBP<140 DBP<60"
    data.loc[
        valid_bp
        & data["dbp_mean"].between(60, 89.999, inclusive="both")
        & (data["sbp_mean"] < 140),
        "bp_phenotype",
    ] = "A SBP<140 DBP60-89"
    data.loc[
        valid_bp
        & data["dbp_mean"].between(60, 89.999, inclusive="both")
        & (data["sbp_mean"] >= 140),
        "bp_phenotype",
    ] = "C SBP>=140 DBP60-89"
    data.loc[
        valid_bp & (data["dbp_mean"] < 60) & (data["sbp_mean"] >= 140),
        "bp_phenotype",
    ] = "D SBP>=140 DBP<60"

    # HRS word-response codes 1-40 are words from the four official target lists;
    # 51+ are explicitly labelled wrong words. Count only target-word codes.
    data["cog_immediate_correct"] = data[recall_i].apply(lambda s: s.between(1, 40)).sum(axis=1).where(data[recall_i].notna().any(axis=1))
    data["cog_delayed_correct"] = data[recall_d].apply(lambda s: s.between(1, 40)).sum(axis=1).where(data[recall_d].notna().any(axis=1))
    orient_items = ["PD151", "PD152", "PD153", "PD154"]
    data["cog_orientation_correct"] = data[orient_items].eq(1).sum(axis=1).where(data[orient_items].notna().any(axis=1))
    data["cognition_raw"] = data[["cog_immediate_correct", "cog_delayed_correct", "cog_orientation_correct"]].mean(axis=1).where(
        data[["cog_immediate_correct", "cog_delayed_correct", "cog_orientation_correct"]].notna().all(axis=1)
    )
    data["cognition_available"] = data["cognition_raw"].notna()

    cesd_valid = data[cesd].where(data[cesd].isin([1, 5]))
    negative_items = ["PD110", "PD111", "PD112", "PD114", "PD116", "PD117"]
    positive_items = ["PD113", "PD115"]
    psych_parts = pd.DataFrame(index=data.index)
    for column in negative_items:
        psych_parts[column] = cesd_valid[column].map({1: 0.0, 5: 1.0})
    for column in positive_items:
        psych_parts[column] = cesd_valid[column].map({1: 1.0, 5: 0.0})
    data["psychology_raw"] = psych_parts.mean(axis=1).where(
        psych_parts.notna().sum(axis=1) >= 6
    )
    data["psychology_available"] = data["psychology_raw"].notna()

    movement_parts = pd.DataFrame(index=data.index)
    for column in ["PG001", "PG003", "PG005"]:
        movement_parts[column] = valid_binary(data[column]).map({1: 0.0, 5: 1.0})
    data["locomotion_raw"] = movement_parts.mean(axis=1).where(
        movement_parts.notna().sum(axis=1) >= 2
    )
    data["locomotion_available"] = data["locomotion_raw"].notna()

    sight = data["r13sight"].where(data["r13sight"].between(1, 5))
    hearing = data["r13hearing"].where(data["r13hearing"].between(1, 5))
    data["sensory_raw"] = pd.concat([6 - sight, 6 - hearing], axis=1).mean(
        axis=1, skipna=False
    )
    data["sensory_available"] = data["sensory_raw"].notna()

    grip = data[["r13lgrip", "r13rgrip"]].where(data[["r13lgrip", "r13rgrip"]].gt(0) & data[["r13lgrip", "r13rgrip"]].le(100))
    data["vitality_raw"] = grip.max(axis=1).where(grip.notna().any(axis=1))
    data["vitality_available"] = data["vitality_raw"].notna()

    domain_cols = [
        "cognition_available",
        "psychology_available",
        "locomotion_available",
        "sensory_available",
        "vitality_available",
    ]
    data["ic_domains_available"] = data[domain_cols].sum(axis=1)
    data["ic_minimum_rule"] = (
        (data["ic_domains_available"] >= 4)
        & data["cognition_available"]
        & data["psychology_available"]
    )
    data["ic_five_domains_complete"] = data["ic_domains_available"].eq(5)

    # Fixed screening order from the protocol.
    flags = {
        "2016 completed respondent interview": data["PA002"].eq(1),
        "Age >=65 years": data["age_years"].between(65, 120),
        "Self interview": data["interview_self_proxy"].eq(1),
        "Doctor-diagnosed hypertension": data[
            "doctor_diagnosed_hypertension"
        ].eq(1),
        "Currently taking antihypertensive medication": data[
            "taking_antihypertensive"
        ].eq(1),
        "At least 2 valid BP readings": data["bp_valid_readings"].ge(2),
        "IC minimum rule (>=4 domains, cognition and psychology)": data[
            "ic_minimum_rule"
        ],
        "Positive physical-measure weight": data["physical_measure_weight"].gt(0),
    }
    cumulative = pd.Series(True, index=data.index)
    flow_rows = []
    for order, (label, flag) in enumerate(flags.items(), start=1):
        previous_n = int(cumulative.sum())
        cumulative &= flag.fillna(False)
        current_n = int(cumulative.sum())
        flow_rows.append(
            {
                "order": order,
                "criterion": label,
                "n_remaining": current_n,
                "n_excluded_at_step": previous_n - current_n,
                "percent_of_start": current_n / int(flags[next(iter(flags))].fillna(False).sum())
                if int(flags[next(iter(flags))].fillna(False).sum())
                else np.nan,
            }
        )
    data["analysis_eligible"] = cumulative

    # Construct IC only after eligibility is fixed; no group comparisons are produced.
    score_base = data["analysis_eligible"]
    raw_scores = {
        "ic_psychology_z": data["psychology_raw"],
        "ic_locomotion_z": data["locomotion_raw"],
        "ic_sensory_z": data["sensory_raw"],
    }
    for column, raw in raw_scores.items():
        data[column] = standardize(raw, score_base & raw.notna())
    cognition_parts = []
    for source in ["cog_immediate_correct", "cog_delayed_correct", "cog_orientation_correct"]:
        z = standardize(data[source], score_base & data[source].notna())
        cognition_parts.append(z)
    data["ic_cognition_z"] = pd.concat(cognition_parts, axis=1).mean(axis=1).where(data["cognition_available"])
    data["ic_vitality_z"] = np.nan
    for sex_code in (1, 2):
        sex_mask = score_base & data["tracker_sex"].eq(sex_code) & data["vitality_raw"].notna()
        data.loc[data["tracker_sex"].eq(sex_code), "ic_vitality_z"] = standardize(data["vitality_raw"], sex_mask)
    z_cols = ["ic_cognition_z", "ic_psychology_z", "ic_locomotion_z", "ic_sensory_z", "ic_vitality_z"]
    data["ic_mean_z"] = data[z_cols].mean(axis=1).where(data["ic_minimum_rule"])
    data["ic_total_z"] = standardize(
        data["ic_mean_z"], score_base & data["ic_mean_z"].notna()
    )

    audit_base = (
        flags["2016 completed respondent interview"]
        & flags["Age >=65 years"]
        & flags["Self interview"]
        & flags["Doctor-diagnosed hypertension"]
        & flags["Currently taking antihypertensive medication"]
        & flags["At least 2 valid BP readings"]
    )
    availability_rows = []
    for label, column in [
        ("Cognition", "cognition_available"),
        ("Psychology", "psychology_available"),
        ("Locomotion", "locomotion_available"),
        ("Sensory", "sensory_available"),
        ("Vitality", "vitality_available"),
        ("At least 4 domains plus cognition and psychology", "ic_minimum_rule"),
        ("All 5 domains", "ic_five_domains_complete"),
        ("Positive physical-measure weight", "physical_measure_weight"),
    ]:
        available = data[column].gt(0) if column == "physical_measure_weight" else data[column]
        denominator = int(audit_base.sum())
        numerator = int((audit_base & available.fillna(False)).sum())
        availability_rows.append(
            {
                "item": label,
                "n_available": numerator,
                "n_denominator": denominator,
                "percent_available": numerator / denominator if denominator else np.nan,
            }
        )

    phenotype_rows = []
    eligible = data["analysis_eligible"] & data["bp_phenotype"].notna()
    total_weight = data.loc[eligible, "physical_measure_weight"].sum()
    for phenotype in [
        "A SBP<140 DBP60-89",
        "B SBP<140 DBP<60",
        "C SBP>=140 DBP60-89",
        "D SBP>=140 DBP<60",
        "E DBP>=90",
    ]:
        mask = eligible & data["bp_phenotype"].eq(phenotype)
        weights = data.loc[mask, "physical_measure_weight"]
        phenotype_rows.append(
            {
                "bp_phenotype": phenotype,
                "unweighted_n": int(mask.sum()),
                "weighted_percent": weights.sum() / total_weight if total_weight else np.nan,
                "effective_sample_size": effective_n(weights),
                "mean_sbp": data.loc[mask, "sbp_mean"].mean(),
                "mean_dbp": data.loc[mask, "dbp_mean"].mean(),
                "mean_pulse_pressure": data.loc[mask, "pulse_pressure"].mean(),
            }
        )

    mapping_rows = [
        ("Participant identifier", "hhidpn", "Harmonized HRS D", "numeric HHID+PN"),
        ("Age", "PA019", "2016 Core A_R", "65-120 years retained"),
        ("Self/proxy interview", "PA009", "2016 Core A_R", "1=self"),
        ("Doctor-diagnosed hypertension", "PC005", "2016 Core C_R", "1=yes"),
        ("Current antihypertensive use", "r13rxhibp", "Harmonized HRS D", "1=yes"),
        ("SBP readings 1-3", "r13systo1-3", "Harmonized HRS D", "mmHg"),
        ("DBP readings 1-3", "r13diasto1-3", "Harmonized HRS D", "mmHg"),
        ("Physical-measure weight", "r13nwtresp", "Harmonized HRS D", "positive values"),
        ("Cognition", "PD182M1-18; PD183M1-19; PD151-154", "2016 Core D_R", "codes 1-40 only as target words; four date items code 1 correct"),
        ("Psychology", "PD110-PD117", "2016 Core D_R", "8-item CES-D direction; >=6 available"),
        ("Locomotion", "PG001; PG003; PG005", "2016 Core G_R", "walking/chair difficulty; >=2 available"),
        ("Sensory", "r13sight; r13hearing", "Harmonized HRS D", "1 excellent to 5 poor; both required"),
        ("Vitality", "max(r13lgrip,r13rgrip)", "Harmonized HRS D", "maximum valid value across both hands; sex-stratified z score"),
        ("PSU / stratum", "not present in supplied files", "Unavailable", "required before survey regression"),
    ]

    flow = pd.DataFrame(flow_rows)
    availability = pd.DataFrame(availability_rows)
    phenotypes = pd.DataFrame(phenotype_rows)
    mapping = pd.DataFrame(
        mapping_rows, columns=["concept", "source_variable", "source_file", "processing_rule"]
    )
    checks = pd.DataFrame(
        [
            ("Raw A_R duplicate hhidpn", int(raw_a["hhidpn"].duplicated().sum())),
            ("Raw C_R duplicate hhidpn", int(raw_c["hhidpn"].duplicated().sum())),
            ("Raw D_R duplicate hhidpn", int(raw_d["hhidpn"].duplicated().sum())),
            ("Raw G_R duplicate hhidpn", int(raw_g["hhidpn"].duplicated().sum())),
            ("Harmonized duplicate hhidpn", int(harmonized["hhidpn"].duplicated().sum())),
            ("Eligible missing phenotype", int((data["analysis_eligible"] & data["bp_phenotype"].isna()).sum())),
            ("Eligible nonpositive weight", int((data["analysis_eligible"] & ~data["physical_measure_weight"].gt(0)).sum())),
            ("Eligible SBP <= DBP", int((data["analysis_eligible"] & (data["sbp_mean"] <= data["dbp_mean"])).sum())),
        ],
        columns=["check", "issue_count"],
    )

    export_columns = [
        "hhidpn",
        "tracker_sex",
        "age_years",
        "interview_self_proxy",
        "doctor_diagnosed_hypertension",
        "taking_antihypertensive",
        "physical_measure_weight",
        "r13systo1",
        "r13systo2",
        "r13systo3",
        "r13diasto1",
        "r13diasto2",
        "r13diasto3",
        "bp1_valid",
        "bp2_valid",
        "bp3_valid",
        "bp_valid_readings",
        "sbp_mean",
        "dbp_mean",
        "pulse_pressure",
        "mean_arterial_pressure",
        "bp_phenotype",
        "cog_immediate_correct",
        "cog_delayed_correct",
        "cog_orientation_correct",
        "cognition_raw",
        "psychology_raw",
        "locomotion_raw",
        "sensory_raw",
        "vitality_raw",
        *domain_cols,
        "ic_domains_available",
        "ic_minimum_rule",
        "ic_five_domains_complete",
        *z_cols,
        "ic_mean_z",
        "ic_total_z",
        "r13wspeed",
        "r13mbmi",
        "analysis_eligible",
    ]
    processed = data.loc[
        flags["2016 completed respondent interview"], export_columns
    ].copy()
    processed.to_csv(
        OUTPUT_DIR / "HRS2016_analysis_ready.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary = {
        "source_version": "Gateway Harmonized HRS Aug 2023 Version D and HRS 2016 Core Final V2.0",
        "flow": flow.to_dict(orient="records"),
        "availability": availability.to_dict(orient="records"),
        "phenotypes": phenotypes.to_dict(orient="records"),
        "mapping": mapping.to_dict(orient="records"),
        "checks": checks.to_dict(orient="records"),
        "notes": [
            "Blind audit only: no BP phenotype versus IC association was estimated.",
            "STRATUM, SECU and PPMWGTR are available from Tracker 2022 Final V1 and merged in the tracker-enriched output.",
        ],
    }
    (WORK_DIR / "hrs2016_audit.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    flow.to_csv(OUTPUT_DIR / "HRS2016_sample_flow.csv", index=False, encoding="utf-8-sig")
    availability.to_csv(
        OUTPUT_DIR / "HRS2016_ic_availability.csv", index=False, encoding="utf-8-sig"
    )
    phenotypes.to_csv(
        OUTPUT_DIR / "HRS2016_bp_phenotypes.csv", index=False, encoding="utf-8-sig"
    )
    mapping.to_csv(
        OUTPUT_DIR / "HRS2016_variable_mapping.csv", index=False, encoding="utf-8-sig"
    )
    checks.to_csv(OUTPUT_DIR / "HRS2016_checks.csv", index=False, encoding="utf-8-sig")

    destination = OUTPUT_DIR / "hrs2016_processing.py"
    if Path(__file__).resolve() != destination.resolve():
        shutil.copy2(__file__, destination)
    print(flow.to_string(index=False))
    print("\nPhenotypes")
    print(phenotypes.to_string(index=False))
    print("\nChecks")
    print(checks.to_string(index=False))


if __name__ == "__main__":
    main()
