from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "work" / "pyreadstat_runtime2"))
import pyreadstat  # noqa: E402

OUT = Path(__file__).resolve().parent
CHARLS = Path(r"C:\Users\SAMSUNG\Desktop\AI\数据库论文\数据\【数据】Harmonized CHARLS\【数据】Harmonized CHARLS\Harmonized_CHARLS_D_Data.dta")
ELSA = Path(r"C:\Users\SAMSUNG\Desktop\AI\数据库论文\数据\【数据】Harmonized ELSA\【数据】Harmonized ELSA\h_elsa_g3.sav")


def zscore(series: pd.Series, reference: pd.Series) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce")
    ref = x[reference & x.notna()]
    sd = ref.std(ddof=0)
    return (x - ref.mean()) / sd if len(ref) > 1 and sd > 0 else pd.Series(np.nan, index=x.index)


def bp_construct(df: pd.DataFrame, prefix: str) -> None:
    valid_cols = []
    for i in (1, 2, 3):
        s, d = f"{prefix}systo{i}", f"{prefix}diasto{i}"
        valid = df[s].between(70, 250) & df[d].between(30, 150) & df[s].gt(df[d])
        df[f"bp{i}_valid"] = valid
        valid_cols.append(f"bp{i}_valid")
    df["bp_valid_readings"] = df[valid_cols].sum(axis=1)
    sbp = pd.concat([df[f"{prefix}systo{i}"].where(df[f"bp{i}_valid"]) for i in (1, 2, 3)], axis=1)
    dbp = pd.concat([df[f"{prefix}diasto{i}"].where(df[f"bp{i}_valid"]) for i in (1, 2, 3)], axis=1)
    df["sbp_mean"] = sbp.mean(axis=1).where(df["bp_valid_readings"].ge(2))
    df["dbp_mean"] = dbp.mean(axis=1).where(df["bp_valid_readings"].ge(2))
    use23 = df["bp2_valid"] & df["bp3_valid"]
    df.loc[use23, "sbp_mean"] = df.loc[use23, [f"{prefix}systo2", f"{prefix}systo3"]].mean(axis=1)
    df.loc[use23, "dbp_mean"] = df.loc[use23, [f"{prefix}diasto2", f"{prefix}diasto3"]].mean(axis=1)
    df["pulse_pressure"] = df["sbp_mean"] - df["dbp_mean"]
    df["mean_arterial_pressure"] = df["dbp_mean"] + df["pulse_pressure"] / 3
    df["bp_phenotype"] = pd.Series(pd.NA, index=df.index, dtype="string")
    ok = df["bp_valid_readings"].ge(2)
    df.loc[ok & df["dbp_mean"].ge(90), "bp_phenotype"] = "E_DBP_ge90"
    df.loc[ok & df["dbp_mean"].between(60, 89.999) & df["sbp_mean"].lt(140), "bp_phenotype"] = "A_SBP_lt140_DBP_60_89"
    df.loc[ok & df["dbp_mean"].lt(60) & df["sbp_mean"].lt(140), "bp_phenotype"] = "B_SBP_lt140_DBP_lt60"
    df.loc[ok & df["dbp_mean"].between(60, 89.999) & df["sbp_mean"].ge(140), "bp_phenotype"] = "C_SBP_ge140_DBP_60_89"
    df.loc[ok & df["dbp_mean"].lt(60) & df["sbp_mean"].ge(140), "bp_phenotype"] = "D_SBP_ge140_DBP_lt60"


def finish_ic(df: pd.DataFrame, base: pd.Series, prefix: str, grip_trials: list[str]) -> None:
    # Recall variables are counts of correctly recalled target words. Orientation components are
    # 0/1 correctness flags produced by comparison with the actual interview date.
    for name, source in [("immediate", f"{prefix}imrc"), ("delayed", f"{prefix}dlrc"), ("orientation", f"{prefix}orient")]:
        df[f"cog_{name}_z"] = zscore(df[source], base)
    df["cognition_z"] = df[["cog_immediate_z", "cog_delayed_z", "cog_orientation_z"]].mean(axis=1).where(
        df[["cog_immediate_z", "cog_delayed_z", "cog_orientation_z"]].notna().all(axis=1)
    )
    df["psychology_z"] = zscore(df["psychology_capacity_raw"], base)
    df["locomotion_z"] = zscore(df["locomotion_capacity_raw"], base)
    df["sensory_z"] = zscore(df["sensory_capacity_raw"], base)

    trials = df[grip_trials].apply(pd.to_numeric, errors="coerce")
    trials = trials.where(trials.gt(0) & trials.le(100))
    df["grip_max_both_hands_kg"] = trials.max(axis=1).where(trials.notna().any(axis=1))
    df["vitality_z"] = np.nan
    for sex_code in (1, 2):
        sex_ref = base & df["ragender"].eq(sex_code)
        df.loc[df["ragender"].eq(sex_code), "vitality_z"] = zscore(df["grip_max_both_hands_kg"], sex_ref)

    domains = ["cognition_z", "psychology_z", "locomotion_z", "sensory_z", "vitality_z"]
    df["ic_domains_available"] = df[domains].notna().sum(axis=1)
    df["ic_minimum_rule"] = df["ic_domains_available"].ge(4) & df["cognition_z"].notna() & df["psychology_z"].notna()
    df["ic_five_domains_complete"] = df[domains].notna().all(axis=1)
    df["ic_mean_z"] = df[domains].mean(axis=1).where(df["ic_minimum_rule"])
    df["ic_total_z"] = zscore(df["ic_mean_z"], base & df["ic_minimum_rule"])


def charls() -> pd.DataFrame:
    cols = [
        "ID", "r3agey", "ragender", "inw3", "r3hibpe", "r3rxhibp_c", "r3wtrespbiob", "r3mbmi",
        *[f"r3systo{i}" for i in (1, 2, 3)], *[f"r3diasto{i}" for i in (1, 2, 3)],
        "r3imrc", "r3dlrc", "r3mo", "r3dy", "r3yr", "r3dw", "r3orient",
        "r3depresl", "r3effortl", "r3sleeprl", "r3whappyl", "r3flonel", "r3botherl",
        "r3goingl", "r3mindtsl", "r3fhopel", "r3fearll", "r3cesd10",
        "r3walk100a", "r3chaira", "r3gripcomp", "r3gripsft", "r3gripref", "r3griptryu", "r3gripequp",
        "r3lgrip1", "r3lgrip2", "r3rgrip1", "r3rgrip2", "r3gripsum",
    ]
    df = pd.read_stata(CHARLS, convert_categoricals=False, columns=cols)
    raw = pd.read_csv(OUT / "CHARLS2015_analysis_ready.csv", usecols=["ID", "proxy", "sensory_raw"])
    df["ID"] = df["ID"].astype("string")
    raw["ID"] = raw["ID"].astype("string")
    df = df.merge(raw, on="ID", how="left", validate="one_to_one")
    bp_construct(df, "r3")
    df["self_interview"] = df["proxy"].eq(0)
    base = df["inw3"].eq(1) & df["r3agey"].ge(65) & df["self_interview"] & df["r3hibpe"].eq(1) & df["r3rxhibp_c"].eq(1) & df["bp_valid_readings"].ge(2)

    items = ["r3depresl", "r3effortl", "r3sleeprl", "r3whappyl", "r3flonel", "r3botherl", "r3goingl", "r3mindtsl", "r3fhopel", "r3fearll"]
    positive = {"r3whappyl", "r3fhopel"}
    burden = pd.DataFrame(index=df.index)
    for v in items:
        x = df[v].where(df[v].between(1, 4))
        burden[v] = (4 - x) if v in positive else (x - 1)
    df["cesd_items_valid"] = burden.notna().sum(axis=1)
    df["cesd_depressive_burden"] = burden.mean(axis=1).where(df["cesd_items_valid"].ge(8))
    df["psychology_capacity_raw"] = -df["cesd_depressive_burden"]
    df["cesd10_manual_sum"] = burden.sum(axis=1).where(df["cesd_items_valid"].eq(10))

    move = 1 - df[["r3walk100a", "r3chaira"]].where(df[["r3walk100a", "r3chaira"]].isin([0, 1]))
    df["locomotion_capacity_raw"] = move.mean(axis=1).where(move.notna().all(axis=1))
    df["sensory_capacity_raw"] = df["sensory_raw"]
    finish_ic(df, base, "r3", ["r3lgrip1", "r3lgrip2", "r3rgrip1", "r3rgrip2"])
    df["base_eligible_before_ic"] = base
    df["analysis_eligible"] = base & df["ic_minimum_rule"] & df["r3wtrespbiob"].gt(0)
    return df


def elsa() -> pd.DataFrame:
    cols = [
        "idauniq", "r8agey", "ragender", "r8proxy", "inw8", "r8hibpe", "r8rxhibp", "r8nwtresp", "r8npwtresp", "r8strat", "r8clust", "r8mbmi",
        *[f"r8systo{i}" for i in (1, 2, 3)], *[f"r8diasto{i}" for i in (1, 2, 3)],
        "r8imrc", "r8dlrc", "r8mo", "r8dy", "r8yr", "r8dw", "r8orient",
        "r8depres", "r8effort", "r8sleepr", "r8whappy", "r8flone", "r8fsad", "r8going", "r8enlife", "r8cesd",
        "r8walk100a", "r8chaira", "r8sight", "r8hearing", "r8gripcomp", "r8gripsft", "r8gripref", "r8gripequp",
        *[f"r8{hand}grip{i}" for hand in ("l", "r") for i in (1, 2, 3)], "r8gripsum",
    ]
    df, _ = pyreadstat.read_sav(ELSA, usecols=cols, apply_value_formats=False)
    bp_construct(df, "r8")
    base = df["inw8"].eq(1) & df["r8agey"].ge(65) & df["r8proxy"].eq(0) & df["r8hibpe"].eq(1) & df["r8rxhibp"].eq(1) & df["bp_valid_readings"].ge(2)

    negative = ["r8depres", "r8effort", "r8sleepr", "r8flone", "r8fsad", "r8going"]
    positive = ["r8whappy", "r8enlife"]
    parts = pd.DataFrame(index=df.index)
    for v in negative:
        parts[v] = df[v].where(df[v].isin([0, 1]))
    for v in positive:
        x = df[v].where(df[v].isin([0, 1]))
        parts[v] = 1 - x
    df["cesd_items_valid"] = parts.notna().sum(axis=1)
    df["cesd_depressive_burden"] = parts.mean(axis=1).where(df["cesd_items_valid"].ge(6))
    df["psychology_capacity_raw"] = -df["cesd_depressive_burden"]
    df["cesd8_manual_sum"] = parts.sum(axis=1).where(df["cesd_items_valid"].eq(8))

    move = 1 - df[["r8walk100a", "r8chaira"]].where(df[["r8walk100a", "r8chaira"]].isin([0, 1]))
    df["locomotion_capacity_raw"] = move.mean(axis=1).where(move.notna().all(axis=1))
    sight = (6 - df["r8sight"]).where(df["r8sight"].between(1, 6))
    hearing = (6 - df["r8hearing"]).where(df["r8hearing"].between(1, 5))
    df["sensory_capacity_raw"] = pd.concat([sight, hearing], axis=1).mean(axis=1).where(sight.notna() & hearing.notna())
    finish_ic(df, base, "r8", [f"r8{hand}grip{i}" for hand in ("l", "r") for i in (1, 2, 3)])
    df["base_eligible_before_ic"] = base
    df["analysis_eligible"] = base & df["ic_minimum_rule"] & df["r8npwtresp"].gt(0) & df["r8strat"].notna() & df["r8clust"].notna()
    return df


def audit(name: str, df: pd.DataFrame, weight: str, id_col: str) -> pd.DataFrame:
    base = df["base_eligible_before_ic"]
    eligible = df["analysis_eligible"]
    rows = [
        ("rows", len(df), "INFO"), ("unique_id", int(df[id_col].nunique()), "PASS" if df[id_col].is_unique else "FAIL"),
        ("base_eligible_before_ic", int(base.sum()), "INFO"), ("analysis_eligible", int(eligible.sum()), "INFO"),
        ("positive_weight_among_eligible", int((eligible & df[weight].gt(0)).sum()), "PASS" if df.loc[eligible, weight].gt(0).all() else "FAIL"),
        ("orientation_equals_components", int((df[f'{name}orient'].eq(df[[f'{name}mo', f'{name}dy', f'{name}yr', f'{name}dw']].sum(axis=1))).fillna(False).sum()), "INFO"),
        ("ic_five_domains_eligible", int((eligible & df["ic_five_domains_complete"]).sum()), "INFO"),
        ("bmi_available_eligible", int((eligible & df[f'{name}mbmi'].notna()).sum()), "INFO"),
        ("recent_weight_loss_variable", 0, "MISSING_IN_HARMONIZED_FILE"),
    ]
    return pd.DataFrame(rows, columns=["check", "value", "status"])


def main() -> None:
    c = charls()
    e = elsa()
    c.to_csv(OUT / "Harmonized_CHARLS2015_IC_rechecked.csv", index=False, encoding="utf-8-sig")
    e.to_csv(OUT / "Harmonized_ELSA_W8W9_analysis_ready.csv", index=False, encoding="utf-8-sig")
    ca = audit("r3", c, "r3wtrespbiob", "ID")
    ea = audit("r8", e, "r8npwtresp", "idauniq")
    ca.insert(0, "cohort", "CHARLS 2015")
    ea.insert(0, "cohort", "ELSA Wave 8/9")
    pd.concat([ca, ea], ignore_index=True).to_csv(OUT / "Harmonized_CHARLS_ELSA_IC_audit.csv", index=False, encoding="utf-8-sig")
    print(pd.concat([ca, ea], ignore_index=True).to_string(index=False))
    print("CHARLS phenotypes", c.loc[c.analysis_eligible, "bp_phenotype"].value_counts().to_dict())
    print("ELSA phenotypes", e.loc[e.analysis_eligible, "bp_phenotype"].value_counts().to_dict())


if __name__ == "__main__":
    main()
