import pandas as pd
import numpy as np
import re
from pathlib import Path

data_path = r"D:/诺丁汉/Research Method/OneDrive_2026-04-20/NHS Hospital Admissions/"

FILES = {
    "2020-2021": f"{data_path}/hosp-epis-stat-admi-diag-2020-21-tab.xlsx",
    "2021-2022": f"{data_path}/hosp-epis-stat-admi-diag-2021-22-tab.xlsx",
}

OUTPUT_DIR = Path("cw2_output")
OUTPUT_DIR.mkdir(exist_ok=True)


def to_numeric_safe(series):
    return pd.to_numeric(
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.strip()
        .replace({"-": np.nan, "nan": np.nan, "None": np.nan}),
        errors="coerce"
    )


def code_to_order(code):
    if pd.isna(code):
        return np.nan
    code = str(code).strip().upper()

    m = re.match(r"^([A-Z])(\d{2})$", code)
    if not m:
        return np.nan

    letter = m.group(1)
    number = int(m.group(2))

    # A00 -> 0*100 + 0
    return (ord(letter) - ord("A")) * 100 + number


def parse_summary_range(summary_code):
    if pd.isna(summary_code):
        return (np.nan, np.nan)

    s = str(summary_code).strip().upper()

    m = re.match(r"^([A-Z]\d{2})-([A-Z]\d{2})$", s)
    if m:
        return code_to_order(m.group(1)), code_to_order(m.group(2))

    m = re.match(r"^([A-Z]\d{2})$", s)
    if m:
        x = code_to_order(m.group(1))
        return x, x

    return (np.nan, np.nan)


def attach_summary_to_diag3(diag3_df, summary_df):
    summary_map = summary_df.copy()
    summary_map[["range_start", "range_end"]] = summary_map["summary_code"].apply(
        lambda x: pd.Series(parse_summary_range(x))
    )

    def find_summary(diag_code):
        x = code_to_order(diag_code)
        if pd.isna(x):
            return pd.Series([np.nan, np.nan])

        matched = summary_map[
            (summary_map["range_start"] <= x) &
            (summary_map["range_end"] >= x)
            ]

        if matched.empty:
            return pd.Series([np.nan, np.nan])

        row = matched.iloc[0]
        return pd.Series([row["summary_code"], row["summary_desc"]])

    diag3_df[["summary_code", "summary_desc"]] = diag3_df["diag3_code"].apply(find_summary)
    return diag3_df


def load_summary_sheet(file_path, year_label):
    """
    Primary Diagnosis Summary
      col 0 = summary code
      col 1 = summary desc
      col 3 = Admissions
      col 7 = Emergency
    """
    raw = pd.read_excel(file_path, sheet_name="Primary Diagnosis Summary", header=None)

    df = raw.iloc[12:, [0, 1, 3, 7]].copy()
    df.columns = ["summary_code", "summary_desc", "admissions", "emergency_admissions"]

    df["summary_code"] = df["summary_code"].astype(str).str.strip()
    df["summary_desc"] = df["summary_desc"].astype(str).str.strip()

    df = df[
        df["summary_code"].notna() &
        (df["summary_code"] != "") &
        (df["summary_code"].str.upper() != "TOTAL")
        ].copy()

    df["admissions"] = to_numeric_safe(df["admissions"])
    df["emergency_admissions"] = to_numeric_safe(df["emergency_admissions"])

    df["year"] = year_label

    df = df[df["admissions"].notna()].copy()

    return df


def load_diag3_sheet(file_path, year_label):
    """
    'Primary Diagnosis 3 Character'
      col 0 = diag3 code
      col 1 = diag3 desc
      col 8 = Admissions
      col 12 = Emergency
    """
    raw = pd.read_excel(file_path, sheet_name="Primary Diagnosis 3 Character", header=None)

    df = raw.iloc[13:, [0, 1, 8, 12]].copy()
    df.columns = ["diag3_code", "diag3_desc", "admissions", "emergency_admissions"]

    df["diag3_code"] = df["diag3_code"].astype(str).str.strip()
    df["diag3_desc"] = df["diag3_desc"].astype(str).str.strip()

    df = df[
        df["diag3_code"].notna() &
        (df["diag3_code"] != "") &
        (df["diag3_code"].str.upper() != "TOTAL")
        ].copy()

    df["admissions"] = to_numeric_safe(df["admissions"])
    df["emergency_admissions"] = to_numeric_safe(df["emergency_admissions"])

    df["year"] = year_label

    df = df[df["admissions"].notna()].copy()

    return df


all_summary = []
all_diag3 = []

for year_label, file_path in FILES.items():
    print(f"\nProcessing: {year_label} -> {file_path}")

    summary_df = load_summary_sheet(file_path, year_label)
    diag3_df = load_diag3_sheet(file_path, year_label)

    diag3_with_summary = attach_summary_to_diag3(diag3_df, summary_df)

    summary_df.to_csv(OUTPUT_DIR / f"summary_{year_label}.csv", index=False, encoding="utf-8-sig")
    diag3_with_summary.to_csv(OUTPUT_DIR / f"diag3_{year_label}.csv", index=False, encoding="utf-8-sig")

    all_summary.append(summary_df)
    all_diag3.append(diag3_with_summary)

summary_all = pd.concat(all_summary, ignore_index=True)
diag3_all = pd.concat(all_diag3, ignore_index=True)

final_df = (
    diag3_all
    .groupby(["summary_code", "summary_desc", "diag3_code", "diag3_desc"], as_index=False)
    .agg(
        admissions_sum=("admissions", "sum"),
        emergency_sum=("emergency_admissions", "sum")
    )
)

final_df["emergency_ratio"] = final_df["emergency_sum"] / final_df["admissions_sum"]

final_df = final_df[final_df["admissions_sum"] > 0].copy()

final_df = final_df.sort_values("admissions_sum", ascending=False)

final_path = OUTPUT_DIR / "cw2_treemap_2020_2022.csv"
final_df.to_csv(final_path, index=False, encoding="utf-8-sig")

print("\nDone.")
print(f"Final file saved to: {final_path}")
print("\nPreview:")
print(final_df.head(10))
