"""
Data cleaning module for the AI HR Workforce Analytics Platform.

Takes a raw pandas DataFrame from an uploaded CSV/Excel file and returns:
  - a cleaned DataFrame
  - a dict of cleaning statistics to show the user what was changed

No ML / prediction logic lives here — purely deterministic cleaning rules.
"""

import pandas as pd
import numpy as np
import re

# Columns we expect to find (case-insensitive, flexible matching is applied
# in _standardize_columns). Datasets that are missing some of these will
# still work — downstream analysis modules check for column existence.
EXPECTED_COLUMNS = [
    "EmployeeID", "Name", "Gender", "Age", "Department", "JobRole",
    "Experience", "Salary", "Attendance", "PerformanceRating",
    "Promoted", "TrainingHours", "Attrition", "JoiningDate",
]


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names: strip whitespace, unify casing/spacing."""
    new_cols = {}
    for col in df.columns:
        clean = re.sub(r"[\s_\-]+", "", str(col).strip()).lower()
        new_cols[col] = clean
    df = df.rename(columns=new_cols)

    # Map normalized names back to our canonical names where possible
    canonical_map = {c.lower(): c for c in EXPECTED_COLUMNS}
    rename_final = {}
    for col in df.columns:
        if col in canonical_map:
            rename_final[col] = canonical_map[col]
    df = df.rename(columns=rename_final)
    return df


def _standardize_department_names(series: pd.Series) -> pd.Series:
    """Title-case department names and collapse common variants."""
    if series is None:
        return series
    s = series.astype(str).str.strip().str.title()
    replacements = {
        "Hr": "HR",
        "It": "IT",
        "R&D": "R&D",
        "Rd": "R&D",
        "Sales & Marketing": "Sales & Marketing",
    }
    return s.replace(replacements)


def clean_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Clean an uploaded HR dataset.

    Returns:
        (cleaned_df, stats) where stats contains counts for the UI:
        rows_before, rows_after, duplicates_removed, missing_values_fixed
    """
    stats = {}
    rows_before = len(df)
    missing_before = int(df.isna().sum().sum())

    # 1. Standardize column names
    df = _standardize_columns(df)

    # 2. Trim whitespace on all string/object columns
    obj_cols = df.select_dtypes(include="object").columns
    for col in obj_cols:
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].replace({"nan": np.nan, "None": np.nan, "": np.nan})

    # 3. Remove exact duplicate rows
    duplicates_removed = int(df.duplicated().sum())
    df = df.drop_duplicates()

    # 4. Convert numeric columns
    numeric_candidates = [
        "Age", "Experience", "Salary", "Attendance",
        "PerformanceRating", "TrainingHours",
    ]
    for col in numeric_candidates:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 5. Format dates
    if "JoiningDate" in df.columns:
        df["JoiningDate"] = pd.to_datetime(df["JoiningDate"], errors="coerce")

    # 6. Standardize department names
    if "Department" in df.columns:
        df["Department"] = _standardize_department_names(df["Department"])

    # 7. Standardize Gender values
    if "Gender" in df.columns:
        df["Gender"] = df["Gender"].astype(str).str.strip().str.title()
        df["Gender"] = df["Gender"].replace(
            {"M": "Male", "F": "Female", "Nan": np.nan}
        )

    # 8. Validate attendance percentages (0-100)
    if "Attendance" in df.columns:
        invalid_attendance = ~df["Attendance"].between(0, 100)
        df.loc[invalid_attendance, "Attendance"] = np.nan

    # 9. Detect invalid salary values (negative or zero)
    if "Salary" in df.columns:
        invalid_salary = df["Salary"] <= 0
        df.loc[invalid_salary, "Salary"] = np.nan

    # 10. Standardize Promoted / Attrition to boolean-like Yes/No
    for flag_col in ("Promoted", "Attrition"):
        if flag_col in df.columns:
            df[flag_col] = df[flag_col].astype(str).str.strip().str.title()
            df[flag_col] = df[flag_col].replace(
                {"1": "Yes", "0": "No", "True": "Yes", "False": "No", "Nan": np.nan}
            )

    # 11. Handle missing values sensibly per column type
    missing_fixed = 0
    for col in df.columns:
        na_count = int(df[col].isna().sum())
        if na_count == 0:
            continue
        if col in numeric_candidates:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            missing_fixed += na_count
        elif col in ("Department", "Gender", "JobRole", "Promoted", "Attrition"):
            mode_val = df[col].mode(dropna=True)
            fill_val = mode_val.iloc[0] if not mode_val.empty else "Unknown"
            df[col] = df[col].fillna(fill_val)
            missing_fixed += na_count
        # Other columns (e.g. Name, EmployeeID, JoiningDate) are left as-is
        # since guessing values for identifiers/dates could mislead analysis.

    rows_after = len(df)

    stats = {
        "rows_before": rows_before,
        "rows_after": rows_after,
        "duplicates_removed": duplicates_removed,
        "missing_values_before": missing_before,
        "missing_values_fixed": missing_fixed,
        "columns_detected": list(df.columns),
    }

    return df, stats
