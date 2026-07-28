"""
Historical attrition analysis — pure Pandas, no ML.

IMPORTANT: This module performs descriptive analysis of *past* attrition
records only. It does NOT predict future attrition and does NOT use any
machine learning or statistical modeling techniques.
"""

import pandas as pd


def analyze_attrition(df: pd.DataFrame) -> dict:
    result = {}
    if "Attrition" not in df.columns:
        return result

    total = len(df)
    left_count = int((df["Attrition"] == "Yes").sum())
    result["overall_attrition_rate"] = round((left_count / total) * 100, 2) if total else 0
    result["total_left"] = left_count
    result["total_stayed"] = total - left_count

    if "Department" in df.columns:
        dept_group = df.groupby("Department")["Attrition"].apply(
            lambda s: round((s == "Yes").mean() * 100, 2)
        )
        result["department_wise_attrition"] = dept_group.sort_values(ascending=False).to_dict()

    if "Gender" in df.columns:
        gender_group = df.groupby("Gender")["Attrition"].apply(
            lambda s: round((s == "Yes").mean() * 100, 2)
        )
        result["gender_wise_attrition"] = gender_group.to_dict()

    if "Salary" in df.columns:
        df = df.copy()
        bins = [0, 30000, 60000, 100000, 1e9]
        labels = ["Low (<30k)", "Mid (30k-60k)", "High (60k-100k)", "Very High (100k+)"]
        df["_salary_band"] = pd.cut(df["Salary"], bins=bins, labels=labels)
        salary_group = df.groupby("_salary_band", observed=True)["Attrition"].apply(
            lambda s: round((s == "Yes").mean() * 100, 2)
        )
        result["salary_wise_attrition"] = salary_group.to_dict()

    if "Experience" in df.columns:
        df = df.copy()
        bins = [-1, 2, 5, 10, 100]
        labels = ["0-2 yrs", "3-5 yrs", "6-10 yrs", "10+ yrs"]
        df["_exp_band"] = pd.cut(df["Experience"], bins=bins, labels=labels)
        exp_group = df.groupby("_exp_band", observed=True)["Attrition"].apply(
            lambda s: round((s == "Yes").mean() * 100, 2)
        )
        result["experience_wise_attrition"] = exp_group.to_dict()

    return result
