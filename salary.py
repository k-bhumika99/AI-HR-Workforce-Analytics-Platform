"""Salary analysis — pure Pandas, no ML."""

import pandas as pd


def analyze_salary(df: pd.DataFrame) -> dict:
    result = {}
    if "Salary" not in df.columns:
        return result

    result["average_salary"] = round(float(df["Salary"].mean()), 2)
    result["highest_salary"] = round(float(df["Salary"].max()), 2)
    result["lowest_salary"] = round(float(df["Salary"].min()), 2)
    result["median_salary"] = round(float(df["Salary"].median()), 2)

    if "Department" in df.columns:
        dept_salary = df.groupby("Department")["Salary"].mean().round(2)
        result["department_wise_salary"] = dept_salary.sort_values(ascending=False).to_dict()

    if "Gender" in df.columns:
        gender_salary = df.groupby("Gender")["Salary"].mean().round(2)
        result["gender_wise_salary"] = gender_salary.to_dict()

    if "JobRole" in df.columns:
        role_salary = df.groupby("JobRole")["Salary"].mean().round(2)
        result["role_wise_salary"] = role_salary.sort_values(ascending=False).to_dict()

    if "JoiningDate" in df.columns and df["JoiningDate"].notna().any():
        trend_df = df.dropna(subset=["JoiningDate"]).copy()
        trend_df["JoinYear"] = trend_df["JoiningDate"].dt.year
        trend = trend_df.groupby("JoinYear")["Salary"].mean().round(2)
        result["salary_trend_by_join_year"] = {str(int(k)): v for k, v in trend.sort_index().to_dict().items()}

    return result
