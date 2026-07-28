"""Performance analysis — pure Pandas, no ML."""

import pandas as pd


def analyze_performance(df: pd.DataFrame) -> dict:
    result = {}
    if "PerformanceRating" not in df.columns:
        return result

    result["average_performance"] = round(float(df["PerformanceRating"].mean()), 2)
    result["performance_distribution"] = (
        df["PerformanceRating"].value_counts().sort_index().to_dict()
    )

    if "Department" in df.columns:
        dept_perf = df.groupby("Department")["PerformanceRating"].mean().round(2)
        dept_perf = dept_perf.sort_values(ascending=False)
        result["top_performing_departments"] = dept_perf.head(5).to_dict()
        result["department_wise_performance"] = dept_perf.to_dict()

    if "Name" in df.columns:
        top_employees = df.sort_values("PerformanceRating", ascending=False).head(10)
        cols = [c for c in ["Name", "Department", "PerformanceRating"] if c in df.columns]
        result["highest_rated_employees"] = top_employees[cols].to_dict(orient="records")

    return result
