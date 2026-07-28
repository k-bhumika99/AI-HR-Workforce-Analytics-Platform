"""Attendance analysis — pure Pandas, no ML."""

import pandas as pd


def analyze_attendance(df: pd.DataFrame) -> dict:
    result = {}
    if "Attendance" not in df.columns:
        return result

    result["average_attendance"] = round(float(df["Attendance"].mean()), 2)
    result["highest_attendance"] = round(float(df["Attendance"].max()), 2)
    result["lowest_attendance"] = round(float(df["Attendance"].min()), 2)

    if "Department" in df.columns:
        dept_attendance = df.groupby("Department")["Attendance"].mean().round(2)
        result["department_wise_attendance"] = dept_attendance.sort_values(ascending=False).to_dict()

    return result
