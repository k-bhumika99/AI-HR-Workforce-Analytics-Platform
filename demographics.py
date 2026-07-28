"""Employee demographic analysis — pure Pandas, no ML."""

import pandas as pd


def analyze_demographics(df: pd.DataFrame) -> dict:
    result = {}

    if "Gender" in df.columns:
        result["gender_distribution"] = df["Gender"].value_counts().to_dict()

    if "Age" in df.columns:
        bins = [17, 25, 35, 45, 55, 100]
        labels = ["18-25", "26-35", "36-45", "46-55", "56+"]
        age_groups = pd.cut(df["Age"], bins=bins, labels=labels, right=True)
        result["age_group_distribution"] = age_groups.value_counts().sort_index().to_dict()
        result["age_stats"] = {
            "min": float(df["Age"].min()),
            "max": float(df["Age"].max()),
            "mean": round(float(df["Age"].mean()), 1),
        }

    if "Department" in df.columns:
        result["department_distribution"] = df["Department"].value_counts().to_dict()

    if "JobRole" in df.columns:
        result["job_role_distribution"] = df["JobRole"].value_counts().to_dict()

    if "Experience" in df.columns:
        bins = [-1, 2, 5, 10, 15, 100]
        labels = ["0-2 yrs", "3-5 yrs", "6-10 yrs", "11-15 yrs", "15+ yrs"]
        exp_groups = pd.cut(df["Experience"], bins=bins, labels=labels)
        result["experience_distribution"] = exp_groups.value_counts().sort_index().to_dict()

    return result
