"""Promotion analysis — pure Pandas, no ML."""

import pandas as pd


def analyze_promotion(df: pd.DataFrame) -> dict:
    result = {}
    if "Promoted" not in df.columns:
        return result

    total = len(df)
    promoted_count = int((df["Promoted"] == "Yes").sum())
    result["promotion_count"] = promoted_count
    result["promotion_percentage"] = round((promoted_count / total) * 100, 2) if total else 0

    if "Department" in df.columns:
        dept_group = df.groupby("Department")["Promoted"].apply(
            lambda s: round((s == "Yes").mean() * 100, 2)
        )
        result["department_wise_promotion_rate"] = dept_group.sort_values(ascending=False).to_dict()

    return result
