"""Department-wise comparison table with ranking — pure Pandas."""

import pandas as pd


def compare_departments(df: pd.DataFrame) -> dict:
    if "Department" not in df.columns:
        return {}

    grouped = df.groupby("Department")
    table = pd.DataFrame()

    if "Salary" in df.columns:
        table["Avg Salary"] = grouped["Salary"].mean().round(2)
    if "Attendance" in df.columns:
        table["Avg Attendance"] = grouped["Attendance"].mean().round(2)
    if "PerformanceRating" in df.columns:
        table["Avg Performance"] = grouped["PerformanceRating"].mean().round(2)
    if "Promoted" in df.columns:
        table["Promotion Rate (%)"] = grouped["Promoted"].apply(
            lambda s: round((s == "Yes").mean() * 100, 2)
        )
    if "TrainingHours" in df.columns:
        table["Avg Training Hours"] = grouped["TrainingHours"].mean().round(2)
    if "Attrition" in df.columns:
        table["Attrition Rate (%)"] = grouped["Attrition"].apply(
            lambda s: round((s == "Yes").mean() * 100, 2)
        )

    if table.empty:
        return {}

    # Overall rank: average of normalized scores (higher = better),
    # with attrition rate inverted since lower attrition is better.
    score_df = pd.DataFrame(index=table.index)
    for col in table.columns:
        if col == "Attrition Rate (%)":
            score_df[col] = 1 - (table[col] - table[col].min()) / (table[col].max() - table[col].min() + 1e-9)
        else:
            score_df[col] = (table[col] - table[col].min()) / (table[col].max() - table[col].min() + 1e-9)

    table["Overall Score"] = score_df.mean(axis=1).round(3)
    table = table.sort_values("Overall Score", ascending=False)
    table["Rank"] = range(1, len(table) + 1)

    return {
        "columns": ["Department"] + list(table.columns),
        "rows": [
            {"Department": dept, **row.to_dict()}
            for dept, row in table.iterrows()
        ],
    }
