"""Training analysis — pure Pandas, no ML."""

import pandas as pd


def analyze_training(df: pd.DataFrame) -> dict:
    result = {}
    if "TrainingHours" not in df.columns:
        return result

    result["average_training_hours"] = round(float(df["TrainingHours"].mean()), 2)

    if "Department" in df.columns:
        dept_training = df.groupby("Department")["TrainingHours"].mean().round(2)
        result["department_wise_training"] = dept_training.sort_values(ascending=False).to_dict()

    if "Name" in df.columns:
        top_trained = df.sort_values("TrainingHours", ascending=False).head(10)
        cols = [c for c in ["Name", "Department", "TrainingHours"] if c in df.columns]
        result["employees_highest_training"] = top_trained[cols].to_dict(orient="records")

    return result
