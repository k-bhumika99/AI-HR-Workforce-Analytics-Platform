"""
Generates a sample HR dataset (sample_hr_data.csv) so you can test the
platform immediately without needing a real company dataset.

Run: python generate_sample_data.py
"""

import numpy as np
import pandas as pd

np.random.seed(42)

N = 500
departments = ["Sales", "Engineering", "HR", "Finance", "Marketing", "Operations", "IT"]
job_roles = ["Executive", "Senior Executive", "Manager", "Senior Manager", "Associate", "Lead"]
genders = ["Male", "Female"]

df = pd.DataFrame({
    "EmployeeID": [f"EMP{1000+i}" for i in range(N)],
    "Name": [f"Employee {i+1}" for i in range(N)],
    "Gender": np.random.choice(genders, N, p=[0.58, 0.42]),
    "Age": np.random.randint(21, 59, N),
    "Department": np.random.choice(departments, N),
    "JobRole": np.random.choice(job_roles, N),
    "Experience": np.random.randint(0, 25, N),
    "Salary": np.random.randint(25000, 180000, N),
    "Attendance": np.round(np.random.uniform(70, 100, N), 1),
    "PerformanceRating": np.random.randint(1, 6, N),
    "Promoted": np.random.choice(["Yes", "No"], N, p=[0.22, 0.78]),
    "TrainingHours": np.random.randint(0, 80, N),
    "Attrition": np.random.choice(["Yes", "No"], N, p=[0.16, 0.84]),
    "JoiningDate": pd.to_datetime("2015-01-01") + pd.to_timedelta(
        np.random.randint(0, 3800, N), unit="D"
    ),
})

# Introduce a few messy rows to demonstrate the cleaning module
df.loc[5:8, "Department"] = " sales "
df = pd.concat([df, df.iloc[[10, 20]]], ignore_index=True)  # duplicates
df.loc[15, "Salary"] = -5000  # invalid salary
df.loc[16, "Attendance"] = 150  # invalid attendance
df.loc[30:33, "PerformanceRating"] = np.nan  # missing values

df.to_csv("sample_hr_data.csv", index=False)
print(f"Sample dataset created: sample_hr_data.csv ({len(df)} rows)")
