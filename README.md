# AI HR Workforce Analytics Platform

A Flask-based HR analytics platform that cleans uploaded employee datasets,
runs statistical analysis with Pandas, renders interactive Plotly dashboards,
and generates AI HR insights using **Google Gemini 2.5 Flash-Lite**.

No machine learning or predictive modeling is used anywhere — all analysis
is descriptive statistics on historical data.

## 1. Setup

```bash
cd AI_HR_Workforce_Analytics
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Configure your Gemini API key

```bash
cp .env.example .env
```

Edit `.env` and add your key (get one free at https://aistudio.google.com/apikey):

```
GEMINI_API_KEY=your_real_key_here
FLASK_SECRET_KEY=some_random_string
```

## 3. (Optional) Generate a sample dataset to test with

```bash
python generate_sample_data.py
```

This creates `sample_hr_data.csv` with 500 employees, plus some intentionally
messy rows (duplicates, invalid salary, invalid attendance, missing ratings)
so you can see the cleaning module in action.

## 4. Run the app

```bash
python app.py
```

Open **http://127.0.0.1:5000** in your browser.

## 5. Using the platform

1. **Landing page** — a marketing homepage explaining the product and its features.
2. **Sign up / Sign in** — create a free account (stored locally in `data/users.json`
   with hashed passwords) before you can upload data or view dashboards.
3. **Upload** — drag and drop `sample_hr_data.csv` (or your own dataset).
4. **Analytics pages** — each HR domain has its own dedicated page in the sidebar:
   Overview, Demographics, Attendance, Performance, Salary, Promotion, Training,
   Attrition, and Departments. Use the filter bar (Department / Gender / Job Role /
   Age Group) on any page to slice the data live.
5. **AI HR Insights** — click "Generate AI HR Insights" to send a small
   statistics summary (not raw employee records) to Gemini and get a
   structured executive report.
6. **Export** — download the filtered dataset as CSV, or a summary report
   (KPIs + AI insights) as PDF, from the sidebar.

## Expected dataset columns

The cleaning module is flexible — it works even if some columns are
missing — but works best with:

```
EmployeeID, Name, Gender, Age, Department, JobRole, Experience,
Salary, Attendance, PerformanceRating, Promoted, TrainingHours,
Attrition, JoiningDate
```

## Project structure

```
AI_HR_Workforce_Analytics/
├── app.py                   # Flask routes (public pages, auth, per-module analytics)
├── auth.py                  # JSON-file user store + password hashing + login_required
├── generate_sample_data.py  # Test data generator
├── requirements.txt
├── .env.example
├── uploads/                 # raw uploaded files (gitignored)
├── data/                    # cleaned data cache + users.json (gitignored)
├── reports/                 # generated PDF/CSV exports (gitignored)
├── analysis/                # Pandas-only analysis modules (one per HR domain)
├── ai/gemini_insights.py    # Gemini 2.5 Flash-Lite integration
├── charts/plotly_charts.py  # Plotly figure builders
├── templates/
│   ├── landing.html         # Marketing homepage
│   ├── signin.html / signup.html
│   ├── public_base.html     # Shell for landing/auth pages
│   ├── base.html            # Shell for the authenticated app (sidebar + topbar)
│   ├── upload.html
│   ├── overview.html        # Dashboard home (KPIs + jump-to cards)
│   ├── demographics.html, attendance.html, performance.html, salary.html,
│   │   promotion.html, training.html, attrition.html, departments.html
│   ├── ai_insights.html, export.html
│   └── partials/filter_bar.html
└── static/css, static/js     # Blue/white styling + interactivity
```

## Accounts &amp; authentication

Sign-up/sign-in uses a simple local JSON file (`data/users.json`) with
Werkzeug-hashed passwords — no external identity provider required. This is
meant for local/demo use; swap in a real database and session store before
deploying multi-user in production.

## Notes

- Cleaned data is cached per-session as a pickle file in `data/` — no
  database is required for this version. For multi-user production
  deployment, swap this for a proper session-scoped store (e.g. Redis).
- The Gemini call only ever receives a small JSON summary (totals, averages,
  rates) — never raw employee rows — keeping token cost low and avoiding
  unnecessary exposure of personal data.
