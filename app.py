"""
AI HR Workforce Analytics Platform
Flask entry point — landing page, auth, upload, per-module analytics pages,
AI insights, and export.
"""

import os
import uuid
import traceback
from datetime import datetime

import pandas as pd
from flask import (
    Flask, render_template, request, redirect, url_for,
    session, flash, send_file
)
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

from auth import create_user, verify_user, login_required
from analysis.cleaning import clean_dataset
from analysis.demographics import analyze_demographics
from analysis.attendance import analyze_attendance
from analysis.performance import analyze_performance
from analysis.salary import analyze_salary
from analysis.promotion import analyze_promotion
from analysis.training import analyze_training
from analysis.attrition import analyze_attrition
from analysis.department import compare_departments
from charts.plotly_charts import pie_chart, donut_chart, bar_chart, line_chart, box_plot, histogram, scatter_plot
from ai.gemini_insights import build_summary, generate_hr_insights

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret-change-me")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
DATA_DIR = os.path.join(BASE_DIR, "data")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
ALLOWED_EXTENSIONS = {"csv", "xlsx", "xls"}

app.config["MAX_CONTENT_LENGTH"] = int(os.getenv("MAX_UPLOAD_MB", 25)) * 1024 * 1024

for d in (UPLOAD_DIR, DATA_DIR, REPORTS_DIR):
    os.makedirs(d, exist_ok=True)


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def _dataset_path(session_id: str) -> str:
    return os.path.join(DATA_DIR, f"{session_id}.pkl")


def load_dataset():
    session_id = session.get("session_id")
    if not session_id:
        return None
    path = _dataset_path(session_id)
    if not os.path.exists(path):
        return None
    return pd.read_pickle(path)


def save_dataset(df: pd.DataFrame):
    session_id = session.get("session_id")
    if not session_id:
        session_id = str(uuid.uuid4())
        session["session_id"] = session_id
    df.to_pickle(_dataset_path(session_id))


def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Apply dashboard filters from query params: department, gender, job role, age group."""
    dept = request.args.get("department")
    gender = request.args.get("gender")
    role = request.args.get("job_role")
    age_group = request.args.get("age_group")

    filtered = df.copy()
    if dept and dept != "All" and "Department" in filtered.columns:
        filtered = filtered[filtered["Department"] == dept]
    if gender and gender != "All" and "Gender" in filtered.columns:
        filtered = filtered[filtered["Gender"] == gender]
    if role and role != "All" and "JobRole" in filtered.columns:
        filtered = filtered[filtered["JobRole"] == role]
    if age_group and age_group != "All" and "Age" in filtered.columns:
        bins = [17, 25, 35, 45, 55, 100]
        labels = ["18-25", "26-35", "36-45", "46-55", "56+"]
        groups = pd.cut(filtered["Age"], bins=bins, labels=labels)
        filtered = filtered[groups.astype(str) == age_group]

    return filtered


def get_filter_context(df: pd.DataFrame):
    """Shared filter-bar context used by every analysis page."""
    filter_options = {
        "departments": sorted(df["Department"].dropna().unique().tolist()) if "Department" in df.columns else [],
        "genders": sorted(df["Gender"].dropna().unique().tolist()) if "Gender" in df.columns else [],
        "job_roles": sorted(df["JobRole"].dropna().unique().tolist()) if "JobRole" in df.columns else [],
        "age_groups": ["18-25", "26-35", "36-45", "46-55", "56+"],
    }
    current_filters = {
        "department": request.args.get("department", "All"),
        "gender": request.args.get("gender", "All"),
        "job_role": request.args.get("job_role", "All"),
        "age_group": request.args.get("age_group", "All"),
    }
    return filter_options, current_filters


def require_dataset():
    """Loads the dataset and applies filters, or returns None if nothing uploaded yet."""
    df = load_dataset()
    if df is None:
        flash("Please upload a dataset first.", "error")
        return None
    filtered_df = apply_filters(df)
    if filtered_df.empty:
        filtered_df = df
    return df, filtered_df


# ============================================================== PUBLIC ===

@app.route("/")
def landing():
    return render_template("landing.html")


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if session.get("user"):
        return redirect(url_for("home"))

    if request.method == "POST":
        name = request.form.get("name", "")
        email = request.form.get("email", "")
        password = request.form.get("password", "")
        confirm = request.form.get("confirm_password", "")

        if password != confirm:
            flash("Passwords do not match.", "error")
            return render_template("signup.html", name=name, email=email)

        ok, error = create_user(name, email, password)
        if not ok:
            flash(error, "error")
            return render_template("signup.html", name=name, email=email)

        flash("Account created! Please sign in to continue.", "success")
        return redirect(url_for("login"))

    return render_template("signup.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if session.get("user"):
        return redirect(url_for("home"))

    if request.method == "POST":
        email = request.form.get("email", "")
        password = request.form.get("password", "")
        user = verify_user(email, password)
        if not user:
            flash("Invalid email or password.", "error")
            return render_template("signin.html", email=email)

        session["user"] = {"username": user["username"], "name": user["name"], "email": user["email"]}
        flash(f"Welcome back, {user['name'].split(' ')[0]}! 👋", "success")
        next_url = request.args.get("next")
        return redirect(next_url or url_for("home"))

    return render_template("signin.html")


@app.route("/logout")
def logout():
    session.pop("user", None)
    flash("You've been signed out. See you soon! 👋", "success")
    return redirect(url_for("landing"))


# ======================================================= AUTHENTICATED ===

@app.route("/home")
@login_required
def home():
    df = load_dataset()
    if df is None:
        return redirect(url_for("upload_page"))
    return redirect(url_for("dashboard"))


@app.route("/upload", methods=["GET", "POST"])
@login_required
def upload_page():
    if request.method == "GET":
        return render_template("upload.html")

    if "file" not in request.files:
        flash("No file part in the request.", "error")
        return redirect(url_for("upload_page"))

    file = request.files["file"]
    if file.filename == "":
        flash("No file selected.", "error")
        return redirect(url_for("upload_page"))

    if not allowed_file(file.filename):
        flash("Invalid file type. Please upload a CSV or Excel (.xlsx) file.", "error")
        return redirect(url_for("upload_page"))

    filename = secure_filename(file.filename)
    saved_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4().hex}_{filename}")
    file.save(saved_path)

    try:
        if filename.lower().endswith(".csv"):
            raw_df = pd.read_csv(saved_path)
        else:
            raw_df = pd.read_excel(saved_path)
    except Exception as e:
        flash(f"Could not read the file: {str(e)}", "error")
        return redirect(url_for("upload_page"))

    if raw_df.empty:
        flash("The uploaded file has no data.", "error")
        return redirect(url_for("upload_page"))

    try:
        cleaned_df, clean_stats = clean_dataset(raw_df)
    except Exception:
        traceback.print_exc()
        flash("An error occurred while cleaning the dataset.", "error")
        return redirect(url_for("upload_page"))

    save_dataset(cleaned_df)
    session["filename"] = filename
    session["clean_stats"] = clean_stats
    session["uploaded_at"] = datetime.now().strftime("%Y-%m-%d %H:%M")

    flash("File uploaded and cleaned successfully. 🎉", "success")
    return redirect(url_for("dashboard"))


@app.route("/dashboard")
@login_required
def dashboard():
    result = require_dataset()
    if result is None:
        return redirect(url_for("upload_page"))
    df, filtered_df = result

    demo = analyze_demographics(filtered_df)
    attendance = analyze_attendance(filtered_df)
    performance = analyze_performance(filtered_df)
    salary = analyze_salary(filtered_df)
    promotion = analyze_promotion(filtered_df)
    training = analyze_training(filtered_df)
    attrition = analyze_attrition(filtered_df)
    dept_comparison = compare_departments(filtered_df)

    kpis = {
        "total_employees": int(len(filtered_df)),
        "average_salary": salary.get("average_salary", 0),
        "average_attendance": attendance.get("average_attendance", 0),
        "average_performance": performance.get("average_performance", 0),
        "promotion_rate": promotion.get("promotion_percentage", 0),
        "attrition_rate": attrition.get("overall_attrition_rate", 0),
        "total_departments": filtered_df["Department"].nunique() if "Department" in filtered_df.columns else 0,
        "total_training_hours": round(float(filtered_df["TrainingHours"].sum()), 1) if "TrainingHours" in filtered_df.columns else 0,
    }

    charts = {
        "gender_pie": pie_chart(demo.get("gender_distribution", {}), "Gender Distribution"),
        "dept_bar": bar_chart(demo.get("department_distribution", {}), "Employees by Department"),
    }

    filter_options, current_filters = get_filter_context(df)

    return render_template(
        "overview.html",
        filename=session.get("filename", "dataset"),
        clean_stats=session.get("clean_stats", {}),
        kpis=kpis,
        dept_comparison=dept_comparison,
        charts=charts,
        filter_options=filter_options,
        current_filters=current_filters,
    )


@app.route("/demographics")
@login_required
def demographics_page():
    result = require_dataset()
    if result is None:
        return redirect(url_for("upload_page"))
    df, filtered_df = result

    demo = analyze_demographics(filtered_df)
    charts = {
        "gender_pie": pie_chart(demo.get("gender_distribution", {}), "Gender Distribution"),
        "dept_bar": bar_chart(demo.get("department_distribution", {}), "Employees by Department"),
        "age_hist": histogram(filtered_df["Age"], "Age Distribution", "Age") if "Age" in filtered_df.columns else "",
        "age_group_bar": line_chart(
            {str(k): v for k, v in demo.get("age_group_distribution", {}).items()},
            "Employees by Age Group", "Age Group", "Employees"
        ),
        "role_bar": bar_chart(demo.get("job_role_distribution", {}), "Employees by Job Role", "Job Role", "Employees"),
        "experience_bar": donut_chart(
            {str(k): v for k, v in demo.get("experience_distribution", {}).items()},
            "Employees by Experience Band"
        ),
    }
    filter_options, current_filters = get_filter_context(df)

    return render_template(
        "demographics.html",
        demo=demo,
        total=len(filtered_df),
        charts=charts,
        filter_options=filter_options,
        current_filters=current_filters,
    )


@app.route("/attendance")
@login_required
def attendance_page():
    result = require_dataset()
    if result is None:
        return redirect(url_for("upload_page"))
    df, filtered_df = result

    attendance = analyze_attendance(filtered_df)
    charts = {
        "attendance_bar": bar_chart(attendance.get("department_wise_attendance", {}), "Attendance by Department", "Department", "Avg Attendance %"),
        "attendance_hist": histogram(filtered_df["Attendance"], "Attendance Distribution", "Attendance %") if "Attendance" in filtered_df.columns else "",
    }
    filter_options, current_filters = get_filter_context(df)

    return render_template(
        "attendance.html",
        attendance=attendance,
        charts=charts,
        filter_options=filter_options,
        current_filters=current_filters,
    )


@app.route("/performance")
@login_required
def performance_page():
    result = require_dataset()
    if result is None:
        return redirect(url_for("upload_page"))
    df, filtered_df = result

    performance = analyze_performance(filtered_df)
    charts = {
        "performance_hist": histogram(filtered_df["PerformanceRating"], "Performance Distribution", "Rating") if "PerformanceRating" in filtered_df.columns else "",
        "top_dept_bar": bar_chart(performance.get("top_performing_departments", {}), "Top Performing Departments", "Department", "Avg Rating"),
    }
    filter_options, current_filters = get_filter_context(df)

    return render_template(
        "performance.html",
        performance=performance,
        charts=charts,
        filter_options=filter_options,
        current_filters=current_filters,
    )


@app.route("/salary")
@login_required
def salary_page():
    result = require_dataset()
    if result is None:
        return redirect(url_for("upload_page"))
    df, filtered_df = result

    salary = analyze_salary(filtered_df)
    charts = {
        "salary_box": box_plot(filtered_df, "Salary", "Department", "Salary Distribution by Department"),
        "salary_bar": bar_chart(salary.get("department_wise_salary", {}), "Average Salary by Department", "Department", "Avg Salary"),
        "gender_salary_bar": donut_chart(salary.get("gender_wise_salary", {}), "Average Salary by Gender"),
        "role_salary_bar": bar_chart(salary.get("role_wise_salary", {}), "Average Salary by Job Role", "Job Role", "Avg Salary"),
        "salary_trend_line": line_chart(salary.get("salary_trend_by_join_year", {}), "Average Salary Trend by Joining Year", "Joining Year", "Avg Salary"),
    }
    filter_options, current_filters = get_filter_context(df)

    return render_template(
        "salary.html",
        salary=salary,
        charts=charts,
        filter_options=filter_options,
        current_filters=current_filters,
    )


@app.route("/promotion")
@login_required
def promotion_page():
    result = require_dataset()
    if result is None:
        return redirect(url_for("upload_page"))
    df, filtered_df = result

    promotion = analyze_promotion(filtered_df)
    charts = {
        "promotion_pie": pie_chart(
            {"Promoted": promotion.get("promotion_count", 0),
             "Not Promoted": int(len(filtered_df)) - promotion.get("promotion_count", 0)},
            "Promotion Split"
        ),
        "promotion_dept_bar": bar_chart(promotion.get("department_wise_promotion_rate", {}), "Promotion Rate by Department", "Department", "Promotion %"),
    }
    filter_options, current_filters = get_filter_context(df)

    return render_template(
        "promotion.html",
        promotion=promotion,
        total=len(filtered_df),
        charts=charts,
        filter_options=filter_options,
        current_filters=current_filters,
    )


@app.route("/training")
@login_required
def training_page():
    result = require_dataset()
    if result is None:
        return redirect(url_for("upload_page"))
    df, filtered_df = result

    training = analyze_training(filtered_df)
    charts = {
        "training_bar": bar_chart(training.get("department_wise_training", {}), "Avg Training Hours by Department", "Department", "Hours"),
        "training_scatter": scatter_plot(filtered_df, "Experience", "TrainingHours", "Experience vs Training Hours", color_col="Department" if "Department" in filtered_df.columns else None),
    }
    filter_options, current_filters = get_filter_context(df)

    return render_template(
        "training.html",
        training=training,
        charts=charts,
        filter_options=filter_options,
        current_filters=current_filters,
    )


@app.route("/attrition")
@login_required
def attrition_page():
    result = require_dataset()
    if result is None:
        return redirect(url_for("upload_page"))
    df, filtered_df = result

    attrition = analyze_attrition(filtered_df)
    charts = {
        "attrition_pie": pie_chart(
            {"Left": attrition.get("total_left", 0), "Stayed": attrition.get("total_stayed", 0)},
            "Attrition Split"
        ),
        "attrition_dept_bar": bar_chart(attrition.get("department_wise_attrition", {}), "Attrition Rate by Department", "Department", "Attrition %"),
        "attrition_gender_bar": donut_chart(attrition.get("gender_wise_attrition", {}), "Attrition Rate by Gender"),
        "attrition_salary_bar": bar_chart(attrition.get("salary_wise_attrition", {}), "Attrition Rate by Salary Band", "Salary Band", "Attrition %"),
        "attrition_exp_bar": line_chart(attrition.get("experience_wise_attrition", {}), "Attrition Rate by Experience Band", "Experience Band", "Attrition %"),
    }
    filter_options, current_filters = get_filter_context(df)

    return render_template(
        "attrition.html",
        attrition=attrition,
        charts=charts,
        filter_options=filter_options,
        current_filters=current_filters,
    )


@app.route("/departments")
@login_required
def departments_page():
    result = require_dataset()
    if result is None:
        return redirect(url_for("upload_page"))
    df, filtered_df = result

    dept_comparison = compare_departments(filtered_df)
    filter_options, current_filters = get_filter_context(df)

    return render_template(
        "departments.html",
        dept_comparison=dept_comparison,
        filter_options=filter_options,
        current_filters=current_filters,
    )


@app.route("/ai-insights", methods=["GET", "POST"])
@login_required
def ai_insights():
    df = load_dataset()
    if df is None:
        flash("Please upload a dataset first.", "error")
        return redirect(url_for("upload_page"))

    filtered_df = apply_filters(df)
    if filtered_df.empty:
        filtered_df = df

    salary = analyze_salary(filtered_df)
    attendance = analyze_attendance(filtered_df)
    performance = analyze_performance(filtered_df)
    promotion = analyze_promotion(filtered_df)
    training = analyze_training(filtered_df)
    attrition = analyze_attrition(filtered_df)

    kpis = {
        "total_employees": int(len(filtered_df)),
        "average_salary": salary.get("average_salary", 0),
        "attendance_percentage": attendance.get("average_attendance", 0),
        "performance_score": performance.get("average_performance", 0),
        "promotion_rate": promotion.get("promotion_percentage", 0),
        "training_hours": training.get("average_training_hours", 0),
    }

    summary = build_summary(kpis, attrition, performance)
    insights = None

    if request.method == "POST":
        insights = generate_hr_insights(summary)
        session["last_insights"] = insights
        session["last_summary"] = summary

    return render_template(
        "ai_insights.html",
        summary=summary,
        insights=insights or session.get("last_insights"),
    )


@app.route("/export")
@login_required
def export_page():
    df = load_dataset()
    if df is None:
        flash("Please upload a dataset first.", "error")
        return redirect(url_for("upload_page"))

    filtered_df = apply_filters(df)
    if filtered_df.empty:
        filtered_df = df

    has_insights = bool(session.get("last_insights") and not session["last_insights"].get("error"))

    return render_template(
        "export.html",
        row_count=len(filtered_df),
        col_count=len(filtered_df.columns),
        has_insights=has_insights,
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M"),
    )


@app.route("/export/csv")
@login_required
def export_csv():
    df = load_dataset()
    if df is None:
        flash("Please upload a dataset first.", "error")
        return redirect(url_for("upload_page"))

    filtered_df = apply_filters(df)
    export_path = os.path.join(REPORTS_DIR, f"hr_data_export_{uuid.uuid4().hex[:8]}.csv")
    filtered_df.to_csv(export_path, index=False)
    return send_file(export_path, as_attachment=True, download_name="hr_workforce_export.csv")


@app.route("/export/pdf")
@login_required
def export_pdf():
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib import colors

    df = load_dataset()
    if df is None:
        flash("Please upload a dataset first.", "error")
        return redirect(url_for("upload_page"))

    filtered_df = apply_filters(df)
    if filtered_df.empty:
        filtered_df = df

    salary = analyze_salary(filtered_df)
    attendance = analyze_attendance(filtered_df)
    performance = analyze_performance(filtered_df)
    promotion = analyze_promotion(filtered_df)
    attrition = analyze_attrition(filtered_df)
    training = analyze_training(filtered_df)

    export_path = os.path.join(REPORTS_DIR, f"hr_report_{uuid.uuid4().hex[:8]}.pdf")
    doc = SimpleDocTemplate(export_path, pagesize=A4)
    styles = getSampleStyleSheet()
    story = [
        Paragraph("AI HR Workforce Analytics — Summary Report", styles["Title"]),
        Spacer(1, 12),
        Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles["Normal"]),
        Spacer(1, 20),
        Paragraph("Key Performance Indicators", styles["Heading2"]),
    ]

    kpi_rows = [
        ["Metric", "Value"],
        ["Total Employees", str(len(filtered_df))],
        ["Average Salary", str(salary.get("average_salary", "N/A"))],
        ["Average Attendance (%)", str(attendance.get("average_attendance", "N/A"))],
        ["Average Performance", str(performance.get("average_performance", "N/A"))],
        ["Promotion Rate (%)", str(promotion.get("promotion_percentage", "N/A"))],
        ["Attrition Rate (%)", str(attrition.get("overall_attrition_rate", "N/A"))],
        ["Average Training Hours", str(training.get("average_training_hours", "N/A"))],
    ]
    table = Table(kpi_rows, colWidths=[250, 200])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1E5FBF")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(table)
    story.append(Spacer(1, 20))

    insights = session.get("last_insights")
    if insights and not insights.get("error"):
        story.append(Paragraph("AI HR Insights", styles["Heading2"]))
        if insights.get("executive_summary"):
            story.append(Paragraph(
                f"{insights.get('executive_emoji', '')} {insights['executive_summary']}",
                styles["Normal"]
            ))
            story.append(Spacer(1, 10))
        for section in insights.get("sections", []):
            story.append(Paragraph(
                f"{section.get('emoji', '')} {section.get('title', '')}",
                styles["Heading3"]
            ))
            for point in section.get("points", []):
                story.append(Paragraph(f"• {point}", styles["Normal"]))
            story.append(Spacer(1, 10))

    doc.build(story)
    return send_file(export_path, as_attachment=True, download_name="hr_workforce_report.pdf")


@app.route("/reset")
@login_required
def reset():
    session.pop("session_id", None)
    session.pop("filename", None)
    session.pop("clean_stats", None)
    session.pop("uploaded_at", None)
    session.pop("last_insights", None)
    session.pop("last_summary", None)
    flash("Session cleared. Upload a new dataset to begin.", "success")
    return redirect(url_for("upload_page"))


@app.errorhandler(413)
def file_too_large(e):
    flash("File is too large. Please upload a smaller file.", "error")
    return redirect(url_for("upload_page"))


if __name__ == "__main__":
    app.run(debug=True, port=5000)
