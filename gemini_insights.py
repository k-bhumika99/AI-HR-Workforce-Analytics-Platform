"""
Google Gemini AI Insights module.

Sends ONLY a small statistical summary (never raw employee records) to
Gemini and asks for a structured, professional HR insights report.

Model: gemini-2.5-flash-lite
"""

import os
import json
import re
from dotenv import load_dotenv
from google import genai

load_dotenv()

MODEL_NAME = "gemini-flash-latest"

# Fixed section order + default emoji (used as fallback if Gemini omits one)
SECTION_DEFS = [
    ("key_insights", "Key HR Insights", "💡"),
    ("workforce_trends", "Workforce Trends", "📈"),
    ("potential_risks", "Potential Risks", "⚠️"),
    ("hr_recommendations", "HR Recommendations", "✅"),
    ("retention_strategies", "Retention Strategies", "🤝"),
    ("training_programs", "Training Programs", "🎓"),
    ("promotion_policies", "Promotion Policies", "🚀"),
    ("salary_improvements", "Salary Improvements", "💰"),
]

PROMPT_TEMPLATE = """You are an HR Analytics Expert. Analyze these workforce statistics and respond with ONLY valid JSON (no markdown, no code fences, no commentary) in exactly this shape:

{{
  "executive_summary": "2-3 short crisp sentences, plain text, no symbols",
  "executive_emoji": "one single emoji that captures the overall workforce health",
  "sections": [
    {{"key": "key_insights", "title": "Key HR Insights", "emoji": "one emoji", "points": ["short crisp point", "short crisp point", "short crisp point"]}},
    {{"key": "workforce_trends", "title": "Workforce Trends", "emoji": "one emoji", "points": ["...", "..."]}},
    {{"key": "potential_risks", "title": "Potential Risks", "emoji": "one emoji", "points": ["...", "..."]}},
    {{"key": "hr_recommendations", "title": "HR Recommendations", "emoji": "one emoji", "points": ["...", "..."]}},
    {{"key": "retention_strategies", "title": "Retention Strategies", "emoji": "one emoji", "points": ["...", "..."]}},
    {{"key": "training_programs", "title": "Training Programs", "emoji": "one emoji", "points": ["...", "..."]}},
    {{"key": "promotion_policies", "title": "Promotion Policies", "emoji": "one emoji", "points": ["...", "..."]}},
    {{"key": "salary_improvements", "title": "Salary Improvements", "emoji": "one emoji", "points": ["...", "..."]}}
  ]
}}

Rules:
- Each point must be ONE short crisp sentence, max 12 words, plain language, no asterisks, no hashes, no dashes, no markdown formatting of any kind.
- Give 3 to 5 points per section.
- Pick an emoji for each section that visually matches its meaning (not always the same one).
- Ground every point in the actual numbers given below — reference department names and figures where relevant.
- Return ONLY the JSON object, nothing else.

Workforce Statistics:
{stats_json}
"""


def _get_client() -> genai.Client:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Add it to your .env file "
            "(see .env.example)."
        )
    return genai.Client(api_key=api_key)


def _strip_markdown_symbols(text: str) -> str:
    """Belt-and-suspenders cleanup in case the model still slips in markdown."""
    if not text:
        return text
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)   # ### headers
    text = re.sub(r"^-{3,}\s*$", "", text, flags=re.MULTILINE)   # --- rules
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)                  # **bold**
    text = re.sub(r"\*(.*?)\*", r"\1", text)                      # *italic*
    text = re.sub(r"^[\*\-]\s+", "", text, flags=re.MULTILINE)    # * / - bullets
    text = text.replace("*", "").replace("#", "")
    return text.strip()


def _extract_json(raw: str) -> dict:
    """Pull a JSON object out of the model's response, tolerating code fences."""
    cleaned = raw.strip()
    cleaned = re.sub(r"^```(?:json)?", "", cleaned.strip())
    cleaned = re.sub(r"```$", "", cleaned.strip())
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if match:
        cleaned = match.group(0)
    return json.loads(cleaned)


def build_summary(kpis: dict, attrition_stats: dict, performance_stats: dict) -> dict:
    """Assemble the small statistics payload sent to Gemini (no raw records)."""
    summary = dict(kpis)  # total employees, avg salary, attendance, etc.

    if attrition_stats.get("department_wise_attrition"):
        dept_attr = attrition_stats["department_wise_attrition"]
        summary["department_with_highest_attrition"] = max(dept_attr, key=dept_attr.get)

    if performance_stats.get("department_wise_performance"):
        dept_perf = performance_stats["department_wise_performance"]
        summary["department_with_lowest_performance"] = min(dept_perf, key=dept_perf.get)

    return summary


def _fallback_structure(raw_text: str) -> dict:
    """
    If the model didn't return clean JSON, degrade gracefully: strip markdown
    symbols and dump everything into a single 'Key HR Insights' section so
    the UI still renders as short crisp bullet cards instead of raw text.
    """
    cleaned = _strip_markdown_symbols(raw_text)
    lines = [l.strip() for l in cleaned.split("\n") if l.strip()]
    points = lines[:8] if lines else ["No insights were returned. Please try again."]
    return {
        "executive_summary": points[0] if points else "",
        "executive_emoji": "📋",
        "sections": [
            {"key": "key_insights", "title": "Key HR Insights", "emoji": "💡", "points": points[1:9] or points},
        ],
    }


def _normalize(data: dict) -> dict:
    """Ensure every expected section exists, fill missing emojis, cap point length."""
    result = {
        "executive_summary": _strip_markdown_symbols(str(data.get("executive_summary", ""))),
        "executive_emoji": data.get("executive_emoji") or "📋",
        "sections": [],
    }

    given_sections = {s.get("key"): s for s in data.get("sections", []) if isinstance(s, dict)}

    for key, default_title, default_emoji in SECTION_DEFS:
        src = given_sections.get(key)
        if src:
            title = src.get("title") or default_title
            emoji = src.get("emoji") or default_emoji
            raw_points = src.get("points") or []
        else:
            title, emoji, raw_points = default_title, default_emoji, []

        points = [_strip_markdown_symbols(str(p)) for p in raw_points if str(p).strip()]
        if points:
            result["sections"].append({"title": title, "emoji": emoji, "points": points})

    return result


def generate_hr_insights(stats_summary: dict) -> dict:
    """
    Calls Gemini 2.5 Flash-Lite with the statistical summary and returns a
    structured dict: {executive_summary, executive_emoji, sections: [...]}
    ready to render as short, crisp, emoji-tagged cards in the UI.
    """
    try:
        client = _get_client()
        prompt = PROMPT_TEMPLATE.format(stats_json=json.dumps(stats_summary, indent=2, default=str))

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
        )
        text = getattr(response, "text", None)
        if not text:
            return {"error": "Gemini returned an empty response. Please try again."}

        try:
            data = _extract_json(text)
            return _normalize(data)
        except (json.JSONDecodeError, ValueError):
            return _fallback_structure(text)

    except RuntimeError as e:
        return {"error": f"Configuration error: {str(e)}"}
    except Exception as e:
        return {"error": f"Error generating AI insights: {str(e)}"}
