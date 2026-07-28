"""
Builds Plotly figures for every module and returns them as JSON strings
that the frontend renders with Plotly.js (via Plotly.newPlot).

Color palette: a soft pastel set (2-3 shade families — blue, mint,
lavender) instead of solid dark/blue, so bars/slices/points are easy
to tell apart while still looking clean and calm.
"""

import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import pandas as pd

# Soft pastel palette — 3 hue families (blue, mint/teal, lavender), each
# with a light and a slightly deeper shade, so charts stay easy to read
# without looking dark or harsh.
PALETTE = [
    "#A8C8F0",  # pastel blue (light)
    "#9FE0D0",  # pastel mint (light)
    "#C9B8F0",  # pastel lavender (light)
    "#7FB0E8",  # pastel blue (mid)
    "#7DDDC5",  # pastel mint (mid)
    "#B39DDB",  # pastel lavender (mid)
]

# Kept for anything that still wants a single-tone accent
ACCENT_BLUE = "#7FB0E8"
ACCENT_VIOLET = "#B39DDB"

TEMPLATE_LAYOUT = dict(
    font=dict(family="Inter, Segoe UI, sans-serif", size=12, color="#1B2A4A"),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=56, r=28, t=70, b=64),
    height=380,
    title=dict(font=dict(size=15, color="#0B3D8C"), x=0.03, xanchor="left", y=0.95),
    legend=dict(orientation="h", yanchor="bottom", y=-0.28, xanchor="center", x=0.5, font=dict(size=11)),
    hoverlabel=dict(bgcolor="white", font_size=12, font_family="Inter, Segoe UI, sans-serif"),
    colorway=PALETTE,
)


def _finalize(fig) -> str:
    fig.update_layout(**TEMPLATE_LAYOUT)
    fig.update_xaxes(automargin=True, showgrid=False)
    fig.update_yaxes(automargin=True, showgrid=True, gridcolor="rgba(27,42,74,0.08)", zeroline=False)
    return pio.to_json(fig)


def pie_chart(data: dict, title: str) -> str:
    if not data:
        return ""
    fig = px.pie(
        names=list(data.keys()), values=list(data.values()), title=title,
        hole=0, color_discrete_sequence=PALETTE,
    )
    fig.update_traces(textinfo="percent+label", textfont_size=11, marker=dict(line=dict(color="#FFFFFF", width=2)))
    return _finalize(fig)


def donut_chart(data: dict, title: str) -> str:
    if not data:
        return ""
    fig = px.pie(
        names=list(data.keys()), values=list(data.values()), title=title,
        hole=0.55, color_discrete_sequence=PALETTE,
    )
    fig.update_traces(textinfo="percent", textfont_size=11, marker=dict(line=dict(color="#FFFFFF", width=2)))
    return _finalize(fig)


def bar_chart(data: dict, title: str, x_title="", y_title="") -> str:
    if not data:
        return ""
    categories = [str(k) for k in data.keys()]
    values = list(data.values())
    fig = px.bar(
        x=categories, y=values, title=title, text=values,
        color=categories, color_discrete_sequence=PALETTE,
    )
    fig.update_xaxes(title=x_title, tickangle=-20 if len(categories) > 6 else 0)
    fig.update_yaxes(title=y_title, rangemode="tozero")

    # headroom above the tallest bar so the outside data label never clips
    max_val = max(values) if values else 0
    if max_val > 0:
        fig.update_yaxes(range=[0, max_val * 1.22])

    fig.update_traces(
        textposition="outside",
        textfont=dict(size=11, color="#1B2A4A"),
        cliponaxis=False,
        marker_line_width=0,
    )
    fig.update_layout(showlegend=False, uniformtext_minsize=9, uniformtext_mode="hide")
    return _finalize(fig)


def line_chart(data: dict, title: str, x_title="", y_title="") -> str:
    if not data:
        return ""
    fig = px.line(
        x=list(data.keys()), y=list(data.values()), title=title, markers=True,
    )
    fig.update_traces(
        line=dict(color=ACCENT_VIOLET, width=3),
        marker=dict(size=8, color=ACCENT_BLUE, line=dict(color="#FFFFFF", width=1)),
        fill="tozeroy", fillcolor="rgba(179, 157, 219, 0.18)",
    )
    fig.update_xaxes(title=x_title)
    fig.update_yaxes(title=y_title, rangemode="tozero")
    fig.update_layout(showlegend=False)
    return _finalize(fig)


def histogram(series: pd.Series, title: str, x_title="") -> str:
    if series is None or series.empty:
        return ""
    s = series.dropna()
    unique_vals = sorted(s.unique())

    fig = px.histogram(s, title=title, color_discrete_sequence=[ACCENT_BLUE])

    if len(unique_vals) <= 15 and all(float(v).is_integer() for v in unique_vals):
        # Discrete data (e.g. ratings 2,3,4,5) — bin exactly on each value so
        # bars sit right on their tick and the chart isn't full of empty gaps.
        step = min((unique_vals[i + 1] - unique_vals[i]) for i in range(len(unique_vals) - 1)) if len(unique_vals) > 1 else 1
        fig.update_traces(xbins=dict(start=min(unique_vals) - step / 2, end=max(unique_vals) + step / 2, size=step))
    else:
        fig.update_traces(nbinsx=20)

    fig.update_traces(marker_line_width=0, opacity=0.9)
    fig.update_xaxes(title=x_title)
    fig.update_yaxes(title="Count")
    fig.update_layout(showlegend=False, bargap=0.15)
    return _finalize(fig)


def box_plot(df: pd.DataFrame, value_col: str, group_col: str, title: str) -> str:
    if df is None or value_col not in df.columns or group_col not in df.columns:
        return ""
    fig = px.box(df, x=group_col, y=value_col, title=title, color=group_col, color_discrete_sequence=PALETTE)
    fig.update_xaxes(tickangle=-20)
    fig.update_layout(showlegend=False)
    return _finalize(fig)


def scatter_plot(df: pd.DataFrame, x_col: str, y_col: str, title: str, color_col: str = None) -> str:
    if df is None or x_col not in df.columns or y_col not in df.columns:
        return ""
    fig = px.scatter(
        df, x=x_col, y=y_col, color=color_col, title=title,
        color_discrete_sequence=PALETTE,
    )
    fig.update_traces(marker=dict(size=8, opacity=0.8, line=dict(color="#FFFFFF", width=0.5)))
    return _finalize(fig)
