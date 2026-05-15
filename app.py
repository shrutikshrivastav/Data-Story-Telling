import os
import json
import warnings
import traceback
from io import BytesIO

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from flask import Flask, render_template, request, jsonify
from scipy import stats

warnings.filterwarnings("ignore")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50MB

UPLOAD_FOLDER = "/tmp/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

PLOTLY_THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="'DM Sans', sans-serif", color="#e2e8f0"),
    colorway=["#6366f1", "#22d3ee", "#f472b6", "#34d399", "#fb923c", "#a78bfa", "#fbbf24"],
    margin=dict(l=40, r=40, t=50, b=40),
)

# ─────────────────────────────────────────────
#  PHASE 1 — DATA CLEANING
# ─────────────────────────────────────────────

def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"[^\w]+", "_", regex=True)
        .str.strip("_")
    )
    return df


def detect_date_columns(df: pd.DataFrame) -> list[str]:
    date_cols = []
    for col in df.select_dtypes(include="object").columns:
        sample = df[col].dropna().head(100)
        converted = pd.to_datetime(sample, infer_datetime_format=True, errors="coerce")
        if converted.notna().mean() > 0.7:
            date_cols.append(col)
    return date_cols


def fix_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.select_dtypes(include="object").columns:
        # try numeric
        converted = pd.to_numeric(df[col].str.replace(r"[,$%]", "", regex=True), errors="coerce")
        if converted.notna().sum() / max(len(df), 1) > 0.7:
            df[col] = converted
            continue
        # try bool
        lower = df[col].str.lower().str.strip()
        bool_map = {"true": True, "false": False, "yes": True, "no": False, "1": True, "0": False}
        if lower.isin(bool_map).mean() > 0.9:
            df[col] = lower.map(bool_map)
    return df


def detect_outliers(df: pd.DataFrame) -> dict:
    outlier_info = {}
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        series = df[col].dropna()
        if len(series) < 10:
            continue
        z = np.abs(stats.zscore(series))
        n_out = int((z > 3).sum())
        if n_out > 0:
            outlier_info[col] = n_out
    return outlier_info


def compute_quality_score(
    total_rows, dup_removed, missing_fixed, outliers_total, total_cells
) -> int:
    if total_cells == 0:
        return 100
    dup_penalty = min(dup_removed / max(total_rows, 1) * 40, 20)
    missing_penalty = min(missing_fixed / max(total_cells, 1) * 100, 30)
    outlier_penalty = min(outliers_total / max(total_cells, 1) * 100, 10)
    score = max(int(100 - dup_penalty - missing_penalty - outlier_penalty), 0)
    return score


def clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    before_rows, before_cols = df.shape

    # 1. Remove completely empty rows/cols
    df.dropna(how="all", inplace=True)
    df.dropna(axis=1, how="all", inplace=True)

    # 2. Normalize column names
    df = normalize_column_names(df)

    # 3. Remove duplicates
    before_dup = len(df)
    df.drop_duplicates(inplace=True)
    dup_removed = before_dup - len(df)

    # 4. Fix dtypes
    df = fix_dtypes(df)

    # 5. Detect & convert date columns
    date_cols = detect_date_columns(df)
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], infer_datetime_format=True, errors="coerce")

    # 6. Handle missing values
    missing_before = int(df.isnull().sum().sum())
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isnull().any():
            df[col].fillna(df[col].median(), inplace=True)
    for col in df.select_dtypes(include=["object", "category"]).columns:
        if df[col].isnull().any():
            mode = df[col].mode()
            df[col].fillna(mode[0] if len(mode) else "Unknown", inplace=True)
    for col in df.select_dtypes(include=["datetime64"]).columns:
        if df[col].isnull().any():
            df[col].fillna(method="ffill", inplace=True)
    missing_after = int(df.isnull().sum().sum())
    missing_fixed = missing_before - missing_after

    # 7. Outlier detection (report only, don't remove)
    outlier_info = detect_outliers(df)
    outliers_total = sum(outlier_info.values())

    after_rows, after_cols = df.shape
    total_cells = after_rows * after_cols
    quality_score = compute_quality_score(
        before_rows, dup_removed, missing_fixed, outliers_total, total_cells
    )

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    datetime_cols = df.select_dtypes(include=["datetime64"]).columns.tolist()

    summary = {
        "before_rows": before_rows,
        "before_cols": before_cols,
        "after_rows": after_rows,
        "after_cols": after_cols,
        "duplicates_removed": dup_removed,
        "missing_fixed": missing_fixed,
        "outliers_detected": outliers_total,
        "outlier_columns": outlier_info,
        "quality_score": quality_score,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "datetime_cols": datetime_cols,
        "date_cols_detected": date_cols,
        "columns": df.columns.tolist(),
    }
    return df, summary


# ─────────────────────────────────────────────
#  PHASE 2 — VISUALIZATIONS
# ─────────────────────────────────────────────

def _apply_theme(fig):
    fig.update_layout(**PLOTLY_THEME)
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.1)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.1)")
    return fig


def _fig_json(fig):
    return json.loads(fig.to_json())


def generate_visualizations(df: pd.DataFrame, summary: dict) -> list[dict]:
    charts = []
    numeric_cols = summary["numeric_cols"]
    categorical_cols = summary["categorical_cols"]
    datetime_cols = summary["datetime_cols"]
    colors = PLOTLY_THEME["colorway"]

    # ── KPI cards data ──────────────────────────────
    kpis = []
    for i, col in enumerate(numeric_cols[:6]):
        s = df[col].dropna()
        kpis.append({
            "label": col.replace("_", " ").title(),
            "value": _smart_format(s.sum()),
            "mean": _smart_format(s.mean()),
            "trend": float(round(((s.iloc[-1] - s.iloc[0]) / max(abs(s.iloc[0]), 1)) * 100, 1)) if len(s) > 1 else 0,
            "color": colors[i % len(colors)],
        })
    if kpis:
        charts.append({"type": "kpi", "data": kpis, "title": "Key Performance Indicators"})

    # ── 1. Distribution histograms ───────────────────
    if numeric_cols:
        n = min(len(numeric_cols), 4)
        cols_to_plot = numeric_cols[:n]
        rows = (n + 1) // 2
        fig = make_subplots(rows=rows, cols=2, subplot_titles=[c.replace("_", " ").title() for c in cols_to_plot])
        for idx, col in enumerate(cols_to_plot):
            r, c = divmod(idx, 2)
            fig.add_trace(
                go.Histogram(
                    x=df[col], name=col, marker_color=colors[idx % len(colors)],
                    opacity=0.85, showlegend=False,
                    hovertemplate=f"<b>{col}</b><br>Range: %{{x}}<br>Count: %{{y}}<extra></extra>",
                ),
                row=r + 1, col=c + 1,
            )
        fig.update_layout(title_text="📊 Numeric Distributions", **PLOTLY_THEME)
        _apply_theme(fig)
        charts.append({"type": "chart", "figure": _fig_json(fig), "title": "Numeric Distributions"})

    # ── 2. Category bar charts ───────────────────────
    for i, col in enumerate(categorical_cols[:3]):
        vc = df[col].value_counts().head(15)
        if len(vc) < 2:
            continue
        fig = go.Figure(
            go.Bar(
                x=vc.index.astype(str),
                y=vc.values,
                marker=dict(
                    color=vc.values,
                    colorscale=[[0, colors[i % len(colors)] + "66"], [1, colors[i % len(colors)]]],
                    showscale=False,
                    line=dict(color="rgba(255,255,255,0.1)", width=1),
                ),
                hovertemplate="<b>%{x}</b><br>Count: %{y:,}<extra></extra>",
            )
        )
        fig.update_layout(title_text=f"📦 {col.replace('_',' ').title()} — Distribution", **PLOTLY_THEME)
        _apply_theme(fig)
        charts.append({"type": "chart", "figure": _fig_json(fig), "title": f"{col.replace('_',' ').title()} Distribution"})

    # ── 3. Pie chart for top category ───────────────
    if categorical_cols:
        col = categorical_cols[0]
        vc = df[col].value_counts().head(8)
        if len(vc) >= 2:
            fig = go.Figure(
                go.Pie(
                    labels=vc.index.astype(str),
                    values=vc.values,
                    hole=0.5,
                    marker=dict(colors=colors, line=dict(color="#0f172a", width=2)),
                    hovertemplate="<b>%{label}</b><br>%{value:,} (%{percent})<extra></extra>",
                    textfont=dict(size=12),
                )
            )
            fig.update_layout(title_text=f"🥧 {col.replace('_',' ').title()} — Share", **PLOTLY_THEME)
            charts.append({"type": "chart", "figure": _fig_json(fig), "title": f"{col.replace('_',' ').title()} Share"})

    # ── 4. Time-series line chart ────────────────────
    if datetime_cols and numeric_cols:
        dcol = datetime_cols[0]
        ncol = numeric_cols[0]
        ts = df[[dcol, ncol]].dropna().sort_values(dcol)
        if len(ts) > 1:
            # resample to at most 200 points
            if len(ts) > 200:
                freq = max(len(ts) // 200, 1)
                ts = ts.iloc[::freq]
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=ts[dcol], y=ts[ncol], mode="lines",
                    line=dict(color=colors[0], width=2.5, shape="spline"),
                    fill="tozeroy", fillcolor=colors[0].replace(")", ",0.1)").replace("rgb", "rgba") if "rgb" in colors[0] else colors[0] + "1a",
                    hovertemplate=f"<b>%{{x|%b %d, %Y}}</b><br>{ncol}: %{{y:,.2f}}<extra></extra>",
                    name=ncol,
                )
            )
            # rolling average
            window = max(len(ts) // 10, 2)
            ts_roll = ts[ncol].rolling(window=window, min_periods=1).mean()
            fig.add_trace(
                go.Scatter(
                    x=ts[dcol], y=ts_roll, mode="lines",
                    line=dict(color=colors[1], width=1.5, dash="dot"),
                    name=f"{window}-pt avg", opacity=0.7,
                    hovertemplate=f"Avg: %{{y:,.2f}}<extra></extra>",
                )
            )
            fig.update_layout(title_text=f"📈 {ncol.replace('_',' ').title()} Over Time", **PLOTLY_THEME)
            _apply_theme(fig)
            charts.append({"type": "chart", "figure": _fig_json(fig), "title": f"{ncol.replace('_',' ').title()} Trend"})

    # ── 5. Scatter plot (top 2 numeric) ─────────────
    if len(numeric_cols) >= 2:
        x_col, y_col = numeric_cols[0], numeric_cols[1]
        color_col = categorical_cols[0] if categorical_cols else None
        sample = df.sample(min(500, len(df)), random_state=42)
        if color_col:
            top_cats = sample[color_col].value_counts().head(8).index
            sample = sample[sample[color_col].isin(top_cats)]
            fig = px.scatter(
                sample, x=x_col, y=y_col, color=color_col,
                color_discrete_sequence=colors,
                hover_data=sample.columns[:4].tolist(),
                opacity=0.75,
            )
        else:
            fig = px.scatter(sample, x=x_col, y=y_col, color_discrete_sequence=[colors[2]], opacity=0.75)
        fig.update_traces(marker=dict(size=7, line=dict(width=0.5, color="rgba(255,255,255,0.3)")))
        fig.update_layout(title_text=f"🔵 {x_col.replace('_',' ').title()} vs {y_col.replace('_',' ').title()}", **PLOTLY_THEME)
        _apply_theme(fig)
        charts.append({"type": "chart", "figure": _fig_json(fig), "title": f"Scatter: {x_col.title()} vs {y_col.title()}"})

    # ── 6. Correlation heatmap ───────────────────────
    if len(numeric_cols) >= 3:
        corr_cols = numeric_cols[:12]
        corr = df[corr_cols].corr()
        labels = [c.replace("_", " ").title() for c in corr_cols]
        fig = go.Figure(
            go.Heatmap(
                z=corr.values,
                x=labels, y=labels,
                colorscale=[
                    [0, "#dc2626"], [0.25, "#f97316"], [0.5, "#0f172a"],
                    [0.75, "#6366f1"], [1, "#22d3ee"],
                ],
                zmid=0, zmin=-1, zmax=1,
                hovertemplate="<b>%{x}</b> × <b>%{y}</b><br>r = %{z:.3f}<extra></extra>",
                text=np.round(corr.values, 2),
                texttemplate="%{text}",
                textfont=dict(size=10),
            )
        )
        fig.update_layout(title_text="🔥 Correlation Heatmap", **PLOTLY_THEME)
        charts.append({"type": "chart", "figure": _fig_json(fig), "title": "Correlation Heatmap"})

    # ── 7. Category × Numeric grouped bar ───────────
    if categorical_cols and len(numeric_cols) >= 2:
        cat_col = categorical_cols[0]
        num1, num2 = numeric_cols[0], numeric_cols[1]
        top_cats = df[cat_col].value_counts().head(10).index
        grouped = df[df[cat_col].isin(top_cats)].groupby(cat_col)[[num1, num2]].mean().reset_index()
        fig = go.Figure()
        fig.add_trace(go.Bar(x=grouped[cat_col].astype(str), y=grouped[num1], name=num1.replace("_"," ").title(), marker_color=colors[0], opacity=0.9))
        fig.add_trace(go.Bar(x=grouped[cat_col].astype(str), y=grouped[num2], name=num2.replace("_"," ").title(), marker_color=colors[1], opacity=0.9))
        fig.update_layout(barmode="group", title_text=f"📊 {cat_col.replace('_',' ').title()} — Comparative Analysis", **PLOTLY_THEME)
        _apply_theme(fig)
        charts.append({"type": "chart", "figure": _fig_json(fig), "title": "Comparative Analysis"})

    return charts


# ─────────────────────────────────────────────
#  PHASE 3 — NARRATIVE
# ─────────────────────────────────────────────

def _smart_format(val) -> str:
    if pd.isna(val):
        return "N/A"
    if abs(val) >= 1_000_000_000:
        return f"{val/1_000_000_000:.2f}B"
    if abs(val) >= 1_000_000:
        return f"{val/1_000_000:.2f}M"
    if abs(val) >= 1_000:
        return f"{val/1_000:.1f}K"
    if isinstance(val, float):
        return f"{val:.2f}"
    return str(val)


def generate_narrative(df: pd.DataFrame, summary: dict) -> dict:
    numeric_cols = summary["numeric_cols"]
    categorical_cols = summary["categorical_cols"]
    datetime_cols = summary["datetime_cols"]
    insights = []
    highlights = []
    executive_lines = []

    rows, cols_n = summary["after_rows"], summary["after_cols"]
    executive_lines.append(
        f"This dataset contains {rows:,} records across {cols_n} dimensions after cleaning. "
        f"The overall data quality score is {summary['quality_score']}/100, "
        f"with {summary['duplicates_removed']} duplicate records removed and "
        f"{summary['missing_fixed']} missing values resolved."
    )

    # Numeric insights
    for col in numeric_cols[:6]:
        s = df[col].dropna()
        if len(s) < 2:
            continue
        mean_val = s.mean()
        median_val = s.median()
        std_val = s.std()
        skew = s.skew()
        total = s.sum()
        min_val, max_val = s.min(), s.max()

        skew_desc = "strongly right-skewed" if skew > 1 else "moderately right-skewed" if skew > 0.5 else "strongly left-skewed" if skew < -1 else "roughly symmetric"
        label = col.replace("_", " ").title()

        insights.append({
            "icon": "📊",
            "title": f"{label} Overview",
            "body": (
                f"Total {label}: <strong>{_smart_format(total)}</strong>. "
                f"Average: <strong>{_smart_format(mean_val)}</strong>, median: <strong>{_smart_format(median_val)}</strong>. "
                f"Range spans from {_smart_format(min_val)} to {_smart_format(max_val)} "
                f"with a standard deviation of {_smart_format(std_val)}. "
                f"Distribution is {skew_desc}."
            ),
            "category": "distribution",
        })

        # Trend (first half vs second half)
        mid = len(s) // 2
        first_avg = s.iloc[:mid].mean()
        second_avg = s.iloc[mid:].mean()
        change_pct = ((second_avg - first_avg) / max(abs(first_avg), 1e-9)) * 100
        direction = "increased" if change_pct > 5 else "decreased" if change_pct < -5 else "remained stable"
        if abs(change_pct) > 5:
            highlights.append({
                "icon": "📈" if change_pct > 0 else "📉",
                "label": f"{label} Trend",
                "value": f"{change_pct:+.1f}%",
                "color": "#34d399" if change_pct > 0 else "#f87171",
            })
            executive_lines.append(
                f"{label} has {direction} by {abs(change_pct):.1f}% from the first half to the second half of the dataset."
            )

    # Categorical insights
    for col in categorical_cols[:3]:
        vc = df[col].value_counts()
        if len(vc) < 2:
            continue
        label = col.replace("_", " ").title()
        top_val = vc.index[0]
        top_pct = vc.iloc[0] / vc.sum() * 100
        second_val = vc.index[1] if len(vc) > 1 else None

        insights.append({
            "icon": "🏆",
            "title": f"{label} Performance",
            "body": (
                f"<strong>{top_val}</strong> is the leading category in <em>{label}</em> "
                f"representing <strong>{top_pct:.1f}%</strong> of total records. "
                + (f"Runner-up is <strong>{second_val}</strong> with {vc.iloc[1]/vc.sum()*100:.1f}%. " if second_val else "")
                + f"Across {len(vc)} unique values, distribution "
                + ("is highly concentrated." if top_pct > 50 else "appears relatively balanced.")
            ),
            "category": "category",
        })
        highlights.append({
            "icon": "🥇",
            "label": f"Top {label}",
            "value": str(top_val)[:20],
            "color": "#6366f1",
        })
        executive_lines.append(
            f"In {label}, '{top_val}' is dominant at {top_pct:.1f}% share across {len(vc)} unique values."
        )

    # Correlation insights
    if len(numeric_cols) >= 2:
        corr = df[numeric_cols[:8]].corr()
        pairs = []
        cols_list = numeric_cols[:8]
        for i in range(len(cols_list)):
            for j in range(i + 1, len(cols_list)):
                r = corr.iloc[i, j]
                if not np.isnan(r):
                    pairs.append((cols_list[i], cols_list[j], r))
        pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        if pairs:
            a, b, r = pairs[0]
            direction = "positive" if r > 0 else "negative"
            strength = "strong" if abs(r) > 0.7 else "moderate" if abs(r) > 0.4 else "weak"
            insights.append({
                "icon": "🔗",
                "title": "Strongest Correlation",
                "body": (
                    f"<strong>{a.replace('_',' ').title()}</strong> and <strong>{b.replace('_',' ').title()}</strong> "
                    f"exhibit a <strong>{strength} {direction} correlation</strong> (r = {r:.3f}). "
                    + (f"This suggests that as {a.replace('_',' ')} increases, {b.replace('_',' ')} tends to {'increase' if r > 0 else 'decrease'} proportionally."
                       if abs(r) > 0.4 else "The relationship is statistically weak.")
                ),
                "category": "correlation",
            })
            executive_lines.append(
                f"Strongest correlation: {a.replace('_',' ').title()} ↔ {b.replace('_',' ').title()} (r={r:.3f}, {strength} {direction})."
            )

    # Time-series narrative
    if datetime_cols and numeric_cols:
        dcol, ncol = datetime_cols[0], numeric_cols[0]
        ts = df[[dcol, ncol]].dropna().sort_values(dcol)
        if len(ts) >= 4:
            label = ncol.replace("_", " ").title()
            first_val = ts[ncol].iloc[0]
            last_val = ts[ncol].iloc[-1]
            peak_date = ts.loc[ts[ncol].idxmax(), dcol]
            peak_val = ts[ncol].max()
            overall_change = ((last_val - first_val) / max(abs(first_val), 1e-9)) * 100
            insights.append({
                "icon": "📅",
                "title": f"{label} Time-Series Analysis",
                "body": (
                    f"Over the observed period, <strong>{label}</strong> moved from "
                    f"<strong>{_smart_format(first_val)}</strong> to <strong>{_smart_format(last_val)}</strong> "
                    f"({overall_change:+.1f}% overall change). "
                    f"Peak value of <strong>{_smart_format(peak_val)}</strong> occurred on "
                    f"<strong>{pd.Timestamp(peak_date).strftime('%b %d, %Y') if pd.notna(peak_date) else 'N/A'}</strong>."
                ),
                "category": "timeseries",
            })

    # Outlier commentary
    if summary.get("outlier_columns"):
        outlier_cols = list(summary["outlier_columns"].items())
        cols_str = ", ".join([f"{c.replace('_',' ').title()} ({n})" for c, n in outlier_cols[:4]])
        insights.append({
            "icon": "⚠️",
            "title": "Anomaly Detection",
            "body": (
                f"Outliers were detected in {len(outlier_cols)} column(s): <strong>{cols_str}</strong>. "
                f"These {summary['outliers_detected']} anomalous data points may represent "
                f"exceptional events, measurement errors, or high-value opportunities worth investigating."
            ),
            "category": "anomaly",
        })
        executive_lines.append(
            f"Anomaly note: {summary['outliers_detected']} outlier data points detected across {len(outlier_cols)} columns."
        )

    # Dataset health
    quality = summary["quality_score"]
    health_label = "Excellent" if quality >= 85 else "Good" if quality >= 70 else "Fair" if quality >= 50 else "Poor"
    executive_lines.append(
        f"Overall dataset health is rated <strong>{health_label}</strong> ({quality}/100). "
        f"The data is {'ready for production analysis.' if quality >= 70 else 'recommended for further validation before deployment.'}"
    )

    return {
        "executive_summary": " ".join(executive_lines),
        "insights": insights,
        "highlights": highlights,
        "quality_label": health_label,
    }


# ─────────────────────────────────────────────
#  ROUTES
# ─────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400

        f = request.files["file"]
        if not f.filename:
            return jsonify({"error": "No filename"}), 400

        fname = f.filename.lower()
        raw = f.read()

        if fname.endswith(".csv"):
            for enc in ("utf-8", "latin-1", "cp1252"):
                try:
                    df_raw = pd.read_csv(BytesIO(raw), encoding=enc, low_memory=False)
                    break
                except Exception:
                    continue
        elif fname.endswith((".xls", ".xlsx")):
            df_raw = pd.read_excel(BytesIO(raw))
        else:
            return jsonify({"error": "Only CSV and Excel files are supported"}), 400

        if df_raw.empty or df_raw.shape[0] < 2:
            return jsonify({"error": "File appears to be empty or has insufficient data"}), 400

        # ── Phase 1
        df_clean, data_summary = clean_data(df_raw.copy())

        # ── Phase 2
        charts = generate_visualizations(df_clean, data_summary)

        # ── Phase 3
        narrative = generate_narrative(df_clean, data_summary)

        # Preview table (first 8 rows, max 8 cols)
        preview_df = df_clean.head(8).iloc[:, :8]
        preview = {
            "headers": preview_df.columns.tolist(),
            "rows": preview_df.astype(str).values.tolist(),
        }

        return jsonify({
            "success": True,
            "filename": f.filename,
            "data_summary": data_summary,
            "charts": charts,
            "narrative": narrative,
            "preview": preview,
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Processing failed: {str(e)}"}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
