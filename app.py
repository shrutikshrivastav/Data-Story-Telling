import base64
import io
import json
import os
import re
from datetime import datetime

import numpy as np
import pandas as pd
import plotly
import plotly.graph_objects as go
import plotly.express as px
from flask import Flask, jsonify, render_template, request
from werkzeug.utils import secure_filename


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024

ALLOWED_EXTENSIONS = {"csv", "xlsx", "xls"}
PLOT_TEMPLATE = "plotly_dark"


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def normalize_column_name(column):
    name = str(column).strip().lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name or "column"


def make_unique_columns(columns):
    seen = {}
    unique = []
    for column in columns:
        base = normalize_column_name(column)
        count = seen.get(base, 0)
        seen[base] = count + 1
        unique.append(base if count == 0 else f"{base}_{count + 1}")
    return unique


def detect_outliers(df, numeric_columns):
    outlier_summary = {}
    outlier_rows = set()
    for column in numeric_columns:
        series = pd.to_numeric(df[column], errors="coerce").dropna()
        if series.empty:
            continue
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        mask = (pd.to_numeric(df[column], errors="coerce") < lower) | (
            pd.to_numeric(df[column], errors="coerce") > upper
        )
        count = int(mask.sum())
        if count:
            outlier_summary[column] = {
                "count": count,
                "lower_bound": round(float(lower), 3),
                "upper_bound": round(float(upper), 3),
            }
            outlier_rows.update(df.index[mask].tolist())
    return outlier_summary, len(outlier_rows)


def coerce_datatypes(df):
    converted = df.copy()
    date_columns = []

    for column in converted.columns:
        if converted[column].dtype == "object":
            numeric_candidate = pd.to_numeric(
                converted[column].astype(str).str.replace(",", "", regex=False).str.strip(),
                errors="coerce",
            )
            numeric_ratio = numeric_candidate.notna().mean()
            if numeric_ratio >= 0.82:
                converted[column] = numeric_candidate
                continue

            date_candidate = pd.to_datetime(converted[column], errors="coerce", infer_datetime_format=True)
            date_ratio = date_candidate.notna().mean()
            unique_dates = date_candidate.dropna().dt.date.nunique()
            if date_ratio >= 0.72 and unique_dates >= 2:
                converted[column] = date_candidate
                date_columns.append(column)

    for column in converted.select_dtypes(include=["datetime64[ns]", "datetimetz"]).columns:
        if column not in date_columns:
            date_columns.append(column)

    return converted, date_columns


def fill_missing_values(df):
    filled = df.copy()
    missing_before = int(filled.isna().sum().sum())

    for column in filled.columns:
        if filled[column].isna().sum() == 0:
            continue
        if pd.api.types.is_numeric_dtype(filled[column]):
            median = filled[column].median()
            filled[column] = filled[column].fillna(0 if pd.isna(median) else median)
        elif pd.api.types.is_datetime64_any_dtype(filled[column]):
            mode = filled[column].mode(dropna=True)
            replacement = mode.iloc[0] if not mode.empty else pd.Timestamp(datetime.utcnow().date())
            filled[column] = filled[column].fillna(replacement)
        else:
            mode = filled[column].mode(dropna=True)
            replacement = mode.iloc[0] if not mode.empty else "Unknown"
            filled[column] = filled[column].fillna(replacement)

    missing_after = int(filled.isna().sum().sum())
    return filled, missing_before, missing_before - missing_after


def clean_data(df):
    original_shape = df.shape
    original_missing = int(df.isna().sum().sum())
    original_duplicates = int(df.duplicated().sum())
    original_empty_rows = int(df.isna().all(axis=1).sum())

    cleaned = df.copy()
    cleaned.columns = make_unique_columns(cleaned.columns)
    cleaned = cleaned.replace(r"^\s*$", np.nan, regex=True)
    cleaned = cleaned.dropna(how="all")
    cleaned = cleaned.drop_duplicates()
    cleaned, date_columns = coerce_datatypes(cleaned)
    cleaned, missing_before, missing_fixed = fill_missing_values(cleaned)

    numeric_columns = cleaned.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = [
        column for column in cleaned.columns
        if column not in numeric_columns and column not in date_columns
    ]
    outliers, outlier_rows = detect_outliers(cleaned, numeric_columns)

    missing_rate = original_missing / max(original_shape[0] * original_shape[1], 1)
    duplicate_rate = original_duplicates / max(original_shape[0], 1)
    outlier_rate = outlier_rows / max(cleaned.shape[0], 1)
    quality_score = max(0, round(100 - (missing_rate * 35 + duplicate_rate * 25 + outlier_rate * 20) * 100, 1))

    summary = {
        "before": {"rows": int(original_shape[0]), "columns": int(original_shape[1]), "missing_values": original_missing},
        "after": {"rows": int(cleaned.shape[0]), "columns": int(cleaned.shape[1]), "missing_values": int(cleaned.isna().sum().sum())},
        "duplicates_removed": original_duplicates,
        "empty_rows_removed": original_empty_rows,
        "missing_values_fixed": missing_fixed,
        "outliers_detected": int(sum(item["count"] for item in outliers.values())),
        "outlier_rows_detected": int(outlier_rows),
        "quality_score": quality_score,
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "date_columns": date_columns,
        "outlier_summary": outliers,
    }
    return cleaned, summary


def encode_figure(fig):
    fig.update_layout(
        template=PLOT_TEMPLATE,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(3,7,18,0.48)",
        font={"color": "#e5e7eb", "family": "Inter, system-ui, sans-serif"},
        margin={"l": 48, "r": 24, "t": 66, "b": 56},
        hoverlabel={"bgcolor": "#111827", "font_size": 13},
        legend={"orientation": "h", "y": -0.18},
    )
    config = {
        "responsive": True,
        "displaylogo": False,
        "modeBarButtonsToAdd": ["drawline", "drawopenpath", "eraseshape"],
        "toImageButtonOptions": {"format": "png", "filename": "data_story_chart", "scale": 2},
    }
    return json.loads(json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)), config


def add_chart(charts, chart_id, title, chart_type, fig, explanation):
    figure, config = encode_figure(fig)
    charts.append({
        "id": chart_id,
        "title": title,
        "type": chart_type,
        "figure": figure,
        "config": config,
        "explanation": explanation,
    })


def generate_visualizations(df, summary):
    charts = []
    numeric = summary["numeric_columns"]
    categorical = summary["categorical_columns"]
    dates = summary["date_columns"]
    palette = ["#38bdf8", "#a78bfa", "#34d399", "#f59e0b", "#fb7185", "#22d3ee"]

    if categorical and numeric:
        cat, val = categorical[0], numeric[0]
        grouped = df.groupby(cat, dropna=False)[val].sum().sort_values(ascending=False).head(15).reset_index()
        fig = px.bar(grouped, x=cat, y=val, color=val, color_continuous_scale="Turbo", title=f"{val} by {cat}")
        fig.update_traces(marker_line_color="rgba(255,255,255,0.2)", marker_line_width=1)
        add_chart(
            charts,
            "bar-performance",
            "Category Performance",
            "Bar Chart",
            fig,
            f"Compares the total {val} across the strongest {cat} groups, making it easy to identify top contributors and underperforming segments.",
        )

    if dates and numeric:
        date_col, val = dates[0], numeric[0]
        timeline = df[[date_col, val]].dropna().copy()
        timeline[date_col] = pd.to_datetime(timeline[date_col], errors="coerce")
        timeline = timeline.dropna().sort_values(date_col)
        if not timeline.empty:
            grain = "M" if timeline[date_col].dt.date.nunique() > 40 else "D"
            trend = timeline.groupby(pd.Grouper(key=date_col, freq=grain))[val].sum().reset_index()
            fig = px.line(trend, x=date_col, y=val, markers=True, title=f"{val} Trend Over Time")
            fig.update_traces(line={"color": "#38bdf8", "width": 4}, marker={"size": 8})
            add_chart(
                charts,
                "line-trend",
                "Time Trend",
                "Line Chart",
                fig,
                f"Shows how {val} changes over time using the detected date field {date_col}, highlighting momentum, dips, and growth periods.",
            )

    if categorical:
        cat = categorical[0]
        counts = df[cat].astype(str).value_counts().head(8).reset_index()
        counts.columns = [cat, "records"]
        fig = px.pie(counts, names=cat, values="records", hole=0.45, title=f"{cat} Mix")
        fig.update_traces(textposition="inside", textinfo="percent+label", marker={"colors": palette})
        add_chart(
            charts,
            "pie-mix",
            "Segment Mix",
            "Pie Chart",
            fig,
            f"Displays the record distribution across the leading {cat} segments so the dominant mix of the dataset is immediately visible.",
        )

    if numeric:
        val = numeric[0]
        fig = px.histogram(df, x=val, nbins=32, marginal="box", title=f"{val} Distribution")
        fig.update_traces(marker_color="#a78bfa", marker_line_color="rgba(255,255,255,0.16)", marker_line_width=1)
        add_chart(
            charts,
            "hist-distribution",
            "Value Distribution",
            "Histogram",
            fig,
            f"Profiles the distribution of {val}, revealing concentration, skew, spread, and potential unusual values.",
        )

    if len(numeric) >= 2:
        x_col, y_col = numeric[0], numeric[1]
        color_col = categorical[0] if categorical else None
        fig = px.scatter(
            df.head(5000),
            x=x_col,
            y=y_col,
            color=color_col,
            title=f"Relationship Between {x_col} and {y_col}",
        )
        fig.update_traces(marker={"size": 9, "opacity": 0.78, "line": {"width": 0.5, "color": "#ffffff"}})
        scatter_source = df[[x_col, y_col]].dropna()
        if len(scatter_source) >= 3 and scatter_source[x_col].nunique() > 1:
            slope, intercept = np.polyfit(scatter_source[x_col], scatter_source[y_col], 1)
            x_line = np.linspace(scatter_source[x_col].min(), scatter_source[x_col].max(), 80)
            y_line = slope * x_line + intercept
            fig.add_trace(go.Scatter(
                x=x_line,
                y=y_line,
                mode="lines",
                name="Trend line",
                line={"color": "#facc15", "width": 3, "dash": "dot"},
            ))
        add_chart(
            charts,
            "scatter-relationship",
            "Relationship Analysis",
            "Scatter Plot",
            fig,
            f"Plots {x_col} against {y_col} to expose correlation, clustering, and segment-level differences in the data.",
        )

    if len(numeric) >= 2:
        corr = df[numeric].corr(numeric_only=True).round(2)
        fig = px.imshow(
            corr,
            text_auto=True,
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            title="Correlation Heatmap",
        )
        add_chart(
            charts,
            "correlation-heatmap",
            "Correlation Heatmap",
            "Heatmap",
            fig,
            "Maps numeric relationships from -1 to +1, helping reveal which measures move together or in opposite directions.",
        )

    return charts


def format_number(value):
    if pd.isna(value):
        return "n/a"
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if abs(value) >= 1_000:
        return f"{value / 1_000:.2f}K"
    return f"{value:.2f}" if isinstance(value, float) else str(value)


def generate_kpis(df, summary):
    kpis = []
    for column in summary["numeric_columns"][:4]:
        series = pd.to_numeric(df[column], errors="coerce").dropna()
        if series.empty:
            continue
        delta = ((series.iloc[-1] - series.iloc[0]) / abs(series.iloc[0]) * 100) if len(series) > 1 and series.iloc[0] != 0 else 0
        kpis.append({
            "label": column.replace("_", " ").title(),
            "value": format_number(float(series.mean())),
            "detail": f"Avg, max {format_number(float(series.max()))}",
            "trend": round(float(delta), 1),
        })

    if not kpis:
        kpis.append({
            "label": "Dataset Records",
            "value": format_number(summary["after"]["rows"]),
            "detail": "Clean rows available for storytelling",
            "trend": 0,
        })
    return kpis


def generate_narrative(df, summary):
    insights = []
    numeric = summary["numeric_columns"]
    categorical = summary["categorical_columns"]
    dates = summary["date_columns"]

    score = summary["quality_score"]
    if score >= 90:
        quality_text = "The dataset is highly reliable after cleaning, with very limited structural friction for analysis."
    elif score >= 75:
        quality_text = "The dataset is suitable for business analysis, though a few quality issues were corrected during preparation."
    else:
        quality_text = "The dataset required meaningful preparation before analysis, so downstream decisions should account for the original quality gaps."
    insights.append({
        "title": "Executive Data Readiness",
        "body": f"{quality_text} The pipeline removed {summary['duplicates_removed']} duplicates, fixed {summary['missing_values_fixed']} missing values, and detected {summary['outliers_detected']} outlier values.",
        "tone": "quality",
    })

    if categorical and numeric:
        cat, val = categorical[0], numeric[0]
        grouped = df.groupby(cat, dropna=False)[val].sum().sort_values(ascending=False)
        if len(grouped):
            top_name = str(grouped.index[0])
            top_value = float(grouped.iloc[0])
            share = top_value / grouped.sum() * 100 if grouped.sum() else 0
            insights.append({
                "title": "Category Leadership",
                "body": f"{top_name} is the strongest {cat.replace('_', ' ')} segment for {val.replace('_', ' ')}, contributing {share:.1f}% of the measured total. This segment should be treated as a key performance anchor.",
                "tone": "performance",
            })

    if dates and numeric:
        date_col, val = dates[0], numeric[0]
        timeline = df[[date_col, val]].dropna().copy()
        timeline[date_col] = pd.to_datetime(timeline[date_col], errors="coerce")
        timeline = timeline.dropna().sort_values(date_col)
        if len(timeline) >= 3:
            monthly = timeline.groupby(pd.Grouper(key=date_col, freq="M"))[val].sum().dropna()
            if len(monthly) >= 2:
                growth = (monthly.iloc[-1] - monthly.iloc[0]) / abs(monthly.iloc[0]) * 100 if monthly.iloc[0] else 0
                direction = "expanded" if growth >= 0 else "contracted"
                best_period = monthly.idxmax().strftime("%b %Y")
                insights.append({
                    "title": "Trend Momentum",
                    "body": f"{val.replace('_', ' ').title()} {direction} by {abs(growth):.1f}% from the first to the latest observed period. The strongest period was {best_period}, which marks the clearest peak in the time series.",
                    "tone": "trend",
                })

    if len(numeric) >= 2:
        corr = df[numeric].corr(numeric_only=True).abs()
        pairs = []
        for i, col_a in enumerate(corr.columns):
            for col_b in corr.columns[i + 1:]:
                value = corr.loc[col_a, col_b]
                if not pd.isna(value):
                    pairs.append((col_a, col_b, float(value)))
        if pairs:
            col_a, col_b, corr_value = sorted(pairs, key=lambda item: item[2], reverse=True)[0]
            relation = "strong" if corr_value >= 0.7 else "moderate" if corr_value >= 0.4 else "weak"
            insights.append({
                "title": "Driver Relationship",
                "body": f"The clearest numeric relationship is between {col_a.replace('_', ' ')} and {col_b.replace('_', ' ')}, with a {relation} correlation of {corr_value:.2f}. This pair deserves attention when explaining performance movement.",
                "tone": "correlation",
            })

    if categorical:
        cat = categorical[0]
        top_counts = df[cat].astype(str).value_counts().head(3)
        if not top_counts.empty:
            leaders = ", ".join([f"{idx} ({count} records)" for idx, count in top_counts.items()])
            insights.append({
                "title": "Segment Concentration",
                "body": f"The dataset is most concentrated in {leaders}. This concentration can shape conclusions, so segment mix should be considered before broad generalization.",
                "tone": "segment",
            })

    if summary["outliers_detected"] > 0:
        columns = ", ".join(list(summary["outlier_summary"].keys())[:4])
        insights.append({
            "title": "Outlier Watchlist",
            "body": f"Outliers were detected in {columns}. These values may represent exceptional business events, data-entry issues, or high-impact opportunities worth reviewing separately.",
            "tone": "risk",
        })

    return insights


def profile_preview(df):
    preview = df.head(8).copy()
    for column in preview.columns:
        if pd.api.types.is_datetime64_any_dtype(preview[column]):
            preview[column] = preview[column].dt.strftime("%Y-%m-%d")
    return {
        "columns": preview.columns.tolist(),
        "rows": preview.astype(object).where(pd.notna(preview), "").values.tolist(),
    }


def read_uploaded_file(file_storage):
    filename = secure_filename(file_storage.filename)
    extension = filename.rsplit(".", 1)[1].lower()
    contents = file_storage.read()
    buffer = io.BytesIO(contents)
    if extension == "csv":
        try:
            return pd.read_csv(buffer)
        except UnicodeDecodeError:
            buffer.seek(0)
            return pd.read_csv(buffer, encoding="latin-1")
    return pd.read_excel(buffer)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file was uploaded."}), 400

    file = request.files["file"]
    if not file or file.filename == "":
        return jsonify({"error": "Please choose a CSV or Excel file."}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "Only CSV, XLSX, and XLS files are supported."}), 400

    try:
        raw_df = read_uploaded_file(file)
        if raw_df.empty:
            return jsonify({"error": "The uploaded file does not contain usable rows."}), 400

        cleaned_df, summary = clean_data(raw_df)
        charts = generate_visualizations(cleaned_df, summary)
        narrative = generate_narrative(cleaned_df, summary)
        kpis = generate_kpis(cleaned_df, summary)

        csv_buffer = io.StringIO()
        cleaned_df.to_csv(csv_buffer, index=False)
        download_payload = base64.b64encode(csv_buffer.getvalue().encode("utf-8")).decode("utf-8")

        return jsonify({
            "filename": secure_filename(file.filename),
            "summary": summary,
            "kpis": kpis,
            "charts": charts,
            "narrative": narrative,
            "preview": profile_preview(cleaned_df),
            "cleaned_csv": download_payload,
        })
    except Exception as exc:
        app.logger.exception("Upload processing failed")
        return jsonify({"error": f"Unable to process this file: {exc}"}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
