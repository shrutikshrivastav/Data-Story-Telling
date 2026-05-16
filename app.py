import base64
import io
import json
import os
import re
import time
import hashlib
from datetime import datetime, timezone
from functools import lru_cache

import numpy as np
import pandas as pd
import plotly
import plotly.graph_objects as go
import plotly.express as px
from flask import Flask, jsonify, render_template, request, send_file
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0

ALLOWED_EXTENSIONS = {"csv", "xlsx", "xls"}
PLOT_TEMPLATE = "plotly_dark"

# Cache for frequent operations
@lru_cache(maxsize=128)
def cached_read_csv(file_hash):
    pass

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

def detect_outliers_optimized(df, numeric_columns):
    """Optimized outlier detection using vectorized operations"""
    outlier_summary = {}
    outlier_rows = set()
    
    for column in numeric_columns:
        series = pd.to_numeric(df[column], errors="coerce")
        if series.dropna().empty:
            continue
            
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
            
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        mask = (series < lower) | (series > upper)
        count = int(mask.sum())
        
        if count:
            outlier_summary[column] = {
                "count": count,
                "lower_bound": round(float(lower), 3),
                "upper_bound": round(float(upper), 3),
            }
            outlier_rows.update(df.index[mask].tolist())
    
    return outlier_summary, len(outlier_rows)

def coerce_datatypes_optimized(df):
    """Faster datatype coercion"""
    converted = df.copy()
    date_columns = []
    
    # Process in parallel for numeric conversion
    for column in converted.columns:
        if converted[column].dtype == "object":
            # Try numeric conversion first
            numeric_candidate = pd.to_numeric(
                converted[column].astype(str).str.replace(",", "", regex=False).str.strip(),
                errors="coerce"
            )
            if numeric_candidate.notna().mean() >= 0.82:
                converted[column] = numeric_candidate
                continue
            
            # Try date conversion
            date_candidate = pd.to_datetime(converted[column], errors="coerce")
            if date_candidate.notna().mean() >= 0.72 and date_candidate.dropna().dt.date.nunique() >= 2:
                converted[column] = date_candidate
                date_columns.append(column)
    
    return converted, date_columns

def fill_missing_values_optimized(df):
    """Faster missing value imputation"""
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
            replacement = mode.iloc[0] if not mode.empty else pd.Timestamp.now()
            filled[column] = filled[column].fillna(replacement)
        else:
            mode = filled[column].mode(dropna=True)
            filled[column] = filled[column].fillna(mode.iloc[0] if not mode.empty else "Unknown")
    
    missing_after = int(filled.isna().sum().sum())
    return filled, missing_before, missing_before - missing_after

def clean_data_optimized(df):
    """Optimized cleaning pipeline"""
    start_time = time.time()
    
    original_shape = df.shape
    original_missing = int(df.isna().sum().sum())
    original_duplicates = int(df.duplicated().sum())
    original_empty_rows = int(df.isna().all(axis=1).sum())
    
    # Cleaning pipeline
    cleaned = df.copy()
    cleaned.columns = make_unique_columns(cleaned.columns)
    cleaned = cleaned.replace(r"^\s*$", np.nan, regex=True)
    cleaned = cleaned.dropna(how="all")
    cleaned = cleaned.drop_duplicates()
    cleaned, date_columns = coerce_datatypes_optimized(cleaned)
    cleaned, missing_before, missing_fixed = fill_missing_values_optimized(cleaned)
    
    # Optimized outlier detection
    numeric_columns = cleaned.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = [col for col in cleaned.columns if col not in numeric_columns and col not in date_columns]
    outliers, outlier_rows = detect_outliers_optimized(cleaned, numeric_columns)
    
    # Quality score calculation
    total_cells = original_shape[0] * original_shape[1]
    missing_rate = original_missing / max(total_cells, 1)
    duplicate_rate = original_duplicates / max(original_shape[0], 1)
    outlier_rate = outlier_rows / max(cleaned.shape[0], 1)
    quality_score = max(0, round(100 - (missing_rate * 35 + duplicate_rate * 25 + outlier_rate * 20) * 100, 1))
    
    summary = {
        "before": {"rows": int(original_shape[0]), "columns": int(original_shape[1]), "missing_values": original_missing},
        "after": {"rows": int(cleaned.shape[0]), "columns": int(cleaned.shape[1]), "missing_values": int(cleaned.isna().sum().sum())},
        "duplicates_removed": original_duplicates,
        "empty_rows_removed": original_empty_rows,
        "missing_values_fixed": missing_fixed,
        "outliers_detected": sum(item["count"] for item in outliers.values()),
        "outlier_rows_detected": int(outlier_rows),
        "quality_score": quality_score,
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "date_columns": date_columns,
        "outlier_summary": outliers,
        "processing_time": round(time.time() - start_time, 2)
    }
    
    return cleaned, summary

def encode_figure(fig):
    """Encode figure with proper responsive config"""
    fig.update_layout(
        template=PLOT_TEMPLATE,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(3,7,18,0.48)",
        font={"color": "#e5e7eb", "family": "Inter, system-ui, sans-serif"},
        margin={"l": 40, "r": 40, "t": 60, "b": 40},
        hoverlabel={"bgcolor": "#111827", "font_size": 13},
        legend={"orientation": "h", "y": -0.15},
        autosize=True,
        height=450
    )
    
    config = {
        "responsive": True,
        "displaylogo": False,
        "modeBarButtonsToRemove": ["lasso2d", "select2d"],
        "toImageButtonOptions": {"format": "png", "filename": "chart", "scale": 2},
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

def generate_visualizations_optimized(df, summary):
    """Generate visualizations with better error handling"""
    charts = []
    numeric = summary["numeric_columns"]
    categorical = summary["categorical_columns"]
    dates = summary["date_columns"]
    
    if not numeric:
        return charts
    
    # Sample large datasets for performance
    sample_df = df if len(df) <= 5000 else df.sample(n=5000, random_state=42)
    
    # 1. Bar Chart (if categorical and numeric exist)
    if categorical and numeric:
        try:
            cat, val = categorical[0], numeric[0]
            grouped = df.groupby(cat, dropna=False)[val].sum().sort_values(ascending=False).head(15).reset_index()
            if len(grouped) > 1:
                fig = px.bar(grouped, x=cat, y=val, color=val, 
                            color_continuous_scale="Turbo", 
                            title=f"{val} by {cat}")
                fig.update_layout(xaxis_tickangle=-45)
                add_chart(charts, "bar-performance", "Category Performance", "Bar Chart", fig,
                         f"Shows total {val} across top {cat} segments")
        except Exception as e:
            print(f"Bar chart error: {e}")
    
    # 2. Time Series (if dates exist)
    if dates and numeric and len(dates) > 0 and len(numeric) > 0:
        try:
            date_col, val = dates[0], numeric[0]
            timeline = df[[date_col, val]].dropna().copy()
            timeline[date_col] = pd.to_datetime(timeline[date_col], errors='coerce')
            timeline = timeline.dropna().sort_values(date_col)
            
            if len(timeline) > 2:
                grain = 'M' if timeline[date_col].nunique() > 40 else 'D'
                trend = timeline.groupby(pd.Grouper(key=date_col, freq=grain))[val].sum().reset_index()
                fig = px.line(trend, x=date_col, y=val, markers=True, 
                             title=f"{val} Trend Over Time")
                fig.update_traces(line=dict(color="#38bdf8", width=3), marker=dict(size=6))
                add_chart(charts, "line-trend", "Time Series Analysis", "Line Chart", fig,
                         f"Shows {val} trends over time")
        except Exception as e:
            print(f"Time series error: {e}")
    
    # 3. Distribution Histogram
    if numeric:
        try:
            val = numeric[0]
            fig = px.histogram(sample_df, x=val, nbins=30, marginal="box", 
                              title=f"{val} Distribution")
            fig.update_traces(marker_color="#a78bfa")
            add_chart(charts, "hist-distribution", "Data Distribution", "Histogram", fig,
                     f"Shows how {val} values are distributed across the dataset")
        except Exception as e:
            print(f"Histogram error: {e}")
    
    # 4. Correlation Heatmap
    if len(numeric) >= 2:
        try:
            corr = df[numeric].corr(numeric_only=True).round(2)
            fig = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r",
                           zmin=-1, zmax=1, title="Correlation Heatmap",
                           aspect="auto")
            fig.update_layout(height=500)
            add_chart(charts, "correlation-heatmap", "Correlation Matrix", "Heatmap", fig,
                     "Shows relationships between numeric variables")
        except Exception as e:
            print(f"Heatmap error: {e}")
    
    # 5. Scatter Plot (if multiple numeric columns)
    if len(numeric) >= 2:
        try:
            x_col, y_col = numeric[0], numeric[1]
            fig = px.scatter(sample_df.head(2000), x=x_col, y=y_col,
                            title=f"{x_col} vs {y_col}",
                            opacity=0.6, trendline="ols")
            fig.update_traces(marker=dict(size=8))
            add_chart(charts, "scatter-relationship", "Correlation Analysis", "Scatter Plot", fig,
                     f"Shows relationship between {x_col} and {y_col}")
        except Exception as e:
            print(f"Scatter error: {e}")
    
    # 6. Pie Chart (if categorical exists)
    if categorical:
        try:
            cat = categorical[0]
            counts = df[cat].astype(str).value_counts().head(6).reset_index()
            counts.columns = [cat, "count"]
            fig = px.pie(counts, names=cat, values="count", hole=0.4,
                        title=f"{cat} Distribution")
            add_chart(charts, "pie-distribution", "Category Distribution", "Pie Chart", fig,
                     f"Shows distribution across {cat} categories")
        except Exception as e:
            print(f"Pie chart error: {e}")
    
    return charts

def generate_kpis_optimized(df, summary):
    """Generate KPIs faster"""
    kpis = []
    for column in summary["numeric_columns"][:4]:
        series = pd.to_numeric(df[column], errors="coerce").dropna()
        if len(series) == 0:
            continue
        
        # Calculate trend if time series
        trend = 0
        if len(series) > 10:
            trend = ((series.iloc[-5:].mean() - series.iloc[:5].mean()) / 
                    abs(series.iloc[:5].mean()) * 100 if series.iloc[:5].mean() != 0 else 0)
        
        kpis.append({
            "label": column.replace("_", " ").title(),
            "value": f"{series.mean():,.2f}",
            "detail": f"Min: {series.min():,.1f} | Max: {series.max():,.1f}",
            "trend": round(float(trend), 1),
        })
    
    if not kpis:
        kpis.append({
            "label": "Total Records",
            "value": f"{summary['after']['rows']:,}",
            "detail": "Rows after cleaning",
            "trend": 0,
        })
    return kpis

def generate_narrative_optimized(df, summary):
    """Generate business narrative"""
    insights = []
    numeric = summary["numeric_columns"]
    categorical = summary["categorical_columns"]
    
    # Quality insight
    score = summary["quality_score"]
    if score >= 85:
        quality_text = "Excellent data quality - ready for analysis"
    elif score >= 70:
        quality_text = "Good data quality with minor issues resolved"
    else:
        quality_text = "Fair quality - significant cleaning performed"
    
    insights.append({
        "title": "Data Quality Assessment",
        "body": f"{quality_text}. {summary['duplicates_removed']} duplicates removed, {summary['missing_values_fixed']} missing values fixed.",
        "tone": "quality"
    })
    
    # Top performer insight
    if categorical and numeric and len(categorical) > 0 and len(numeric) > 0:
        try:
            grouped = df.groupby(categorical[0])[numeric[0]].sum().sort_values(ascending=False)
            if len(grouped) > 0:
                top = grouped.index[0]
                value = grouped.iloc[0]
                insights.append({
                    "title": "Top Performing Segment",
                    "body": f"{top} leads in {numeric[0]} with {value:,.0f} total value, representing {value/grouped.sum()*100:.1f}% of total.",
                    "tone": "performance"
                })
        except:
            pass
    
    # Trend insight
    if summary["date_columns"] and numeric:
        insights.append({
            "title": "Temporal Pattern",
            "body": f"Time-based analysis reveals patterns in {numeric[0]}. Monitor seasonal trends for better forecasting.",
            "tone": "trend"
        })
    
    # Correlation insight
    if len(numeric) >= 2:
        corr = df[numeric].corr(numeric_only=True).abs().unstack()
        if len(corr) > 0:
            max_corr = corr[corr < 1].max()
            if not pd.isna(max_corr):
                insights.append({
                    "title": "Key Relationship",
                    "body": f"Strong correlation detected ({max_corr:.2f}) between variables. This suggests potential causal relationships worth investigating.",
                    "tone": "correlation"
                })
    
    return insights

def read_uploaded_file_optimized(file_storage):
    """Read file with better memory management"""
    filename = secure_filename(file_storage.filename)
    extension = filename.rsplit(".", 1)[1].lower()
    
    if extension == "csv":
        # Read in chunks for large files
        try:
            return pd.read_csv(file_storage)
        except:
            file_storage.seek(0)
            return pd.read_csv(file_storage, encoding='latin-1')
    else:
        return pd.read_excel(file_storage)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400
    
    if not allowed_file(file.filename):
        return jsonify({"error": "Only CSV, XLSX, XLS files allowed"}), 400
    
    try:
        start_total = time.time()
        
        # Read file
        raw_df = read_uploaded_file_optimized(file)
        
        if raw_df.empty:
            return jsonify({"error": "File is empty"}), 400
        
        # Clean data
        cleaned_df, summary = clean_data_optimized(raw_df)
        
        # Generate outputs
        charts = generate_visualizations_optimized(cleaned_df, summary)
        narrative = generate_narrative_optimized(cleaned_df, summary)
        kpis = generate_kpis_optimized(cleaned_df, summary)
        
        # Prepare download
        csv_buffer = io.StringIO()
        cleaned_df.head(1000).to_csv(csv_buffer, index=False)  # Limit download size
        download_payload = base64.b64encode(csv_buffer.getvalue().encode("utf-8")).decode("utf-8")
        
        # Preview (first 10 rows)
        preview_df = cleaned_df.head(10).copy()
        for col in preview_df.columns:
            if pd.api.types.is_datetime64_any_dtype(preview_df[col]):
                preview_df[col] = preview_df[col].dt.strftime("%Y-%m-%d")
        
        total_time = round(time.time() - start_total, 2)
        summary["total_processing_time"] = total_time
        
        return jsonify({
            "filename": secure_filename(file.filename),
            "summary": summary,
            "kpis": kpis,
            "charts": charts,
            "narrative": narrative,
            "preview": {
                "columns": preview_df.columns.tolist(),
                "rows": preview_df.astype(object).where(pd.notna(preview_df), "").values.tolist()
            },
            "cleaned_csv": download_payload,
        })
        
    except Exception as e:
        app.logger.error(f"Upload error: {str(e)}")
        return jsonify({"error": f"Processing failed: {str(e)}"}), 500

@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": "File too large (max 32MB)"}), 413

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
