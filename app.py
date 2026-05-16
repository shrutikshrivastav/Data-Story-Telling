import os
import io
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
from flask import Flask, render_template, request, jsonify
from datetime import datetime

app = Flask(__name__)

# ---------- Data Cleaning ----------
def clean_data(df):
    before_shape = df.shape

    # Normalize column names
    df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]

    # Remove duplicates
    duplicates_removed = df.duplicated().sum()
    df = df.drop_duplicates()

    # Handle missing values
    missing_values_fixed = df.isnull().sum().sum()
    df = df.dropna(how="all")
    df = df.fillna(method="ffill").fillna(method="bfill")

    # Detect numeric and categorical
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()

    # Detect date columns
    for col in categorical_cols:
        try:
            df[col] = pd.to_datetime(df[col])
        except Exception:
            pass

    # Detect outliers
    outliers_detected = detect_outliers(df[numeric_cols]) if numeric_cols else 0

    after_shape = df.shape
    quality_score = max(0, 100 - (duplicates_removed + missing_values_fixed + outliers_detected))

    summary = {
        "before_rows": before_shape[0],
        "before_cols": before_shape[1],
        "after_rows": after_shape[0],
        "after_cols": after_shape[1],
        "duplicates_removed": int(duplicates_removed),
        "missing_values_fixed": int(missing_values_fixed),
        "outliers_detected": int(outliers_detected),
        "quality_score": quality_score
    }

    return df, summary, numeric_cols, categorical_cols


def detect_outliers(df):
    outliers = 0
    for col in df.columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outliers += ((df[col] < lower) | (df[col] > upper)).sum()
    return int(outliers)


# ---------- Visualization ----------
def generate_visualizations(df, numeric_cols, categorical_cols):
    figs = []

    if numeric_cols:
        figs.append(px.histogram(df, x=numeric_cols[0], title="Distribution Histogram"))
        figs.append(px.scatter(df, x=numeric_cols[0], y=numeric_cols[-1], title="Scatter Plot"))
        figs.append(px.line(df, x=df.index, y=numeric_cols[0], title="Trend Line"))

        corr = df[numeric_cols].corr()
        heatmap = ff.create_annotated_heatmap(
            z=corr.values,
            x=list(corr.columns),
            y=list(corr.index),
            colorscale="Viridis"
        )
        heatmap.update_layout(title="Correlation Heatmap")
        figs.append(heatmap)

    if categorical_cols:
        figs.append(px.bar(df, x=categorical_cols[0], title="Category Bar Chart"))
        figs.append(px.pie(df, names=categorical_cols[0], title="Category Pie Chart"))

    for fig in figs:
        fig.update_layout(template="plotly_dark", transition_duration=500)

    return [fig.to_html(full_html=False) for fig in figs]


# ---------- Narrative ----------
def generate_narrative(df, summary, numeric_cols, categorical_cols):
    insights = []

    insights.append(f"The dataset contains {summary['after_rows']} rows and {summary['after_cols']} columns after cleaning.")
    insights.append(f"Duplicates removed: {summary['duplicates_removed']}, Missing values fixed: {summary['missing_values_fixed']}, Outliers detected: {summary['outliers_detected']}.")
    insights.append(f"Overall dataset quality score: {summary['quality_score']}%.")

    if numeric_cols:
        for col in numeric_cols:
            max_val = df[col].max()
            min_val = df[col].min()
            mean_val = df[col].mean()
            insights.append(f"Column '{col}' ranges from {min_val:.2f} to {max_val:.2f} with an average of {mean_val:.2f}.")

    if categorical_cols:
        for col in categorical_cols:
            top_cat = df[col].value_counts().idxmax()
            insights.append(f"Category '{col}' is dominated by '{top_cat}'.")

    insights.append(f"Analysis generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.")

    return insights


# ---------- Routes ----------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    file = request.files["file"]
    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    if file.filename.endswith(".csv"):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)

    cleaned_df, summary, numeric_cols, categorical_cols = clean_data(df)
    visualizations = generate_visualizations(cleaned_df, numeric_cols, categorical_cols)
    narrative = generate_narrative(cleaned_df, summary, numeric_cols, categorical_cols)

    return jsonify({
        "summary": summary,
        "visualizations": visualizations,
        "narrative": narrative
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
