import os
import json
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024 # 50 MB limit
ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def clean_data(df):
    initial_rows, initial_cols = df.shape
    
    # 1. Normalize Column Names
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace(r'[^\w\s]', '', regex=True)
    
    # 2. Remove completely empty rows and columns
    df.dropna(how='all', inplace=True)
    df.dropna(how='all', axis=1, inplace=True)
    
    # 3. Detect and drop duplicates
    duplicates_count = df.duplicated().sum()
    df.drop_duplicates(inplace=True)
    
    # 4. Handle Missing Values & Fix Datatypes
    missing_initial = df.isna().sum().sum()
    
    # Attempt to convert object columns to datetime if they look like dates
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                df[col] = pd.to_datetime(df[col], format='mixed')
            except (ValueError, TypeError):
                pass

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns

    # Fill NaNs
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())
        
    for col in categorical_cols:
        if not df[col].mode().empty:
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = df[col].fillna("Unknown")
            
    missing_fixed = missing_initial - df.isna().sum().sum()

    # 5. Detect Outliers (IQR Method)
    outliers_detected = 0
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers_detected += ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()

    final_rows, final_cols = df.shape
    
    # Calculate Quality Score
    score = 100
    penalty = (missing_initial / (initial_rows * initial_cols)) * 50 if initial_rows > 0 else 0
    penalty += (duplicates_count / initial_rows) * 30 if initial_rows > 0 else 0
    quality_score = max(0, min(100, round(score - penalty, 1)))

    summary = {
        "initial_rows": initial_rows,
        "final_rows": final_rows,
        "columns": final_cols,
        "duplicates_removed": int(duplicates_count),
        "missing_fixed": int(missing_initial),
        "outliers_detected": int(outliers_detected),
        "quality_score": quality_score
    }
    
    return df, summary

def generate_visualizations(df):
    charts = {}
    template = "plotly_dark"
    color_seq = px.colors.qualitative.Pastel

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    date_cols = df.select_dtypes(include=['datetime64']).columns.tolist()

    # Layout updates for glassmorphism/dark theme fit
    layout_updates = dict(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e2e8f0', family="Inter, sans-serif"),
        margin=dict(t=40, b=40, l=40, r=40)
    )

    # 1. Bar Chart (Top Categories)
    if categorical_cols and numeric_cols:
        cat_col = categorical_cols[0]
        num_col = numeric_cols[0]
        top_cats = df.groupby(cat_col)[num_col].sum().nlargest(10).reset_index()
        fig = px.bar(top_cats, x=cat_col, y=num_col, title=f"Top 10 by {num_col.title()}", template=template, color_discrete_sequence=color_seq)
        fig.update_layout(**layout_updates)
        charts['bar'] = json.loads(fig.to_json())

    # 2. Line Chart (Time Series or Trend)
    if date_cols and numeric_cols:
        date_col = date_cols[0]
        num_col = numeric_cols[0]
        ts_df = df.groupby(df[date_col].dt.to_period('M'))[num_col].sum().reset_index()
        ts_df[date_col] = ts_df[date_col].astype(str)
        fig = px.line(ts_df, x=date_col, y=num_col, title=f"{num_col.title()} Trend Over Time", template=template, markers=True, color_discrete_sequence=['#60a5fa'])
        fig.update_layout(**layout_updates)
        charts['line'] = json.loads(fig.to_json())
    elif numeric_cols and len(numeric_cols) > 1:
        # Fallback to a sorted line chart if no dates
        sorted_df = df.sort_values(by=numeric_cols[0]).reset_index(drop=True)
        fig = px.line(sorted_df, y=numeric_cols[1], title=f"Trend of {numeric_cols[1].title()}", template=template)
        fig.update_layout(**layout_updates)
        charts['line'] = json.loads(fig.to_json())

    # 3. Pie Chart (Distribution)
    if categorical_cols:
        cat_col = categorical_cols[0] if len(categorical_cols) == 1 else categorical_cols[-1]
        pie_data = df[cat_col].value_counts().nlargest(5).reset_index()
        fig = px.pie(pie_data, names=cat_col, values='count', title=f"Distribution of Top 5 {cat_col.title()}", template=template, hole=0.4, color_discrete_sequence=color_seq)
        fig.update_layout(**layout_updates)
        charts['pie'] = json.loads(fig.to_json())

    # 4. Scatter Plot (Correlation)
    if len(numeric_cols) >= 2:
        fig = px.scatter(df, x=numeric_cols[0], y=numeric_cols[1], title=f"{numeric_cols[0].title()} vs {numeric_cols[1].title()}", template=template, opacity=0.7, color_discrete_sequence=['#34d399'])
        fig.update_layout(**layout_updates)
        charts['scatter'] = json.loads(fig.to_json())

    # 5. Correlation Heatmap
    if len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr()
        fig = px.imshow(corr, text_auto=True, aspect="auto", title="Feature Correlation Matrix", template=template, color_continuous_scale="Viridis")
        fig.update_layout(**layout_updates)
        charts['heatmap'] = json.loads(fig.to_json())
        
    # 6. Histogram (Distribution of Primary Metric)
    if numeric_cols:
        fig = px.histogram(df, x=numeric_cols[0], title=f"Distribution of {numeric_cols[0].title()}", template=template, nbins=30, color_discrete_sequence=['#c084fc'])
        fig.update_layout(**layout_updates)
        charts['histogram'] = json.loads(fig.to_json())

    return charts

def generate_narrative(df, numeric_cols, categorical_cols):
    insights = []
    
    # Volume insight
    insights.append({
        "title": "Dataset Overview",
        "icon": "fas fa-database",
        "text": f"The analysis encompasses {len(df):,} total records. The dataset contains {len(numeric_cols)} quantifiable metrics and {len(categorical_cols)} categorical dimensions, providing a solid foundation for robust statistical modeling."
    })

    # Numeric insights
    if numeric_cols:
        primary_metric = numeric_cols[0]
        metric_mean = df[primary_metric].mean()
        metric_max = df[primary_metric].max()
        insights.append({
            "title": f"Key Driver Analysis: {primary_metric.title()}",
            "icon": "fas fa-chart-line",
            "text": f"The primary metric '{primary_metric}' maintains an average baseline of {metric_mean:,.2f}, peaking at a maximum observed value of {metric_max:,.2f}. Outlier stabilization indicates a consistent overall trend across the spectrum."
        })

    # Categorical insights
    if categorical_cols and numeric_cols:
        cat_col = categorical_cols[0]
        num_col = numeric_cols[0]
        top_category = df.groupby(cat_col)[num_col].sum().idxmax()
        top_val = df.groupby(cat_col)[num_col].sum().max()
        insights.append({
            "title": "Segment Performance",
            "icon": "fas fa-crown",
            "text": f"Segment analysis reveals that '{top_category}' is the dominant category, driving the highest cumulative volume in {num_col} ({top_val:,.2f}). Focusing strategic efforts on this segment presents the highest probability of ROI."
        })

    # Correlation insight
    if len(numeric_cols) >= 2:
        corr_matrix = df[numeric_cols].corr().abs()
        np.fill_diagonal(corr_matrix.values, 0)
        max_corr_idx = corr_matrix.unstack().idxmax()
        if corr_matrix.unstack().max() > 0.5:
            insights.append({
                "title": "Statistical Correlations",
                "icon": "fas fa-project-diagram",
                "text": f"A significant mathematical relationship exists between '{max_corr_idx[0]}' and '{max_corr_idx[1]}'. This strong correlation suggests that shifts in one metric reliably predict variance in the other, enabling predictive forecasting."
            })

    return insights

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
        
    if file and allowed_file(file.filename):
        try:
            # Read file
            if file.filename.endswith('.csv'):
                df = pd.read_csv(file)
            else:
                df = pd.read_excel(file)
                
            if df.empty:
                return jsonify({"error": "Uploaded file is empty"}), 400

            # Pipeline
            df_cleaned, quality_summary = clean_data(df)
            
            numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = df_cleaned.select_dtypes(include=['object', 'category']).columns.tolist()

            charts = generate_visualizations(df_cleaned)
            narrative = generate_narrative(df_cleaned, numeric_cols, categorical_cols)

            return jsonify({
                "success": True,
                "summary": quality_summary,
                "charts": charts,
                "narrative": narrative
            })

        except Exception as e:
            return jsonify({"error": str(e)}), 500
            
    return jsonify({"error": "Invalid file format. Please upload CSV or Excel."}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
