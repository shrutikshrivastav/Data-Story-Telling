import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template
import io
import plotly.express as px
import plotly.graph_objects as go
import json

app = Flask(__name__)

def generate_insights(df):
    """Automated Statistical Storytelling Logic"""
    narrative = []
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()

    # 1. Executive Overview
    quality_score = int(100 - (df.isnull().sum().sum() / df.size * 100))
    narrative.append({
        "title": "Executive Summary",
        "desc": f"Dataset contains {len(df):,} records across {len(df.columns)} dimensions. Data integrity is verified at {quality_score}%."
    })

    # 2. Statistical Correlation Insight
    if len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr()
        # Find strongest relationship
        corr_unstacked = corr.abs().unstack().sort_values(ascending=False)
        strongest = corr_unstacked[corr_unstacked < 1].idxmax()
        val = corr.loc[strongest[0], strongest[1]]
        narrative.append({
            "title": "Predictive Relationship",
            "desc": f"Strong correlation detected between '{strongest[0]}' and '{strongest[1]}' (r = {val:.2f}). This indicates a scalable pattern for forecasting."
        })

    # 3. Categorical Distribution
    if cat_cols:
        top_cat = df[cat_cols[0]].value_counts().idxmax()
        narrative.append({
            "title": "Market Segmentation",
            "desc": f"Primary dominant segment identified as '{top_cat}', representing the highest frequency in the current data batch."
        })

    return narrative, quality_score

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    filename = file.filename

    try:
        # Load Data
        if filename.endswith('.csv'):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)

        # --- DATA CLEANING (The 'Topper' Secret) ---
        # 1. Fix Missing Values
        for col in df.columns:
            if df[col].dtype in [np.float64, np.int64]:
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna('Unknown')

        # 2. Generate Stats & Narrative
        narrative, quality_score = generate_insights(df)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        # --- CHART GENERATION ---
        charts = {}
        
        # A. Main Bar Chart (First Categorical vs First Numeric)
        cat_cols = df.select_dtypes(include=['object']).columns.tolist()
        if cat_cols and numeric_cols:
            fig_bar = px.bar(df.head(20), x=cat_cols[0], y=numeric_cols[0], 
                           title=f"Top Segments by {numeric_cols[0]}",
                           color_discrete_sequence=['#4F46E5'])
            charts['main_bar'] = json.loads(fig_bar.to_json())

        # B. Distribution Chart
        if numeric_cols:
            fig_dist = px.histogram(df, x=numeric_cols[0], 
                                  title=f"Frequency Distribution: {numeric_cols[0]}",
                                  color_discrete_sequence=['#818CF8'])
            charts['distribution'] = json.loads(fig_dist.to_json())

        # C. Correlation Heatmap
        if len(numeric_cols) >= 2:
            corr = df[numeric_cols].corr()
            fig_heat = px.imshow(corr, text_auto=True, aspect="auto",
                                title="Statistical Correlation Matrix",
                                color_continuous_scale='RdBu_r')
            charts['correlation'] = json.loads(fig_heat.to_json())

        # --- PREPARE FINAL JSON ---
        # Handling NaN/Inf to prevent JSON errors
        df_display = df.head(100).replace({np.nan: None, np.inf: None, -np.inf: None})
        
        response = {
            "filename": filename,
            "summary": {
                "rows": len(df),
                "cols": len(df.columns),
                "score": quality_score
            },
            "table": {
                "cols": [{"header": c, "field": c} for c in df.columns],
                "rows": df_display.to_dict(orient='records')
            },
            "charts": charts,
            "narrative": narrative
        }

        return jsonify(response)

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"error": f"Analysis failed: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
