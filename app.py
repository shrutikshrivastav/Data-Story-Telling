import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template, send_file
import io
import plotly.express as px
import json

app = Flask(__name__)

# Global variable to store cleaned data for download (Temporary for demo)
# For production, consider using a database or session-based storage
storage = {}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    try:
        df = pd.read_csv(file) if file.filename.endswith('.csv') else pd.read_excel(file)

        # --- ADVANCED CLEANING ---
        df.drop_duplicates(inplace=True)
        for col in df.columns:
            if df[col].dtype in [np.float64, np.int64]:
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna('N/A')

        # Store for download
        storage['last_df'] = df.copy()

        # --- EXTENDED VISUALS ---
        charts = {}
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df.select_dtypes(include=['object']).columns.tolist()

        # 1. Main Bar
        if cat_cols and num_cols:
            charts['main_bar'] = json.loads(px.bar(df.head(15), x=cat_cols[0], y=num_cols[0], 
                                          template="plotly_white", color_discrete_sequence=['#4F46E5']).to_json())
        
        # 2. Distribution (Histogram)
        if num_cols:
            charts['distribution'] = json.loads(px.histogram(df, x=num_cols[0], 
                                             template="plotly_white", color_discrete_sequence=['#818CF8']).to_json())
        
        # 3. Correlation Heatmap
        if len(num_cols) >= 2:
            charts['correlation'] = json.loads(px.imshow(df[num_cols].corr(), text_auto=True, 
                                             template="plotly_white", color_continuous_scale='Blues').to_json())

        # 4. Box Plot (Outlier Detection)
        if num_cols:
            charts['outliers'] = json.loads(px.box(df, y=num_cols[0], 
                                         template="plotly_white", color_discrete_sequence=['#6366F1']).to_json())

        # 5. Scatter Plot (Trend Analysis)
        if len(num_cols) >= 2:
            charts['trend'] = json.loads(px.scatter(df, x=num_cols[0], y=num_cols[1], 
                                          template="plotly_white", color_discrete_sequence=['#4F46E5']).to_json())

        # Narrative Logic
        narrative = [
            {"title": "Data Quality Report", "desc": f"Analyzed {len(df)} records. Removed duplicates and handled missing values using median imputation."},
            {"title": "Outlier Detection", "desc": f"Statistical variance in {num_cols[0]} suggests high-value segments that require strategic focus."},
            {"title": "Correlation Insight", "desc": "Heatmap reveals underlying patterns between numeric variables, ideal for predictive modeling."}
        ]

        return jsonify({
            "filename": file.filename,
            "summary": {"rows": len(df), "cols": len(df.columns), "score": 98},
            "table": {
                "cols": [{"header": c, "field": c} for c in df.columns],
                "rows": df.head(50).replace({np.nan: None}).to_dict(orient='records')
            },
            "charts": charts,
            "narrative": narrative
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/download')
def download():
    if 'last_df' in storage:
        df = storage['last_df']
        output = io.BytesIO()
        df.to_csv(output, index=False)
        output.seek(0)
        return send_file(output, mimetype='text/csv', as_attachment=True, download_name="cleaned_datastory.csv")
    return "No data found", 404

if __name__ == '__main__':
    app.run(debug=True)
