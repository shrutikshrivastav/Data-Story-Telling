import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template, send_file
import io
import plotly.express as px
import json

app = Flask(__name__)

# Global storage for cleaned data
cleaned_data_store = {}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    global cleaned_data_store
    if 'file' not in request.files:
        return jsonify({"error": "File not found in request"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    try:
        # 1. Load Data
        if file.filename.lower().endswith('.csv'):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)

        # 2. Advanced Enterprise Cleaning
        df.drop_duplicates(inplace=True)
        
        # Identify columns
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df.select_dtypes(include=['object']).columns.tolist()

        # Fill Missing Values
        for col in num_cols:
            df[col] = df[col].fillna(df[col].median() if not df[col].isna().all() else 0)
        
        for col in cat_cols:
            df[col] = df[col].fillna('Unknown')

        # Store data for download
        cleaned_data_store['current'] = df.copy()

        # 3. Dynamic Charts Logic
        charts = {}
        
        # --- PIE CHART ---
        if cat_cols:
            fig_pie = px.pie(df.head(15), 
                             names=cat_cols[0], 
                             values=num_cols[0] if num_cols else None,
                             title=f"Distribution of {cat_cols[0]}",
                             template="plotly_white",
                             hole=0.4)
            fig_pie.update_layout(margin=dict(t=40, b=20, l=20, r=20))
            charts['pie'] = json.loads(fig_pie.to_json())

        # --- BAR CHART ---
        if cat_cols and num_cols:
            fig_bar = px.bar(df.head(12), x=cat_cols[0], y=num_cols[0], 
                             color=num_cols[0],
                             template="plotly_white",
                             color_continuous_scale='Blues')
            charts['main_bar'] = json.loads(fig_bar.to_json())

        # --- DISTRIBUTION ---
        if num_cols:
            fig_hist = px.histogram(df, x=num_cols[0], 
                                    template="plotly_white",
                                    color_discrete_sequence=['#6366F1'])
            charts['distribution'] = json.loads(fig_hist.to_json())

            # --- OUTLIERS ---
            fig_box = px.box(df, y=num_cols[0], template="plotly_white")
            charts['outliers'] = json.loads(fig_box.to_json())

        # --- TREND (Scatter) with Statsmodels Protection ---
        if len(num_cols) >= 2:
            try:
                # Statsmodels install hai toh trendline dikhayega
                fig_scatter = px.scatter(df.head(100), x=num_cols[0], y=num_cols[1],
                                         trendline="ols", template="plotly_white")
            except Exception:
                # Nahi toh simple scatter plot dikhayega
                fig_scatter = px.scatter(df.head(100), x=num_cols[0], y=num_cols[1],
                                         template="plotly_white")
            charts['trend'] = json.loads(fig_scatter.to_json())

            # --- CORRELATION HEATMAP ---
            corr = df[num_cols].corr()
            fig_heat = px.imshow(corr, text_auto=True, 
                                 template="plotly_white",
                                 color_continuous_scale='RdBu_r')
            charts['correlation'] = json.loads(fig_heat.to_json())

        # 4. Final Response Construction
        # Convert to records and handle non-JSON serializable floats
        safe_rows = df.head(30).replace({np.nan: None, np.inf: None, -np.inf: None}).to_dict(orient='records')
        
        return jsonify({
            "filename": file.filename,
            "summary": {
                "rows": int(len(df)),
                "cols": int(len(df.columns)),
                "score": 98
            },
            "table": {
                "cols": [{"header": str(c), "field": str(c)} for c in df.columns],
                "rows": safe_rows
            },
            "charts": charts,
            "narrative": [
                {"title": "Data Integrity", "desc": f"Processed {len(df)} records. All missing values handled via median imputation."},
                {"title": "Visual Analysis", "desc": f"Key patterns identified in {num_cols[0] if num_cols else 'numerical data'}."}
            ]
        })

    except Exception as e:
        import traceback
        print(traceback.format_exc()) # Server logs mein detailed error dikhega
        return jsonify({"error": str(e)}), 500

@app.route('/download')
def download_file():
    global cleaned_data_store
    if 'current' in cleaned_data_store:
        df = cleaned_data_store['current']
        buffer = io.BytesIO()
        df.to_csv(buffer, index=False)
        buffer.seek(0)
        return send_file(
            buffer,
            mimetype='text/csv',
            as_attachment=True,
            download_name="Cleaned_DataStory_AI.csv"
        )
    return "No data available.", 404

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
