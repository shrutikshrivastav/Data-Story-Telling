import base64
import io
import json
import os
import re
import time
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
app.config["SECRET_KEY"] = "your-secret-key-here"

ALLOWED_EXTENSIONS = {"csv", "xlsx", "xls"}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def clean_column_name(col):
    """Clean column names for better display"""
    name = str(col).strip().lower()
    name = re.sub(r'[^a-z0-9]+', '_', name)
    name = name.strip('_')
    return name or 'column'

def process_data(df):
    """Main data processing pipeline"""
    start_time = time.time()
    
    # Store original stats
    original_rows = len(df)
    original_cols = len(df.columns)
    original_missing = df.isna().sum().sum()
    original_duplicates = df.duplicated().sum()
    
    # Clean column names
    df.columns = [clean_name(col) for col in df.columns]
    
    # Remove completely empty rows
    df = df.dropna(how='all')
    
    # Remove duplicates
    df = df.drop_duplicates()
    
    # Handle missing values
    for col in df.columns:
        if df[col].isna().sum() > 0:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col].fillna(df[col].median() if not df[col].isna().all() else 0, inplace=True)
            else:
                df[col].fillna("Unknown", inplace=True)
    
    # Convert types
    for col in df.columns:
        if df[col].dtype == 'object':
            # Try numeric
            numeric_attempt = pd.to_numeric(df[col], errors='coerce')
            if numeric_attempt.notna().mean() > 0.8:
                df[col] = numeric_attempt
            else:
                # Try date
                date_attempt = pd.to_datetime(df[col], errors='coerce')
                if date_attempt.notna().mean() > 0.7:
                    df[col] = date_attempt
    
    # Detect numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    date_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    
    # Calculate quality score
    final_missing = df.isna().sum().sum()
    missing_rate = original_missing / (original_rows * original_cols) if original_rows * original_cols > 0 else 0
    duplicate_rate = original_duplicates / original_rows if original_rows > 0 else 0
    
    quality_score = max(0, min(100, 100 - (missing_rate * 30 + duplicate_rate * 20) * 100))
    
    processing_time = round(time.time() - start_time, 2)
    
    return {
        'df': df,
        'summary': {
            'before': {'rows': original_rows, 'columns': original_cols, 'missing': int(original_missing)},
            'after': {'rows': len(df), 'columns': len(df.columns), 'missing': int(final_missing)},
            'duplicates_removed': int(original_duplicates),
            'quality_score': round(quality_score, 1),
            'numeric_columns': numeric_cols,
            'categorical_columns': categorical_cols,
            'date_columns': date_cols,
            'processing_time': processing_time
        }
    }

def create_charts(df, summary):
    """Generate interactive charts"""
    charts = []
    numeric_cols = summary['numeric_columns']
    categorical_cols = summary['categorical_columns']
    date_cols = summary['date_columns']
    
    # Sample large datasets for performance
    plot_df = df if len(df) <= 3000 else df.sample(n=3000, random_state=42)
    
    # 1. Key Metrics Bar Chart
    if categorical_cols and numeric_cols:
        try:
            cat = categorical_cols[0]
            num = numeric_cols[0]
            top_data = df.groupby(cat)[num].sum().sort_values(ascending=False).head(10).reset_index()
            fig = px.bar(top_data, x=cat, y=num, color=num, 
                        color_continuous_scale='Viridis',
                        title=f'Top {cat} by {num}')
            fig.update_layout(height=450, template='plotly_dark')
            charts.append({
                'id': 'chart1',
                'title': f'{num} Distribution by Category',
                'type': 'bar',
                'figure': json.loads(json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)),
                'insight': f'The top performing category shows significant variation in {num} values.'
            })
        except:
            pass
    
    # 2. Trend Line Chart
    if date_cols and numeric_cols:
        try:
            date_col = date_cols[0]
            num_col = numeric_cols[0]
            timeline = df[[date_col, num_col]].dropna().copy()
            timeline[date_col] = pd.to_datetime(timeline[date_col])
            timeline = timeline.sort_values(date_col)
            if len(timeline) > 1:
                fig = px.line(timeline, x=date_col, y=num_col, 
                             title=f'{num_col} Over Time',
                             markers=True)
                fig.update_layout(height=450, template='plotly_dark')
                charts.append({
                    'id': 'chart2',
                    'title': 'Time Series Analysis',
                    'type': 'line',
                    'figure': json.loads(json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)),
                    'insight': f'{num_col} shows temporal patterns worth monitoring.'
                })
        except:
            pass
    
    # 3. Distribution Histogram
    if numeric_cols:
        try:
            num = numeric_cols[0]
            fig = px.histogram(plot_df, x=num, nbins=30, 
                              marginal='box',
                              title=f'Distribution of {num}')
            fig.update_layout(height=450, template='plotly_dark')
            charts.append({
                'id': 'chart3',
                'title': f'{num} Distribution Pattern',
                'type': 'histogram',
                'figure': json.loads(json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)),
                'insight': 'The distribution reveals data concentration patterns and potential outliers.'
            })
        except:
            pass
    
    # 4. Correlation Heatmap
    if len(numeric_cols) >= 2:
        try:
            corr_matrix = df[numeric_cols].corr()
            fig = px.imshow(corr_matrix, text_auto=True, aspect='auto',
                           color_continuous_scale='RdBu_r',
                           title='Feature Correlation Matrix')
            fig.update_layout(height=500, template='plotly_dark')
            charts.append({
                'id': 'chart4',
                'title': 'Correlation Analysis',
                'type': 'heatmap',
                'figure': json.loads(json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)),
                'insight': 'Strong correlations indicate relationships between variables.'
            })
        except:
            pass
    
    # 5. Pie Chart
    if categorical_cols:
        try:
            cat = categorical_cols[0]
            top_cats = df[cat].value_counts().head(6).reset_index()
            top_cats.columns = [cat, 'count']
            fig = px.pie(top_cats, names=cat, values='count', 
                        title=f'{cat} Distribution',
                        hole=0.3)
            fig.update_layout(height=450, template='plotly_dark')
            charts.append({
                'id': 'chart5',
                'title': f'{cat} Breakdown',
                'type': 'pie',
                'figure': json.loads(json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)),
                'insight': 'Category distribution shows data composition and dominant segments.'
            })
        except:
            pass
    
    # 6. Scatter Plot
    if len(numeric_cols) >= 2:
        try:
            fig = px.scatter(plot_df.head(1000), x=numeric_cols[0], y=numeric_cols[1],
                            color=categorical_cols[0] if categorical_cols else None,
                            title=f'{numeric_cols[0]} vs {numeric_cols[1]}',
                            opacity=0.6)
            fig.update_layout(height=450, template='plotly_dark')
            charts.append({
                'id': 'chart6',
                'title': 'Relationship Analysis',
                'type': 'scatter',
                'figure': json.loads(json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)),
                'insight': 'Scatter plot reveals correlation patterns between key metrics.'
            })
        except:
            pass
    
    return charts

def generate_insights(df, summary):
    """Generate AI-powered insights"""
    insights = []
    numeric_cols = summary['numeric_columns']
    categorical_cols = summary['categorical_columns']
    
    # Data quality insight
    quality = summary['quality_score']
    if quality >= 85:
        quality_text = "Excellent data quality with minimal preprocessing required."
    elif quality >= 70:
        quality_text = "Good data quality with some missing values handled."
    else:
        quality_text = "Fair data quality - significant cleaning was performed."
    
    insights.append({
        'title': '📊 Data Quality Assessment',
        'content': f"{quality_text} Quality Score: {quality}/100. {summary['duplicates_removed']} duplicates removed.",
        'icon': 'quality'
    })
    
    # Top performer insight
    if categorical_cols and numeric_cols:
        try:
            cat = categorical_cols[0]
            num = numeric_cols[0]
            top = df.groupby(cat)[num].sum().sort_values(ascending=False).head(1)
            if len(top) > 0:
                insights.append({
                    'title': '🏆 Top Performance Insight',
                    'content': f"{top.index[0]} leads in {num} with {top.values[0]:,.2f} total value.",
                    'icon': 'performance'
                })
        except:
            pass
    
    # Trend insight
    if summary['date_columns'] and numeric_cols:
        insights.append({
            'title': '📈 Trend Detection',
            'content': f"Time-based patterns detected in {numeric_cols[0]}. Monitor seasonal variations for better forecasting.",
            'icon': 'trend'
        })
    
    # Correlation insight
    if len(numeric_cols) >= 2:
        try:
            corr = df[numeric_cols].corr().values
            max_corr = np.max(corr[corr < 0.99])
            if max_corr > 0.7:
                insights.append({
                    'title': '🔗 Strong Correlation Found',
                    'content': f"Strong relationship detected between variables ({max_corr:.2f}). This suggests predictive potential.",
                    'icon': 'correlation'
                })
        except:
            pass
    
    # Outlier insight
    outlier_count = 0
    for col in numeric_cols[:3]:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        outliers = df[(df[col] < q1 - 1.5*iqr) | (df[col] > q3 + 1.5*iqr)]
        outlier_count += len(outliers)
    
    if outlier_count > 0:
        insights.append({
            'title': '⚠️ Anomaly Detection',
            'content': f"Detected {outlier_count} potential outliers that may represent exceptional business events.",
            'icon': 'outlier'
        })
    
    return insights

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Only CSV, XLSX, XLS files allowed'}), 400
    
    try:
        # Read file
        filename = secure_filename(file.filename)
        ext = filename.rsplit('.', 1)[1].lower()
        
        if ext == 'csv':
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)
        
        if df.empty:
            return jsonify({'error': 'File is empty'}), 400
        
        # Process data
        result = process_data(df)
        cleaned_df = result['df']
        summary = result['summary']
        
        # Generate visualizations
        charts = create_charts(cleaned_df, summary)
        
        # Generate insights
        insights = generate_insights(cleaned_df, summary)
        
        # Generate KPIs
        kpis = []
        for col in summary['numeric_columns'][:4]:
            if col in cleaned_df.columns:
                mean_val = cleaned_df[col].mean()
                kpis.append({
                    'label': col.replace('_', ' ').title(),
                    'value': f"{mean_val:,.2f}",
                    'change': round((cleaned_df[col].iloc[-1] - cleaned_df[col].iloc[0]) / cleaned_df[col].iloc[0] * 100 if len(cleaned_df) > 1 and cleaned_df[col].iloc[0] != 0 else 0, 1)
                })
        
        # Prepare download
        csv_buffer = io.StringIO()
        cleaned_df.to_csv(csv_buffer, index=False)
        download_payload = base64.b64encode(csv_buffer.getvalue().encode()).decode()
        
        # Preview
        preview = cleaned_df.head(8).fillna('').astype(str)
        
        return jsonify({
            'success': True,
            'filename': filename,
            'summary': summary,
            'kpis': kpis,
            'charts': charts,
            'insights': insights,
            'preview': {
                'columns': preview.columns.tolist(),
                'rows': preview.values.tolist()
            },
            'download_url': download_payload
        })
        
    except Exception as e:
        return jsonify({'error': f'Processing error: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
