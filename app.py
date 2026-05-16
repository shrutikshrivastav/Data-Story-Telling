import base64
import io
import json
import os
import re
import time
import chardet
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

def detect_encoding(file_bytes):
    """Detect file encoding automatically"""
    detector = chardet.UniversalDetector()
    for line in file_bytes.splitlines()[:100]:
        detector.feed(line)
        if detector.done:
            break
    detector.close()
    return detector.result['encoding'] or 'utf-8'

def read_file_safely(file_storage):
    """Read CSV or Excel file with proper encoding detection"""
    filename = secure_filename(file_storage.filename)
    extension = filename.rsplit('.', 1)[1].lower()
    
    if extension == 'csv':
        # Read raw bytes for encoding detection
        file_bytes = file_storage.read()
        encoding = detect_encoding(file_bytes)
        file_storage.seek(0)
        
        # Try different encodings
        encodings_to_try = [encoding, 'utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
        
        for enc in encodings_to_try:
            try:
                file_storage.seek(0)
                return pd.read_csv(file_storage, encoding=enc, encoding_errors='replace')
            except:
                continue
        
        # Last resort
        file_storage.seek(0)
        return pd.read_csv(file_storage, encoding='latin-1', on_bad_lines='skip')
    
    elif extension in ['xlsx', 'xls']:
        try:
            return pd.read_excel(file_storage, engine='openpyxl')
        except:
            file_storage.seek(0)
            return pd.read_excel(file_storage, engine='xlrd')
    
    else:
        raise ValueError(f"Unsupported file type: {extension}")

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
    df.columns = [clean_column_name(col) for col in df.columns]
    
    # Remove completely empty rows
    df = df.dropna(how='all')
    
    # Remove duplicates
    df = df.drop_duplicates()
    
    # Handle missing values
    for col in df.columns:
        if df[col].isna().sum() > 0:
            if pd.api.types.is_numeric_dtype(df[col]):
                median_val = df[col].median()
                if pd.isna(median_val):
                    median_val = 0
                df[col].fillna(median_val, inplace=True)
            else:
                mode_val = df[col].mode()
                if len(mode_val) > 0 and not pd.isna(mode_val[0]):
                    df[col].fillna(mode_val[0], inplace=True)
                else:
                    df[col].fillna("Unknown", inplace=True)
    
    # Convert types safely
    for col in df.columns:
        if df[col].dtype == 'object':
            # Try numeric conversion
            try:
                numeric_attempt = pd.to_numeric(df[col], errors='coerce')
                if numeric_attempt.notna().mean() > 0.8:
                    df[col] = numeric_attempt
            except:
                pass
            
            # Try date conversion
            if df[col].dtype == 'object':
                try:
                    date_attempt = pd.to_datetime(df[col], errors='coerce')
                    if date_attempt.notna().mean() > 0.7:
                        df[col] = date_attempt
                except:
                    pass
    
    # Detect column types
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    date_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    
    # Calculate quality score
    final_missing = df.isna().sum().sum()
    total_cells = original_rows * original_cols
    missing_rate = original_missing / total_cells if total_cells > 0 else 0
    duplicate_rate = original_duplicates / original_rows if original_rows > 0 else 0
    
    quality_score = max(0, min(100, 100 - (missing_rate * 30 + duplicate_rate * 20) * 100))
    
    processing_time = round(time.time() - start_time, 2)
    
    return {
        'df': df,
        'summary': {
            'before': {
                'rows': original_rows, 
                'columns': original_cols, 
                'missing': int(original_missing)
            },
            'after': {
                'rows': len(df), 
                'columns': len(df.columns), 
                'missing': int(final_missing)
            },
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
    
    if not numeric_cols:
        return charts
    
    # Sample for performance
    plot_df = df if len(df) <= 3000 else df.sample(n=3000, random_state=42)
    
    # 1. Bar Chart
    if categorical_cols and numeric_cols:
        try:
            cat = categorical_cols[0]
            num = numeric_cols[0]
            top_data = df.groupby(cat)[num].sum().sort_values(ascending=False).head(10).reset_index()
            if len(top_data) > 1:
                fig = px.bar(top_data, x=cat, y=num, color=num, 
                            color_continuous_scale='Blues',
                            title=f'{num} by {cat}')
                fig.update_layout(
                    height=420,
                    template='plotly_dark',
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#94a3b8')
                )
                charts.append({
                    'id': 'chart1',
                    'title': f'Top {cat} by {num}',
                    'type': 'bar',
                    'figure': json.loads(json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)),
                    'insight': f'{cat} shows significant variation in {num} values.'
                })
        except Exception as e:
            print(f"Bar chart error: {e}")
    
    # 2. Time Series
    if date_cols and numeric_cols:
        try:
            date_col = date_cols[0]
            num_col = numeric_cols[0]
            timeline = df[[date_col, num_col]].dropna().copy()
            timeline[date_col] = pd.to_datetime(timeline[date_col], errors='coerce')
            timeline = timeline.dropna().sort_values(date_col)
            
            if len(timeline) > 3:
                fig = px.line(timeline, x=date_col, y=num_col, 
                             title=f'{num_col} over time',
                             markers=True)
                fig.update_layout(
                    height=420,
                    template='plotly_dark',
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                charts.append({
                    'id': 'chart2',
                    'title': 'Time Series Analysis',
                    'type': 'line',
                    'figure': json.loads(json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)),
                    'insight': f'{num_col} shows temporal patterns that may indicate seasonality.'
                })
        except Exception as e:
            print(f"Time series error: {e}")
    
    # 3. Distribution
    if numeric_cols:
        try:
            num = numeric_cols[0]
            fig = px.histogram(plot_df, x=num, nbins=30, 
                              marginal='box',
                              title=f'Distribution of {num}')
            fig.update_layout(
                height=420,
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            charts.append({
                'id': 'chart3',
                'title': f'{num} Distribution',
                'type': 'histogram',
                'figure': json.loads(json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)),
                'insight': 'Distribution shape indicates data concentration and potential outliers.'
            })
        except Exception as e:
            print(f"Histogram error: {e}")
    
    # 4. Correlation Heatmap
    if len(numeric_cols) >= 2:
        try:
            corr_matrix = df[numeric_cols].corr()
            fig = px.imshow(corr_matrix, text_auto=True, aspect='auto',
                           color_continuous_scale='RdBu_r',
                           title='Correlation Matrix')
            fig.update_layout(
                height=480,
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            charts.append({
                'id': 'chart4',
                'title': 'Variable Correlations',
                'type': 'heatmap',
                'figure': json.loads(json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)),
                'insight': 'Strong correlations suggest relationships between variables.'
            })
        except Exception as e:
            print(f"Heatmap error: {e}")
    
    # 5. Pie Chart
    if categorical_cols:
        try:
            cat = categorical_cols[0]
            top_cats = df[cat].value_counts().head(6).reset_index()
            top_cats.columns = [cat, 'count']
            if len(top_cats) > 1:
                fig = px.pie(top_cats, names=cat, values='count', 
                            title=f'{cat} Distribution',
                            hole=0.3)
                fig.update_layout(
                    height=420,
                    template='plotly_dark',
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                charts.append({
                    'id': 'chart5',
                    'title': f'{cat} Breakdown',
                    'type': 'pie',
                    'figure': json.loads(json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)),
                    'insight': f'{cat} distribution shows data composition across categories.'
                })
        except Exception as e:
            print(f"Pie chart error: {e}")
    
    # 6. Scatter Plot
    if len(numeric_cols) >= 2:
        try:
            x_col = numeric_cols[0]
            y_col = numeric_cols[1]
            fig = px.scatter(plot_df.head(1000), x=x_col, y=y_col,
                            color=categorical_cols[0] if categorical_cols else None,
                            title=f'{x_col} vs {y_col}',
                            opacity=0.6,
                            trendline='ols' if len(plot_df) > 10 else None)
            fig.update_layout(
                height=420,
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            charts.append({
                'id': 'chart6',
                'title': f'{x_col} vs {y_col}',
                'type': 'scatter',
                'figure': json.loads(json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)),
                'insight': f'Relationship between {x_col} and {y_col} shows correlation pattern.'
            })
        except Exception as e:
            print(f"Scatter error: {e}")
    
    return charts

def generate_insights(df, summary):
    """Generate data insights"""
    insights = []
    numeric_cols = summary['numeric_columns']
    categorical_cols = summary['categorical_columns']
    
    # Quality insight
    quality = summary['quality_score']
    if quality >= 85:
        quality_text = "Dataset quality is excellent with minimal preprocessing required."
    elif quality >= 70:
        quality_text = "Dataset quality is good with standard cleaning applied."
    else:
        quality_text = "Dataset required significant cleaning but is now ready for analysis."
    
    insights.append({
        'title': 'Data Quality Assessment',
        'content': f"{quality_text} Quality score: {quality}/100. Removed {summary['duplicates_removed']} duplicate records.",
        'icon': 'quality'
    })
    
    # Top performer
    if categorical_cols and numeric_cols:
        try:
            cat = categorical_cols[0]
            num = numeric_cols[0]
            grouped = df.groupby(cat)[num].sum().sort_values(ascending=False)
            if len(grouped) > 0:
                top_category = grouped.index[0]
                top_value = grouped.iloc[0]
                total = grouped.sum()
                share = (top_value / total * 100) if total > 0 else 0
                insights.append({
                    'title': 'Category Leadership',
                    'content': f"{top_category} leads in {num} with {top_value:,.2f}, representing {share:.1f}% of total.",
                    'icon': 'performance'
                })
        except:
            pass
    
    # Trend insight
    if summary['date_columns'] and numeric_cols:
        insights.append({
            'title': 'Temporal Patterns',
            'content': f"Time-based analysis shows variations in {numeric_cols[0]}. Monitor these patterns for forecasting.",
            'icon': 'trend'
        })
    
    # Correlation insight
    if len(numeric_cols) >= 2:
        try:
            corr_matrix = df[numeric_cols].corr()
            max_corr = 0
            corr_pair = ""
            for i in range(len(numeric_cols)):
                for j in range(i+1, len(numeric_cols)):
                    val = abs(corr_matrix.iloc[i, j])
                    if val > max_corr and val < 0.99:
                        max_corr = val
                        corr_pair = f"{numeric_cols[i]} and {numeric_cols[j]}"
            
            if max_corr > 0.5:
                strength = "strong" if max_corr > 0.7 else "moderate"
                direction = "positive" if corr_matrix.iloc[i, j] > 0 else "negative"
                insights.append({
                    'title': 'Variable Relationships',
                    'content': f"{strength.capitalize()} {direction} correlation ({max_corr:.2f}) found between {corr_pair}.",
                    'icon': 'correlation'
                })
        except:
            pass
    
    # Outlier detection
    outlier_count = 0
    outlier_cols = []
    for col in numeric_cols[:3]:
        try:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            if iqr > 0:
                outliers = df[(df[col] < q1 - 1.5*iqr) | (df[col] > q3 + 1.5*iqr)]
                if len(outliers) > 0:
                    outlier_count += len(outliers)
                    outlier_cols.append(col)
        except:
            pass
    
    if outlier_count > 0:
        insights.append({
            'title': 'Anomaly Detection',
            'content': f"Found {outlier_count} potential outliers in {', '.join(outlier_cols[:2])}. These may represent exceptional cases.",
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
        # Read file with proper encoding
        df = read_file_safely(file)
        
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
            if col in cleaned_df.columns and len(cleaned_df[col].dropna()) > 0:
                mean_val = cleaned_df[col].mean()
                if len(cleaned_df) > 1 and cleaned_df[col].iloc[0] != 0:
                    change = ((cleaned_df[col].iloc[-1] - cleaned_df[col].iloc[0]) / abs(cleaned_df[col].iloc[0]) * 100)
                else:
                    change = 0
                kpis.append({
                    'label': col.replace('_', ' ').title(),
                    'value': f"{mean_val:,.2f}",
                    'change': round(change, 1)
                })
        
        # Prepare download
        csv_buffer = io.StringIO()
        cleaned_df.to_csv(csv_buffer, index=False)
        download_payload = base64.b64encode(csv_buffer.getvalue().encode()).decode()
        
        # Preview (first 8 rows)
        preview_df = cleaned_df.head(8).fillna('')
        for col in preview_df.columns:
            if pd.api.types.is_datetime64_any_dtype(preview_df[col]):
                preview_df[col] = preview_df[col].dt.strftime('%Y-%m-%d')
        
        return jsonify({
            'success': True,
            'filename': file.filename,
            'summary': summary,
            'kpis': kpis,
            'charts': charts,
            'insights': insights,
            'preview': {
                'columns': preview_df.columns.tolist(),
                'rows': preview_df.astype(str).values.tolist()
            },
            'download_url': download_payload
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Processing error: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
