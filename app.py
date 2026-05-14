import os
import uuid
import json
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import google.generativeai as genai
from flask import Flask, request, jsonify, render_template, send_file, session
from werkzeug.utils import secure_filename
from datetime import datetime
import traceback

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-prod')
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['CLEANED_FOLDER'] = 'cleaned'
app.config['CHARTS_FOLDER'] = 'charts'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB

# Create runtime folders
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['CLEANED_FOLDER'], exist_ok=True)
os.makedirs(app.config['CHARTS_FOLDER'], exist_ok=True)

# Configure Gemini API
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-pro')
else:
    gemini_model = None
    print("WARNING: GEMINI_API_KEY not set. AI features disabled.")

# ------------------------- Data Cleaning -------------------------
def normalize_column_names(df):
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    df.columns = df.columns.str.replace(r'[^a-z0-9_]', '', regex=True)
    return df

def detect_and_convert_dates(df):
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                converted = pd.to_datetime(df[col], errors='ignore')
                if converted.dtype == 'datetime64[ns]':
                    df[col] = converted
            except:
                pass
    return df

def handle_outliers(df, threshold=3):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - threshold * IQR
        upper = Q3 + threshold * IQR
        df[col] = df[col].clip(lower, upper)
    return df

def clean_data(df):
    df = df.drop_duplicates()
    df = normalize_column_names(df)
    df = df.dropna(how='all')
    df = df.dropna(axis=1, how='all')
    df = detect_and_convert_dates(df)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    
    for col in numeric_cols:
        if df[col].isnull().any():
            df[col].fillna(df[col].median(), inplace=True)
    for col in categorical_cols:
        if df[col].isnull().any():
            df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown', inplace=True)
    
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                df[col] = pd.to_numeric(df[col])
            except:
                pass
    
    df = handle_outliers(df)
    return df

# ------------------------- EDA & Charts -------------------------
def generate_eda_report(df):
    total_rows = len(df)
    total_cols = len(df.columns)
    missing_values = df.isnull().sum().sum()
    duplicates = df.duplicated().sum()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    date_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    missing_per_col = df.isnull().sum().to_dict()
    numeric_stats = {}
    for col in numeric_cols:
        numeric_stats[col] = {
            'mean': df[col].mean(),
            'median': df[col].median(),
            'std': df[col].std(),
            'min': df[col].min(),
            'max': df[col].max()
        }
    return {
        'total_rows': total_rows, 'total_cols': total_cols,
        'missing_values': missing_values, 'duplicates': duplicates,
        'numeric_cols': numeric_cols, 'categorical_cols': categorical_cols,
        'date_cols': date_cols, 'missing_per_col': missing_per_col,
        'numeric_stats': numeric_stats
    }

def generate_correlation_heatmap(df):
    numeric_df = df.select_dtypes(include=[np.number])
    if len(numeric_df.columns) >= 2:
        corr = numeric_df.corr()
        fig = go.Figure(data=go.Heatmap(
            z=corr.values, x=corr.columns, y=corr.columns,
            colorscale='RdBu', zmin=-1, zmax=1,
            text=corr.round(2).values, texttemplate='%{text}'
        ))
        fig.update_layout(title='Correlation Heatmap', height=500)
        return fig
    return None

def generate_histograms(df):
    figs = []
    for col in df.select_dtypes(include=[np.number]).columns[:3]:
        fig = px.histogram(df, x=col, title=f'Distribution of {col}', nbins=30)
        figs.append(fig)
    return figs

def generate_bar_chart(df):
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    if len(cat_cols) > 0:
        col = cat_cols[0]
        counts = df[col].value_counts().head(10)
        fig = px.bar(x=counts.index, y=counts.values, title=f'Top 10 in {col}')
        return fig
    return None

def generate_pie_chart(df):
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    if len(cat_cols) > 0:
        col = cat_cols[0]
        counts = df[col].value_counts().head(5)
        fig = px.pie(values=counts.values, names=counts.index, title=f'{col} Distribution')
        return fig
    return None

def generate_line_chart(df):
    date_cols = df.select_dtypes(include=['datetime64']).columns
    num_cols = df.select_dtypes(include=[np.number]).columns
    if len(date_cols) > 0 and len(num_cols) > 0:
        df_sorted = df.sort_values(date_cols[0])
        fig = px.line(df_sorted, x=date_cols[0], y=num_cols[0],
                      title=f'Trend: {num_cols[0]} over {date_cols[0]}')
        return fig
    return None

def generate_scatter_plot(df):
    num_cols = df.select_dtypes(include=[np.number]).columns
    if len(num_cols) >= 2:
        fig = px.scatter(df, x=num_cols[0], y=num_cols[1],
                         title=f'{num_cols[0]} vs {num_cols[1]}')
        return fig
    return None

def generate_all_charts(df):
    charts = []
    uid = str(uuid.uuid4())[:8]
    
    # Heatmap
    fig = generate_correlation_heatmap(df)
    if fig:
        path = os.path.join(app.config['CHARTS_FOLDER'], f'heatmap_{uid}.html')
        fig.write_html(path)
        charts.append(('Correlation Heatmap', fig.to_json(), path))
    
    # Histograms
    for i, fig in enumerate(generate_histograms(df)):
        path = os.path.join(app.config['CHARTS_FOLDER'], f'hist_{i}_{uid}.html')
        fig.write_html(path)
        charts.append((f'Histogram {i+1}', fig.to_json(), path))
    
    # Bar chart
    fig = generate_bar_chart(df)
    if fig:
        path = os.path.join(app.config['CHARTS_FOLDER'], f'barchart_{uid}.html')
        fig.write_html(path)
        charts.append(('Bar Chart', fig.to_json(), path))
    
    # Pie chart
    fig = generate_pie_chart(df)
    if fig:
        path = os.path.join(app.config['CHARTS_FOLDER'], f'pie_{uid}.html')
        fig.write_html(path)
        charts.append(('Pie Chart', fig.to_json(), path))
    
    # Line chart
    fig = generate_line_chart(df)
    if fig:
        path = os.path.join(app.config['CHARTS_FOLDER'], f'line_{uid}.html')
        fig.write_html(path)
        charts.append(('Time Series Trend', fig.to_json(), path))
    
    # Scatter plot
    fig = generate_scatter_plot(df)
    if fig:
        path = os.path.join(app.config['CHARTS_FOLDER'], f'scatter_{uid}.html')
        fig.write_html(path)
        charts.append(('Scatter Plot', fig.to_json(), path))
    
    return charts

# ------------------------- AI Insights -------------------------
def generate_insights(df):
    if not gemini_model:
        return "Gemini API key missing. Add GEMINI_API_KEY environment variable."
    
    eda = generate_eda_report(df)
    context = f"""
Dataset: {eda['total_rows']} rows, {eda['total_cols']} columns
Numeric: {', '.join(eda['numeric_cols'])}
Categorical: {', '.join(eda['categorical_cols'])}
Dates: {', '.join(eda['date_cols'])}
Missing values: {eda['missing_values']}
Sample: {df.head(3).to_string()}
"""
    prompt = f"""You are a senior business analyst. Write 4-5 paragraphs of insights, trends, risks, and recommendations based on this data. Professional tone.

{context}
Insights:"""
    try:
        resp = gemini_model.generate_content(prompt)
        return resp.text
    except Exception as e:
        return f"AI error: {str(e)}"

def answer_question(df, question):
    if not gemini_model:
        return "AI disabled: No API key."
    context = f"Columns: {list(df.columns)}\nFirst rows:\n{df.head(5).to_string()}"
    prompt = f"Answer the question based on this data.\nData:\n{context}\nQuestion: {question}\nAnswer:"
    try:
        resp = gemini_model.generate_content(prompt)
        return resp.text
    except Exception as e:
        return f"Error: {str(e)}"

# ------------------------- Flask Routes -------------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    ext = file.filename.split('.')[-1].lower()
    uid = str(uuid.uuid4())[:8]
    orig_path = os.path.join(app.config['UPLOAD_FOLDER'], f'{uid}_{secure_filename(file.filename)}')
    file.save(orig_path)
    
    try:
        if ext in ['xlsx', 'xls']:
            df = pd.read_excel(orig_path)
        else:
            df = pd.read_csv(orig_path)
    except Exception as e:
        return jsonify({'error': f'Read error: {str(e)}'}), 400
    
    cleaned_df = clean_data(df)
    cleaned_path = os.path.join(app.config['CLEANED_FOLDER'], f'{uid}_cleaned.csv')
    cleaned_df.to_csv(cleaned_path, index=False)
    
    session['cleaned_file'] = cleaned_path
    session['data_id'] = uid
    
    eda = generate_eda_report(cleaned_df)
    charts = generate_all_charts(cleaned_df)
    insights = generate_insights(cleaned_df)
    session['insights'] = insights
    
    preview = cleaned_df.head(100).to_dict(orient='records')
    columns = list(cleaned_df.columns)
    kpis = {
        'total_rows': eda['total_rows'],
        'total_cols': eda['total_cols'],
        'missing_values': eda['missing_values'],
        'duplicates': eda['duplicates'],
        'numeric_cols_count': len(eda['numeric_cols'])
    }
    
    chart_data = [{'title': t, 'json': j, 'file': f} for t, j, f in charts]
    
    return jsonify({
        'success': True, 'eda': eda, 'kpis': kpis,
        'preview': preview, 'columns': columns,
        'charts': chart_data, 'insights': insights, 'data_id': uid
    })

@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    question = data.get('question', '')
    if not question:
        return jsonify({'error': 'No question'}), 400
    cleaned_file = session.get('cleaned_file')
    if not cleaned_file or not os.path.exists(cleaned_file):
        return jsonify({'error': 'No dataset loaded'}), 400
    df = pd.read_csv(cleaned_file)
    answer = answer_question(df, question)
    return jsonify({'answer': answer})

@app.route('/download-cleaned')
def download_cleaned():
    path = session.get('cleaned_file')
    if not path or not os.path.exists(path):
        return "No cleaned file", 404
    return send_file(path, as_attachment=True, download_name='cleaned_data.csv')

@app.route('/download-insights')
def download_insights():
    insights = session.get('insights', '')
    if not insights:
        return "No insights", 404
    from io import StringIO
    si = StringIO()
    si.write(insights)
    si.seek(0)
    return send_file(si, as_attachment=True, download_name='ai_insights.txt', mimetype='text/plain')

@app.route('/download-chart/<filename>')
def download_chart(filename):
    safe_path = os.path.join(app.config['CHARTS_FOLDER'], os.path.basename(filename))
    if os.path.exists(safe_path):
        return send_file(safe_path, as_attachment=True, download_name=filename)
    return "Not found", 404

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
