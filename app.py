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
import traceback

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-prod')
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['CLEANED_FOLDER'] = 'cleaned'
app.config['CHARTS_FOLDER'] = 'charts'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'supersecretkey123')

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
            'mean': float(df[col].mean()) if not pd.isna(df[col].mean()) else 0,
            'median': float(df[col].median()) if not pd.isna(df[col].median()) else 0,
            'std': float(df[col].std()) if not pd.isna(df[col].std()) else 0,
            'min': float(df[col].min()) if not pd.isna(df[col].min()) else 0,
            'max': float(df[col].max()) if not pd.isna(df[col].max()) else 0
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
            text=corr.round(2).values, texttemplate='%{text}', textfont={"size": 10}
        ))
        fig.update_layout(title='Correlation Heatmap', height=500, template='plotly_dark')
        return fig
    return None

def generate_histograms(df):
    figs = []
    for col in df.select_dtypes(include=[np.number]).columns[:3]:
        fig = px.histogram(df, x=col, title=f'Distribution of {col}', nbins=30, template='plotly_dark')
        figs.append(fig)
    return figs

def generate_bar_chart(df):
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    if len(cat_cols) > 0:
        col = cat_cols[0]
        counts = df[col].value_counts().head(10)
        fig = px.bar(x=counts.index, y=counts.values, title=f'Top 10 in {col}', template='plotly_dark')
        fig.update_layout(xaxis_title=col, yaxis_title='Count')
        return fig
    return None

def generate_pie_chart(df):
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    if len(cat_cols) > 0:
        col = cat_cols[0]
        counts = df[col].value_counts().head(5)
        fig = px.pie(values=counts.values, names=counts.index, title=f'{col} Distribution', template='plotly_dark')
        return fig
    return None

def generate_line_chart(df):
    date_cols = df.select_dtypes(include=['datetime64']).columns
    num_cols = df.select_dtypes(include=[np.number]).columns
    if len(date_cols) > 0 and len(num_cols) > 0:
        df_sorted = df.sort_values(date_cols[0])
        fig = px.line(df_sorted, x=date_cols[0], y=num_cols[0],
                      title=f'Trend: {num_cols[0]} over {date_cols[0]}', template='plotly_dark')
        return fig
    return None

def generate_scatter_plot(df):
    num_cols = df.select_dtypes(include=[np.number]).columns
    if len(num_cols) >= 2:
        fig = px.scatter(df, x=num_cols[0], y=num_cols[1],
                         title=f'{num_cols[0]} vs {num_cols[1]}', template='plotly_dark')
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
        return "⚠️ Gemini API key missing. Please add GEMINI_API_KEY environment variable to enable AI insights."
    
    eda = generate_eda_report(df)
    context = f"""
Dataset Summary:
- {eda['total_rows']} rows, {eda['total_cols']} columns
- Numeric columns: {', '.join(eda['numeric_cols'][:5])}
- Categorical columns: {', '.join(eda['categorical_cols'][:5])}
- Date columns: {', '.join(eda['date_cols'])}
- Missing values: {eda['missing_values']}
"""
    prompt = f"""As a senior business analyst, provide 3-4 paragraphs of professional insights, trends, and actionable recommendations based on this dataset.

{context}

Focus on: key patterns, business opportunities, potential risks, and strategic recommendations.

Insights:"""
    try:
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"AI Error: {str(e)}. Please check your API key."

def answer_question(df, question):
    if not gemini_model:
        return "AI features disabled. Please configure GEMINI_API_KEY."
    
    context = f"Dataset has {len(df)} rows and columns: {', '.join(list(df.columns)[:10])}"
    prompt = f"Based on the dataset with {context}, answer: {question}\nAnswer concisely:"
    try:
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error: {str(e)}"

# ------------------------- Flask Routes -------------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        ext = file.filename.split('.')[-1].lower()
        if ext not in ['csv', 'xlsx', 'xls']:
            return jsonify({'error': 'Invalid file type. Please upload CSV or Excel file'}), 400
        
        uid = str(uuid.uuid4())[:8]
        filename = secure_filename(f"{uid}_{file.filename}")
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Read file
        try:
            if ext in ['xlsx', 'xls']:
                df = pd.read_excel(filepath)
            else:
                df = pd.read_csv(filepath, encoding='utf-8')
        except Exception as e:
            return jsonify({'error': f'Error reading file: {str(e)}'}), 400
        
        # Clean data
        cleaned_df = clean_data(df)
        cleaned_path = os.path.join(app.config['CLEANED_FOLDER'], f'{uid}_cleaned.csv')
        cleaned_df.to_csv(cleaned_path, index=False)
        
        session['cleaned_file'] = cleaned_path
        session['data_id'] = uid
        
        # Generate all outputs
        eda = generate_eda_report(cleaned_df)
        charts = generate_all_charts(cleaned_df)
        insights = generate_insights(cleaned_df)
        session['insights'] = insights
        
        # Prepare preview (convert to serializable format)
        preview = cleaned_df.head(100).fillna('').to_dict(orient='records')
        columns = list(cleaned_df.columns)
        
        # Prepare chart data for frontend
        chart_data = []
        for title, json_str, file_path in charts:
            chart_data.append({
                'title': title,
                'json': json_str,
                'file': os.path.basename(file_path)
            })
        
        kpis = {
            'total_rows': eda['total_rows'],
            'total_cols': eda['total_cols'],
            'missing_values': int(eda['missing_values']),
            'duplicates': int(eda['duplicates']),
            'numeric_cols_count': len(eda['numeric_cols'])
        }
        
        return jsonify({
            'success': True,
            'kpis': kpis,
            'preview': preview,
            'columns': columns,
            'charts': chart_data,
            'insights': insights,
            'data_id': uid
        })
        
    except Exception as e:
        print(f"Upload error: {traceback.format_exc()}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/ask', methods=['POST'])
def ask():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Invalid request'}), 400
            
        question = data.get('question', '')
        if not question:
            return jsonify({'error': 'No question provided'}), 400
        
        cleaned_file = session.get('cleaned_file')
        if not cleaned_file or not os.path.exists(cleaned_file):
            return jsonify({'error': 'No dataset loaded. Please upload a file first.'}), 400
        
        df = pd.read_csv(cleaned_file)
        answer = answer_question(df, question)
        return jsonify({'answer': answer})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/download-cleaned')
def download_cleaned():
    cleaned_file = session.get('cleaned_file')
    if not cleaned_file or not os.path.exists(cleaned_file):
        return "No cleaned file available. Please upload a dataset first.", 404
    return send_file(cleaned_file, as_attachment=True, download_name='cleaned_data.csv')

@app.route('/download-insights')
def download_insights():
    insights = session.get('insights', '')
    if not insights:
        return "No insights available. Please upload a dataset first.", 404
    
    from io import BytesIO
    output = BytesIO()
    output.write(insights.encode('utf-8'))
    output.seek(0)
    return send_file(output, as_attachment=True, download_name='ai_insights.txt', mimetype='text/plain')

@app.route('/download-chart/<filename>')
def download_chart(filename):
    safe_path = os.path.join(app.config['CHARTS_FOLDER'], filename)
    if os.path.exists(safe_path):
        return send_file(safe_path, as_attachment=True, download_name=filename)
    return "Chart not found", 404

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)

