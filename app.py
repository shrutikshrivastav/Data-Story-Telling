import os
import json
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime
import io
from scipy import stats

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size
app.config['UPLOAD_FOLDER'] = '/tmp/uploads'

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ==================== DATA CLEANING & PROCESSING ====================

def clean_data(df):
    """Comprehensive data cleaning with quality metrics"""
    
    initial_shape = df.shape
    metrics = {
        'initial_rows': initial_shape[0],
        'initial_columns': initial_shape[1],
        'duplicates_removed': 0,
        'missing_values_fixed': 0,
        'outliers_detected': 0,
        'rows_removed': 0,
        'missing_cells': df.isnull().sum().sum(),
    }
    
    df_clean = df.copy()
    
    # Remove completely empty rows
    empty_rows = df_clean.isnull().all(axis=1).sum()
    df_clean = df_clean.dropna(how='all')
    metrics['rows_removed'] += empty_rows
    
    # Remove duplicate rows
    duplicates = df_clean.duplicated().sum()
    df_clean = df_clean.drop_duplicates()
    metrics['duplicates_removed'] = duplicates
    
    # Handle missing values
    for col in df_clean.columns:
        if df_clean[col].dtype in ['float64', 'int64']:
            missing_count = df_clean[col].isnull().sum()
            if missing_count > 0:
                df_clean[col].fillna(df_clean[col].median(), inplace=True)
                metrics['missing_values_fixed'] += missing_count
        else:
            missing_count = df_clean[col].isnull().sum()
            if missing_count > 0:
                df_clean[col].fillna('Unknown', inplace=True)
                metrics['missing_values_fixed'] += missing_count
    
    # Normalize column names
    df_clean.columns = df_clean.columns.str.lower().str.replace(' ', '_').str.replace('[^a-z0-9_]', '', regex=True)
    
    # Detect and handle data types
    type_mapping = {}
    for col in df_clean.columns:
        try:
            # Try to detect date columns
            if 'date' in col or 'time' in col:
                df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce')
                type_mapping[col] = 'datetime'
            # Try to convert to numeric
            elif df_clean[col].dtype == 'object':
                numeric_test = pd.to_numeric(df_clean[col], errors='coerce')
                if numeric_test.notna().sum() / len(df_clean) > 0.8:
                    df_clean[col] = numeric_test
                    type_mapping[col] = 'numeric'
                else:
                    type_mapping[col] = 'categorical'
            else:
                type_mapping[col] = 'numeric' if pd.api.types.is_numeric_dtype(df_clean[col]) else 'categorical'
        except:
            type_mapping[col] = 'categorical'
    
    metrics['type_mapping'] = type_mapping
    metrics['final_rows'] = len(df_clean)
    metrics['final_columns'] = len(df_clean.columns)
    
    return df_clean, metrics

def detect_outliers(df):
    """Detect outliers using IQR method"""
    outlier_info = {}
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
        
        if len(outliers) > 0:
            outlier_info[col] = {
                'count': len(outliers),
                'percentage': round((len(outliers) / len(df)) * 100, 2),
                'bounds': {
                    'lower': round(lower_bound, 2),
                    'upper': round(upper_bound, 2)
                }
            }
    
    return outlier_info

def calculate_quality_score(df, metrics, outliers):
    """Calculate overall dataset quality score"""
    score = 100
    
    # Penalty for missing values
    if metrics['missing_values_fixed'] > 0:
        missing_ratio = (metrics['missing_values_fixed'] / (df.shape[0] * df.shape[1])) * 100
        score -= min(missing_ratio * 0.1, 15)
    
    # Penalty for duplicates
    if metrics['duplicates_removed'] > 0:
        dup_ratio = (metrics['duplicates_removed'] / metrics['initial_rows']) * 100
        score -= min(dup_ratio * 0.1, 10)
    
    # Penalty for outliers
    total_outliers = sum(v['count'] for v in outliers.values())
    if total_outliers > 0:
        outlier_ratio = (total_outliers / (df.shape[0] * len(df.select_dtypes(include=[np.number]).columns))) * 100
        score -= min(outlier_ratio * 0.05, 10)
    
    return max(score, 10)

# ==================== VISUALIZATION GENERATION ====================

def generate_visualizations(df):
    """Generate comprehensive visualizations"""
    visualizations = {}
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    
    # 1. Numeric Distribution (Histogram)
    if len(numeric_cols) > 0:
        primary_numeric = numeric_cols[0]
        fig = px.histogram(
            df,
            x=primary_numeric,
            nbins=30,
            title=f"Distribution: {primary_numeric.replace('_', ' ').title()}",
            labels={primary_numeric: primary_numeric.replace('_', ' ').title(), 'count': 'Frequency'},
            color_discrete_sequence=['#00d9ff']
        )
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(30,30,50,0.3)',
            font=dict(family='Segoe UI, sans-serif', size=12, color='#e0e0e0'),
            hovermode='x unified',
            showlegend=False,
            margin=dict(l=40, r=40, t=50, b=40)
        )
        visualizations['distribution'] = fig.to_html(include_plotlyjs=False, div_id='dist-chart')
    
    # 2. Top Categories (Bar Chart)
    if len(categorical_cols) > 0:
        primary_cat = categorical_cols[0]
        top_cats = df[primary_cat].value_counts().head(10)
        fig = go.Figure(data=[
            go.Bar(
                x=top_cats.values,
                y=top_cats.index,
                orientation='h',
                marker=dict(
                    color=top_cats.values,
                    colorscale='Viridis',
                    line=dict(color='rgba(0, 217, 255, 0.5)', width=1)
                ),
                text=top_cats.values,
                textposition='outside',
                hovertemplate='%{y}<br>Count: %{x}<extra></extra>'
            )
        ])
        fig.update_layout(
            title=f"Top Categories: {primary_cat.replace('_', ' ').title()}",
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(30,30,50,0.3)',
            font=dict(family='Segoe UI, sans-serif', size=12, color='#e0e0e0'),
            showlegend=False,
            margin=dict(l=150, r=40, t=50, b=40),
            xaxis=dict(gridcolor='rgba(255,255,255,0.1)')
        )
        visualizations['categories'] = fig.to_html(include_plotlyjs=False, div_id='cat-chart')
    
    # 3. Correlation Heatmap
    if len(numeric_cols) > 1:
        corr_matrix = df[numeric_cols].corr()
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=np.round(corr_matrix.values, 2),
            texttemplate='%{text:.2f}',
            textfont={"size": 10},
            colorbar=dict(thickness=15, len=0.7)
        ))
        fig.update_layout(
            title="Correlation Matrix",
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(30,30,50,0.3)',
            font=dict(family='Segoe UI, sans-serif', size=11, color='#e0e0e0'),
            margin=dict(l=100, r=40, t=50, b=100)
        )
        visualizations['correlation'] = fig.to_html(include_plotlyjs=False, div_id='corr-chart')
    
    # 4. Numeric Scatter (if multiple numeric columns)
    if len(numeric_cols) >= 2:
        fig = px.scatter(
            df,
            x=numeric_cols[0],
            y=numeric_cols[1],
            title=f"{numeric_cols[0].replace('_', ' ').title()} vs {numeric_cols[1].replace('_', ' ').title()}",
            color_discrete_sequence=['#00d9ff']
        )
        fig.update_traces(marker=dict(size=6, opacity=0.6, line=dict(width=0.5, color='rgba(0, 217, 255, 0.8)')))
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(30,30,50,0.3)',
            font=dict(family='Segoe UI, sans-serif', size=12, color='#e0e0e0'),
            hovermode='closest',
            showlegend=False,
            margin=dict(l=40, r=40, t=50, b=40)
        )
        visualizations['scatter'] = fig.to_html(include_plotlyjs=False, div_id='scatter-chart')
    
    # 5. Summary Statistics
    if len(numeric_cols) > 0:
        summary_stats = []
        for col in numeric_cols[:5]:  # Limit to first 5 for performance
            summary_stats.append({
                'column': col.replace('_', ' ').title(),
                'mean': round(df[col].mean(), 2),
                'median': round(df[col].median(), 2),
                'std': round(df[col].std(), 2),
                'min': round(df[col].min(), 2),
                'max': round(df[col].max(), 2)
            })
        visualizations['summary_stats'] = summary_stats
    
    return visualizations

# ==================== NARRATIVE GENERATION ====================

def generate_narrative(df, metrics, visualizations):
    """Generate professional business insights and narrative"""
    
    narrative = {
        'executive_summary': '',
        'data_quality': '',
        'key_insights': [],
        'trends': [],
        'recommendations': []
    }
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Executive Summary
    narrative['executive_summary'] = (
        f"Dataset Analysis Report: {metrics['final_rows']:,} records across {metrics['final_columns']} dimensions. "
        f"After comprehensive data cleansing, the dataset demonstrates strong quality with consolidated structure "
        f"ready for analysis and strategic decision-making."
    )
    
    # Data Quality Narrative
    if metrics['duplicates_removed'] > 0:
        narrative['data_quality'] += f"• Removed {metrics['duplicates_removed']} duplicate records to ensure data integrity\n"
    if metrics['missing_values_fixed'] > 0:
        narrative['data_quality'] += f"• Resolved {metrics['missing_values_fixed']} missing values using statistical imputation\n"
    narrative['data_quality'] += f"• Standardized {metrics['final_columns']} column names for consistency\n"
    narrative['data_quality'] += f"• Detected and classified {len(df.select_dtypes(include=[np.number]).columns)} numeric and {len(categorical_cols)} categorical variables\n"
    
    if not narrative['data_quality']:
        narrative['data_quality'] = "• Dataset integrity verified with minimal data quality issues\n• All records validated and standardized\n"
    
    # Key Insights from Numeric Data
    if len(numeric_cols) > 0:
        for col in numeric_cols[:3]:
            mean_val = df[col].mean()
            max_val = df[col].max()
            min_val = df[col].min()
            std_val = df[col].std()
            
            col_name = col.replace('_', ' ').title()
            
            if std_val > 0:
                cv = (std_val / mean_val) * 100 if mean_val != 0 else 0
                if cv > 50:
                    variability = "demonstrates high variability"
                elif cv > 25:
                    variability = "shows moderate variation"
                else:
                    variability = "remains relatively stable"
            else:
                variability = "remains constant"
            
            insight = (
                f"{col_name} {variability}, ranging from {min_val:.2f} to {max_val:.2f} "
                f"with an average of {mean_val:.2f}"
            )
            narrative['key_insights'].append(insight)
    
    # Category Insights
    if len(categorical_cols) > 0:
        for col in categorical_cols[:2]:
            top_cat = df[col].value_counts().idxmax()
            top_count = df[col].value_counts().max()
            top_pct = (top_count / len(df)) * 100
            
            col_name = col.replace('_', ' ').title()
            narrative['key_insights'].append(
                f"{col_name} distribution is led by '{top_cat}' category, "
                f"representing {top_pct:.1f}% of all records"
            )
    
    # Trend Analysis
    if len(numeric_cols) >= 2:
        corr_matrix = df[numeric_cols].corr()
        strong_corr = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.7:
                    col1 = corr_matrix.columns[i].replace('_', ' ').title()
                    col2 = corr_matrix.columns[j].replace('_', ' ').title()
                    direction = "strongly positive" if corr_val > 0 else "strong inverse"
                    strong_corr.append(
                        f"{col1} and {col2} exhibit a {direction} relationship ({abs(corr_val):.2f})"
                    )
        
        if strong_corr:
            narrative['trends'].extend(strong_corr)
    
    # Statistical Insights
    if len(numeric_cols) > 0:
        for col in numeric_cols[:2]:
            skewness = stats.skew(df[col].dropna())
            if abs(skewness) > 1:
                skew_type = "right-skewed" if skewness > 0 else "left-skewed"
                narrative['trends'].append(
                    f"{col.replace('_', ' ').title()} distribution is {skew_type}, "
                    f"indicating potential outliers or concentration in specific ranges"
                )
    
    # Recommendations
    if len(numeric_cols) > 0:
        top_col = df[numeric_cols[0]]
        percentile_75 = top_col.quantile(0.75)
        percentile_25 = top_col.quantile(0.25)
        
        narrative['recommendations'].append(
            f"Focus optimization efforts on the upper quartile of "
            f"{numeric_cols[0].replace('_', ' ').title()} "
            f"(values above {percentile_75:.2f}) for maximum impact"
        )
    
    narrative['recommendations'].append(
        "Continue monitoring data quality metrics and implement regular validation cycles"
    )
    narrative['recommendations'].append(
        "Leverage identified correlations to develop predictive models for strategic planning"
    )
    
    return narrative

# ==================== FLASK ROUTES ====================

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        # Check if file is in request
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Validate file type
        if not (file.filename.endswith('.csv') or file.filename.endswith('.xlsx')):
            return jsonify({'error': 'Only CSV and Excel files are supported'}), 400
        
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # ==================== PHASE 1: DATA CLEANING ====================
        
        # Read file
        if filename.endswith('.csv'):
            df = pd.read_csv(filepath)
        else:
            df = pd.read_excel(filepath)
        
        # Store original data
        original_shape = df.shape
        original_sample = df.head(3).to_dict('records')
        
        # Clean data
        df_clean, metrics = clean_data(df)
        
        # Detect outliers
        outliers = detect_outliers(df_clean)
        metrics['outliers'] = outliers
        
        # Calculate quality score
        quality_score = calculate_quality_score(df_clean, metrics, outliers)
        metrics['quality_score'] = quality_score
        
        # Cleaned data sample
        cleaned_sample = df_clean.head(3).to_dict('records')
        
        # ==================== PHASE 2: VISUALIZATIONS ====================
        
        visualizations = generate_visualizations(df_clean)
        
        # ==================== PHASE 3: NARRATIVE ====================
        
        narrative = generate_narrative(df_clean, metrics, visualizations)
        
        # Prepare response
        response = {
            'success': True,
            'phase1': {
                'before': {
                    'rows': original_shape[0],
                    'columns': original_shape[1],
                    'sample': original_sample
                },
                'after': {
                    'rows': metrics['final_rows'],
                    'columns': metrics['final_columns'],
                    'sample': cleaned_sample
                },
                'metrics': {
                    'duplicates_removed': metrics['duplicates_removed'],
                    'missing_values_fixed': metrics['missing_values_fixed'],
                    'outliers_detected': len(outliers),
                    'rows_removed': metrics['rows_removed'],
                    'quality_score': round(quality_score, 1)
                },
                'outliers_detail': outliers
            },
            'phase2': visualizations,
            'phase3': narrative
        }
        
        # Cleanup
        os.remove(filepath)
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': f'Processing error: {str(e)}'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
