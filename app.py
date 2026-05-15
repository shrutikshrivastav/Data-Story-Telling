import os
import io
import base64
import json
from datetime import datetime
import pandas as pd
import numpy as np
import plotly
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from flask import Flask, render_template, request, jsonify
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

def normalize_columns(df):
    """Normalize column names"""
    df.columns = [col.strip().lower().replace(' ', '_').replace('-', '_') for col in df.columns]
    return df

def detect_date_columns(df):
    """Detect and convert date columns"""
    date_columns = []
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                pd.to_datetime(df[col], errors='raise')
                date_columns.append(col)
            except:
                pass
    return date_columns

def clean_data(df):
    """Clean the dataset and return cleaned dataframe with stats"""
    stats = {
        'original_rows': len(df),
        'original_columns': len(df.columns),
        'duplicates_removed': 0,
        'missing_values_fixed': 0,
        'outliers_detected': 0,
        'data_quality_score': 0
    }
    
    # Normalize column names
    df = normalize_columns(df)
    
    # Remove duplicates
    duplicates = df.duplicated().sum()
    df = df.drop_duplicates()
    stats['duplicates_removed'] = duplicates
    
    # Remove completely empty rows
    df = df.dropna(how='all')
    
    # Handle missing values
    missing_before = df.isnull().sum().sum()
    
    for col in df.columns:
        if df[col].dtype in ['int64', 'float64']:
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown')
    
    stats['missing_values_fixed'] = missing_before
    
    # Detect and convert date columns
    date_columns = detect_date_columns(df)
    for col in date_columns:
        try:
            df[col] = pd.to_datetime(df[col])
        except:
            pass
    
    # Fix datatypes
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                df[col] = pd.to_numeric(df[col])
            except:
                pass
    
    # Detect outliers
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    outliers_count = 0
    for col in numeric_columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = df[(df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))].shape[0]
        outliers_count += outliers
    stats['outliers_detected'] = outliers_count
    
    # Calculate data quality score
    total_cells = stats['original_rows'] * stats['original_columns']
    issues = stats['duplicates_removed'] + stats['missing_values_fixed'] + stats['outliers_detected']
    score = max(0, 100 - (issues / total_cells * 100)) if total_cells > 0 else 100
    stats['data_quality_score'] = round(min(100, score), 1)
    
    return df, stats

def identify_columns(df):
    """Identify numeric and categorical columns"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    date_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    return numeric_cols, categorical_cols, date_cols

def generate_visualizations(df):
    """Generate interactive Plotly charts"""
    numeric_cols, categorical_cols, date_cols = identify_columns(df)
    charts = []
    
    # Bar Chart - Top categories
    if categorical_cols and numeric_cols:
        cat_col = categorical_cols[0]
        num_col = numeric_cols[0]
        grouped = df.groupby(cat_col)[num_col].sum().sort_values(ascending=False).head(10)
        fig = px.bar(x=grouped.index, y=grouped.values, 
                     title=f'Top 10 {cat_col} by {num_col}',
                     template='plotly_dark',
                     color=grouped.values,
                     color_continuous_scale='viridis')
        fig.update_layout(showlegend=False)
        charts.append({
            'id': 'bar_chart',
            'title': 'Bar Chart Analysis',
            'chart': json.loads(fig.to_json())
        })
    
    # Line Chart - if date column exists
    if date_cols and numeric_cols:
        date_col = date_cols[0]
        num_col = numeric_cols[0]
        df_sorted = df.sort_values(date_col)
        fig = px.line(df_sorted, x=date_col, y=num_col, 
                      title=f'{num_col} Trend Over Time',
                      template='plotly_dark')
        fig.update_traces(line_color='#00ff88', line_width=2)
        charts.append({
            'id': 'line_chart',
            'title': 'Trend Analysis',
            'chart': json.loads(fig.to_json())
        })
    
    # Pie Chart
    if categorical_cols:
        cat_col = categorical_cols[0]
        value_counts = df[cat_col].value_counts().head(8)
        fig = px.pie(values=value_counts.values, names=value_counts.index,
                     title=f'{cat_col} Distribution',
                     template='plotly_dark',
                     hole=0.3)
        fig.update_traces(marker=dict(colors=px.colors.sequential.Viridis))
        charts.append({
            'id': 'pie_chart',
            'title': 'Distribution Analysis',
            'chart': json.loads(fig.to_json())
        })
    
    # Histogram
    if numeric_cols:
        num_col = numeric_cols[0]
        fig = px.histogram(df, x=num_col, nbins=30,
                          title=f'{num_col} Distribution',
                          template='plotly_dark',
                          color_discrete_sequence=['#00ff88'])
        charts.append({
            'id': 'histogram',
            'title': 'Histogram Analysis',
            'chart': json.loads(fig.to_json())
        })
    
    # Scatter Plot
    if len(numeric_cols) >= 2:
        fig = px.scatter(df, x=numeric_cols[0], y=numeric_cols[1],
                        title=f'{numeric_cols[0]} vs {numeric_cols[1]}',
                        template='plotly_dark',
                        color_continuous_scale='viridis',
                        opacity=0.6)
        fig.update_traces(marker=dict(color='#00ff88', size=8))
        charts.append({
            'id': 'scatter_plot',
            'title': 'Scatter Analysis',
            'chart': json.loads(fig.to_json())
        })
    
    # Correlation Heatmap
    if len(numeric_cols) >= 2:
        corr_matrix = df[numeric_cols].corr()
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='Viridis',
            text=np.round(corr_matrix.values, 2),
            texttemplate='%{text}',
            textfont={"size": 10}
        ))
        fig.update_layout(
            title='Correlation Heatmap',
            template='plotly_dark'
        )
        charts.append({
            'id': 'heatmap',
            'title': 'Correlation Analysis',
            'chart': json.loads(fig.to_json())
        })
    
    return charts

def generate_narrative(df, stats):
    """Generate business insights and storytelling"""
    numeric_cols, categorical_cols, date_cols = identify_columns(df)
    insights = []
    
    # Dataset overview
    insights.append({
        'type': 'overview',
        'title': '📊 Dataset Overview',
        'content': f'The dataset contains {stats["original_rows"]} records across {len(df.columns)} variables, with a data quality score of {stats["data_quality_score"]}%. '
                  f'After cleaning, we removed {stats["duplicates_removed"]} duplicate entries and fixed {stats["missing_values_fixed"]} missing values.'
    })
    
    # Numeric insights
    if numeric_cols:
        for col in numeric_cols[:3]:
            mean_val = df[col].mean()
            median_val = df[col].median()
            std_val = df[col].std()
            max_val = df[col].max()
            min_val = df[col].min()
            
            trend = "stable"
            if std_val > mean_val * 0.5:
                trend = "highly variable"
            elif std_val < mean_val * 0.1:
                trend = "consistent"
            
            insights.append({
                'type': 'numeric',
                'title': f'📈 {col.title()} Analysis',
                'content': f'The average {col} is {mean_val:.2f} with a standard deviation of {std_val:.2f}, indicating {trend} behavior. '
                          f'Values range from {min_val:.2f} to {max_val:.2f}, with the median at {median_val:.2f}. '
                          f'{"This suggests significant variation in the data." if trend == "highly variable" else "The data shows moderate stability."}'
            })
    
    # Categorical insights
    if categorical_cols:
        cat_col = categorical_cols[0]
        top_cat = df[cat_col].value_counts().head(3)
        total_cats = df[cat_col].nunique()
        
        insights.append({
            'type': 'categorical',
            'title': f'🏷️ {cat_col.title()} Breakdown',
            'content': f'The top category is "{top_cat.index[0]}" representing {top_cat.values[0]} records ({(top_cat.values[0]/len(df)*100):.1f}% of total). '
                      f'There are {total_cats} unique categories in this field. '
                      f'The top 3 categories account for {sum(top_cat.values[:3])/len(df)*100:.1f}% of all data.'
        })
    
    # Correlation insights
    if len(numeric_cols) >= 2:
        corr_matrix = df[numeric_cols].corr()
        # Find highest correlation
        max_corr = 0
        max_pair = ('', '')
        for i in range(len(numeric_cols)):
            for j in range(i+1, len(numeric_cols)):
                if abs(corr_matrix.iloc[i, j]) > max_corr:
                    max_corr = abs(corr_matrix.iloc[i, j])
                    max_pair = (numeric_cols[i], numeric_cols[j])
        
        if max_corr > 0.3:
            relationship = "strong positive" if corr_matrix.loc[max_pair[0], max_pair[1]] > 0 else "strong negative"
            insights.append({
                'type': 'correlation',
                'title': '🔗 Key Relationship Found',
                'content': f'The strongest correlation exists between {max_pair[0]} and {max_pair[1]} ({corr_matrix.loc[max_pair[0], max_pair[1]]:.2f}), '
                          f'indicating a {relationship} relationship. '
                          f'This suggests that changes in {max_pair[0]} significantly impact {max_pair[1]}.'
            })
    
    # Time-based insights
    if date_cols and numeric_cols:
        date_col = date_cols[0]
        num_col = numeric_cols[0]
        df_sorted = df.sort_values(date_col)
        
        if len(df_sorted) >= 3:
            first_quarter = df_sorted[num_col].iloc[:len(df_sorted)//4].mean()
            last_quarter = df_sorted[num_col].iloc[-len(df_sorted)//4:].mean()
            
            if last_quarter > first_quarter:
                growth = ((last_quarter - first_quarter) / first_quarter) * 100
                insights.append({
                    'type': 'trend',
                    'title': '📅 Growth Trend Detected',
                    'content': f'Comparing the first and last quarters, {num_col} shows a growth of {growth:.1f}%. '
                              f'The average increased from {first_quarter:.2f} to {last_quarter:.2f}. '
                              f'This upward trajectory suggests positive momentum in the business.'
                })
            elif last_quarter < first_quarter:
                decline = ((first_quarter - last_quarter) / first_quarter) * 100
                insights.append({
                    'type': 'trend',
                    'title': '📉 Decline Pattern Observed',
                    'content': f'There is a concerning decline of {decline:.1f}% in {num_col} from {first_quarter:.2f} to {last_quarter:.2f}. '
                              f'This requires immediate attention and further investigation.'
                })
    
    # Statistical summary
    insights.append({
        'type': 'summary',
        'title': '🎯 Key Takeaways',
        'content': f'The data reveals that {categorical_cols[0] if categorical_cols else "the primary variable"} is the main driver of {numeric_cols[0] if numeric_cols else "business performance"}. '
                  f'With a quality score of {stats["data_quality_score"]}%, the dataset is {"excellent" if stats["data_quality_score"] > 90 else "good" if stats["data_quality_score"] > 70 else "moderate"}. '
                  f'{"The presence of outliers suggests the need for further investigation of extreme values." if stats["outliers_detected"] > 0 else ""}'
    })
    
    return insights

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    try:
        # Read file
        if file.filename.endswith('.csv'):
            df = pd.read_csv(file)
        elif file.filename.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(file)
        else:
            return jsonify({'error': 'Unsupported file format'}), 400
        
        # Store original data for before/after comparison
        original_df = df.copy()
        original_stats = {
            'rows': len(df),
            'columns': len(df.columns),
            'missing': int(df.isnull().sum().sum()),
            'duplicates': int(df.duplicated().sum()),
            'column_names': list(df.columns)
        }
        
        # Clean data
        cleaned_df, cleaning_stats = clean_data(df)
        
        # Generate visualizations
        charts = generate_visualizations(cleaned_df)
        
        # Generate narrative
        narrative = generate_narrative(cleaned_df, cleaning_stats)
        
        # Prepare response
        response = {
            'original_data': original_stats,
            'cleaning_stats': cleaning_stats,
            'cleaned_columns': list(cleaned_df.columns),
            'numeric_cols': len(cleaned_df.select_dtypes(include=[np.number]).columns),
            'categorical_cols': len(cleaned_df.select_dtypes(include=['object', 'category']).columns),
            'charts': charts,
            'narrative': narrative,
            'preview': cleaned_df.head(10).to_dict('records'),
            'filename': file.filename
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
