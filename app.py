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

def convert_to_native_types(obj):
    """Convert numpy types to native Python types for JSON serialization"""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Timestamp):
        return str(obj)
    elif isinstance(obj, dict):
        return {key: convert_to_native_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_native_types(item) for item in obj)
    return obj

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
        'original_rows': int(len(df)),
        'original_columns': int(len(df.columns)),
        'duplicates_removed': int(0),
        'missing_values_fixed': int(0),
        'outliers_detected': int(0),
        'data_quality_score': float(0)
    }
    
    # Normalize column names
    df = normalize_columns(df)
    
    # Remove duplicates
    duplicates = int(df.duplicated().sum())
    df = df.drop_duplicates()
    stats['duplicates_removed'] = duplicates
    
    # Remove completely empty rows
    df = df.dropna(how='all')
    
    # Handle missing values
    missing_before = int(df.isnull().sum().sum())
    
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
    
    # Detect outliers using IQR method
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    outliers_count = 0
    for col in numeric_columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = int(df[(df[col] < lower_bound) | (df[col] > upper_bound)].shape[0])
        outliers_count += outliers
    stats['outliers_detected'] = outliers_count
    
    # Calculate data quality score
    total_cells = int(stats['original_rows'] * stats['original_columns'])
    issues = int(stats['duplicates_removed'] + stats['missing_values_fixed'] + stats['outliers_detected'])
    if total_cells > 0:
        score = max(0, 100 - (issues / total_cells * 100))
    else:
        score = 100
    stats['data_quality_score'] = float(round(min(100, score), 1))
    
    # Convert all stats to native Python types
    stats = convert_to_native_types(stats)
    
    return df, stats

def identify_columns(df):
    """Identify numeric and categorical columns"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    date_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    return numeric_cols, categorical_cols, date_cols

def generate_visualizations(df):
    """Generate interactive Plotly charts with professional styling"""
    numeric_cols, categorical_cols, date_cols = identify_columns(df)
    charts = []
    
    # Professional color palette
    colors = ['#3b82f6', '#6366f1', '#8b5cf6', '#ec4899', '#f43f5e', 
              '#f97316', '#eab308', '#22c55e', '#14b8a6', '#06b6d4']
    
    # Common layout settings for professional look
    common_layout = {
        'template': 'plotly_white',
        'paper_bgcolor': 'white',
        'plot_bgcolor': '#fafbfc',
        'font': {'family': 'Plus Jakarta Sans, sans-serif', 'color': '#334155', 'size': 12},
        'margin': {'t': 50, 'l': 60, 'r': 20, 'b': 60},
        'xaxis': {'gridcolor': '#f1f5f9', 'linecolor': '#e2e8f0', 'showgrid': True},
        'yaxis': {'gridcolor': '#f1f5f9', 'linecolor': '#e2e8f0', 'showgrid': True},
        'hovermode': 'x unified',
        'legend': {'orientation': 'h', 'yanchor': 'bottom', 'y': 1.02, 'xanchor': 'right', 'x': 1}
    }
    
    # Bar Chart - Top categories
    if categorical_cols and numeric_cols:
        try:
            cat_col = categorical_cols[0]
            num_col = numeric_cols[0]
            grouped = df.groupby(cat_col)[num_col].sum().sort_values(ascending=False).head(10)
            
            fig = go.Figure(data=[
                go.Bar(
                    x=grouped.index.astype(str),
                    y=grouped.values,
                    marker_color=colors[:len(grouped)],
                    marker_line_color='white',
                    marker_line_width=1,
                    text=grouped.values.round(2),
                    textposition='outside',
                    textfont=dict(size=11, color='#334155'),
                    hovertemplate='%{x}<br>%{y:,.2f}<extra></extra>'
                )
            ])
            
            layout = common_layout.copy()
            layout['title'] = dict(text=f'Top 10 {cat_col.title()} by {num_col.title()}', 
                                  font=dict(size=16, color='#1e293b', family='Plus Jakarta Sans, sans-serif'))
            layout['xaxis']['tickangle'] = -45
            
            fig.update_layout(layout)
            charts.append({
                'id': 'bar_chart',
                'title': 'Bar Chart Analysis',
                'chart': json.loads(fig.to_json())
            })
        except Exception as e:
            print(f"Error creating bar chart: {e}")
    
    # Line Chart - if date column exists
    if date_cols and numeric_cols:
        try:
            date_col = date_cols[0]
            num_col = numeric_cols[0]
            df_sorted = df.sort_values(date_col).copy()
            
            fig = go.Figure(data=[
                go.Scatter(
                    x=df_sorted[date_col],
                    y=df_sorted[num_col],
                    mode='lines+markers',
                    line=dict(color='#3b82f6', width=2.5),
                    marker=dict(size=6, color='#3b82f6'),
                    fill='tozeroy',
                    fillcolor='rgba(59, 130, 246, 0.1)',
                    hovertemplate='%{x}<br>%{y:,.2f}<extra></extra>'
                )
            ])
            
            layout = common_layout.copy()
            layout['title'] = dict(text=f'{num_col.title()} Trend Over Time', 
                                  font=dict(size=16, color='#1e293b', family='Plus Jakarta Sans, sans-serif'))
            layout['showlegend'] = False
            
            fig.update_layout(layout)
            charts.append({
                'id': 'line_chart',
                'title': 'Trend Analysis',
                'chart': json.loads(fig.to_json())
            })
        except Exception as e:
            print(f"Error creating line chart: {e}")
    
    # Pie Chart
    if categorical_cols:
        try:
            cat_col = categorical_cols[0]
            value_counts = df[cat_col].value_counts().head(8)
            
            fig = go.Figure(data=[
                go.Pie(
                    labels=value_counts.index.astype(str),
                    values=value_counts.values,
                    hole=0.4,
                    marker=dict(colors=colors[:len(value_counts)], line=dict(color='white', width=2)),
                    textinfo='label+percent',
                    textfont=dict(size=11, color='#334155'),
                    hovertemplate='%{label}<br>%{value:,.0f} records<br>%{percent}<extra></extra>'
                )
            ])
            
            layout = common_layout.copy()
            layout['title'] = dict(text=f'{cat_col.title()} Distribution', 
                                  font=dict(size=16, color='#1e293b', family='Plus Jakarta Sans, sans-serif'))
            layout.pop('xaxis', None)
            layout.pop('yaxis', None)
            layout.pop('hovermode', None)
            
            fig.update_layout(layout)
            charts.append({
                'id': 'pie_chart',
                'title': 'Distribution Analysis',
                'chart': json.loads(fig.to_json())
            })
        except Exception as e:
            print(f"Error creating pie chart: {e}")
    
    # Histogram
    if numeric_cols:
        try:
            num_col = numeric_cols[0]
            
            fig = go.Figure(data=[
                go.Histogram(
                    x=df[num_col],
                    nbinsx=30,
                    marker_color='#3b82f6',
                    marker_line_color='white',
                    marker_line_width=1,
                    hovertemplate='Range: %{x}<br>Count: %{y}<extra></extra>'
                )
            ])
            
            layout = common_layout.copy()
            layout['title'] = dict(text=f'{num_col.title()} Distribution', 
                                  font=dict(size=16, color='#1e293b', family='Plus Jakarta Sans, sans-serif'))
            layout['showlegend'] = False
            
            fig.update_layout(layout)
            charts.append({
                'id': 'histogram',
                'title': 'Histogram Analysis',
                'chart': json.loads(fig.to_json())
            })
        except Exception as e:
            print(f"Error creating histogram: {e}")
    
    # Scatter Plot
    if len(numeric_cols) >= 2:
        try:
            fig = go.Figure(data=[
                go.Scatter(
                    x=df[numeric_cols[0]],
                    y=df[numeric_cols[1]],
                    mode='markers',
                    marker=dict(
                        size=8,
                        color='#3b82f6',
                        opacity=0.6,
                        line=dict(color='white', width=1)
                    ),
                    hovertemplate=f'{numeric_cols[0]}: %{{x:,.2f}}<br>{numeric_cols[1]}: %{{y:,.2f}}<extra></extra>'
                )
            ])
            
            layout = common_layout.copy()
            layout['title'] = dict(text=f'{numeric_cols[0].title()} vs {numeric_cols[1].title()}', 
                                  font=dict(size=16, color='#1e293b', family='Plus Jakarta Sans, sans-serif'))
            layout['showlegend'] = False
            
            fig.update_layout(layout)
            charts.append({
                'id': 'scatter_plot',
                'title': 'Scatter Analysis',
                'chart': json.loads(fig.to_json())
            })
        except Exception as e:
            print(f"Error creating scatter plot: {e}")
    
    # Correlation Heatmap
    if len(numeric_cols) >= 2:
        try:
            corr_matrix = df[numeric_cols].corr()
            
            fig = go.Figure(data=[
                go.Heatmap(
                    z=corr_matrix.values,
                    x=corr_matrix.columns.tolist(),
                    y=corr_matrix.columns.tolist(),
                    colorscale=[
                        [0, '#ef4444'],
                        [0.5, '#f8fafc'],
                        [1, '#3b82f6']
                    ],
                    zmin=-1,
                    zmax=1,
                    text=np.round(corr_matrix.values, 2),
                    texttemplate='%{text}',
                    textfont={"size": 11, "color": "#1e293b", "family": "Plus Jakarta Sans, sans-serif"},
                    hoverongaps=False,
                    hovertemplate='%{x} vs %{y}<br>Correlation: %{z:.2f}<extra></extra>'
                )
            ])
            
            layout = common_layout.copy()
            layout['title'] = dict(text='Correlation Heatmap', 
                                  font=dict(size=16, color='#1e293b', family='Plus Jakarta Sans, sans-serif'))
            layout['xaxis']['tickangle'] = -45
            layout['height'] = 500
            
            fig.update_layout(layout)
            charts.append({
                'id': 'heatmap',
                'title': 'Correlation Analysis',
                'chart': json.loads(fig.to_json())
            })
        except Exception as e:
            print(f"Error creating heatmap: {e}")
    
    return charts

def generate_narrative(df, stats):
    """Generate business insights and storytelling"""
    numeric_cols, categorical_cols, date_cols = identify_columns(df)
    insights = []
    
    # Ensure all stats values are native Python types
    original_rows = int(stats.get('original_rows', len(df)))
    duplicates_removed = int(stats.get('duplicates_removed', 0))
    missing_fixed = int(stats.get('missing_values_fixed', 0))
    quality_score = float(stats.get('data_quality_score', 0))
    outliers = int(stats.get('outliers_detected', 0))
    
    # Dataset overview
    insights.append({
        'type': 'overview',
        'title': 'Dataset Overview',
        'content': f'The dataset contains {original_rows} records across {len(df.columns)} variables, with a data quality score of {quality_score}%. '
                  f'After cleaning, we removed {duplicates_removed} duplicate entries and fixed {missing_fixed} missing values.'
    })
    
    # Numeric insights
    if numeric_cols:
        for idx, col in enumerate(numeric_cols[:3]):
            try:
                mean_val = float(df[col].mean())
                median_val = float(df[col].median())
                std_val = float(df[col].std())
                max_val = float(df[col].max())
                min_val = float(df[col].min())
                
                trend = "stable"
                if mean_val != 0 and std_val > mean_val * 0.5:
                    trend = "highly variable"
                elif mean_val != 0 and std_val < mean_val * 0.1:
                    trend = "consistent"
                
                insights.append({
                    'type': 'numeric',
                    'title': f'{col.title()} Analysis',
                    'content': f'The average {col} is {mean_val:.2f} with a standard deviation of {std_val:.2f}, indicating {trend} behavior. '
                              f'Values range from {min_val:.2f} to {max_val:.2f}, with the median at {median_val:.2f}. '
                              f'{"This suggests significant variation in the data." if trend == "highly variable" else "The data shows moderate stability."}'
                })
            except Exception as e:
                print(f"Error analyzing {col}: {e}")
    
    # Categorical insights
    if categorical_cols:
        try:
            cat_col = categorical_cols[0]
            top_cat = df[cat_col].value_counts().head(3)
            total_cats = int(df[cat_col].nunique())
            total_rows = len(df)
            
            if len(top_cat) > 0 and total_rows > 0:
                top_percentage = float((top_cat.values[0] / total_rows) * 100)
                top3_percentage = float((sum(top_cat.values[:3]) / total_rows) * 100)
                
                insights.append({
                    'type': 'categorical',
                    'title': f'{cat_col.title()} Breakdown',
                    'content': f'The top category is "{str(top_cat.index[0])}" representing {int(top_cat.values[0])} records ({top_percentage:.1f}% of total). '
                              f'There are {total_cats} unique categories in this field. '
                              f'The top 3 categories account for {top3_percentage:.1f}% of all data.'
                })
        except Exception as e:
            print(f"Error analyzing categories: {e}")
    
    # Correlation insights
    if len(numeric_cols) >= 2:
        try:
            corr_matrix = df[numeric_cols].corr()
            max_corr = 0.0
            max_pair = ('', '')
            
            for i in range(len(numeric_cols)):
                for j in range(i+1, len(numeric_cols)):
                    current_corr = abs(float(corr_matrix.iloc[i, j]))
                    if current_corr > max_corr:
                        max_corr = current_corr
                        max_pair = (numeric_cols[i], numeric_cols[j])
            
            if max_corr > 0.3:
                corr_value = float(corr_matrix.loc[max_pair[0], max_pair[1]])
                relationship = "strong positive" if corr_value > 0 else "strong negative"
                insights.append({
                    'type': 'correlation',
                    'title': 'Key Relationship Found',
                    'content': f'The strongest correlation exists between {max_pair[0]} and {max_pair[1]} ({corr_value:.2f}), '
                              f'indicating a {relationship} relationship. '
                              f'This suggests that changes in {max_pair[0]} significantly impact {max_pair[1]}.'
                })
        except Exception as e:
            print(f"Error calculating correlations: {e}")
    
    # Time-based insights
    if date_cols and numeric_cols:
        try:
            date_col = date_cols[0]
            num_col = numeric_cols[0]
            df_sorted = df.sort_values(date_col)
            
            if len(df_sorted) >= 4:
                first_quarter = float(df_sorted[num_col].iloc[:len(df_sorted)//4].mean())
                last_quarter = float(df_sorted[num_col].iloc[-len(df_sorted)//4:].mean())
                
                if last_quarter > first_quarter and first_quarter != 0:
                    growth = float(((last_quarter - first_quarter) / first_quarter) * 100)
                    insights.append({
                        'type': 'trend',
                        'title': 'Growth Trend Detected',
                        'content': f'Comparing the first and last quarters, {num_col} shows a growth of {growth:.1f}%. '
                                  f'The average increased from {first_quarter:.2f} to {last_quarter:.2f}. '
                                  f'This upward trajectory suggests positive momentum in the business.'
                    })
                elif last_quarter < first_quarter and first_quarter != 0:
                    decline = float(((first_quarter - last_quarter) / first_quarter) * 100)
                    insights.append({
                        'type': 'trend',
                        'title': 'Decline Pattern Observed',
                        'content': f'There is a concerning decline of {decline:.1f}% in {num_col} from {first_quarter:.2f} to {last_quarter:.2f}. '
                                  f'This requires immediate attention and further investigation.'
                    })
        except Exception as e:
            print(f"Error analyzing trends: {e}")
    
    # Statistical summary
    primary_cat = categorical_cols[0] if categorical_cols else "the primary variable"
    primary_num = numeric_cols[0] if numeric_cols else "business performance"
    
    quality_description = "excellent" if quality_score > 90 else "good" if quality_score > 70 else "moderate"
    outlier_message = "The presence of outliers suggests the need for further investigation of extreme values." if outliers > 0 else ""
    
    insights.append({
        'type': 'summary',
        'title': 'Key Takeaways',
        'content': f'The data reveals that {primary_cat} is the main driver of {primary_num}. '
                  f'With a quality score of {quality_score}%, the dataset is {quality_description}. '
                  f'{outlier_message}'
    })
    
    # Convert all insights to ensure JSON serializable
    insights = convert_to_native_types(insights)
    
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
        original_stats = {
            'rows': int(len(df)),
            'columns': int(len(df.columns)),
            'missing': int(df.isnull().sum().sum()),
            'duplicates': int(df.duplicated().sum()),
            'column_names': [str(col) for col in df.columns]
        }
        
        # Clean data
        cleaned_df, cleaning_stats = clean_data(df)
        
        # Generate visualizations
        charts = generate_visualizations(cleaned_df)
        
        # Generate narrative
        narrative = generate_narrative(cleaned_df, cleaning_stats)
        
        # Prepare preview data (convert to native types)
        preview_data = cleaned_df.head(10).copy()
        # Convert datetime columns to string for JSON
        for col in preview_data.select_dtypes(include=['datetime64']).columns:
            preview_data[col] = preview_data[col].astype(str)
        # Replace NaN with None for JSON
        preview_data = preview_data.where(pd.notnull(preview_data), None)
        preview_records = preview_data.to_dict('records')
        preview_records = convert_to_native_types(preview_records)
        
        # Prepare response with all native Python types
        response = {
            'original_data': convert_to_native_types(original_stats),
            'cleaning_stats': convert_to_native_types(cleaning_stats),
            'cleaned_columns': [str(col) for col in cleaned_df.columns],
            'numeric_cols': int(len(cleaned_df.select_dtypes(include=[np.number]).columns)),
            'categorical_cols': int(len(cleaned_df.select_dtypes(include=['object', 'category']).columns)),
            'charts': charts,
            'narrative': narrative,
            'preview': preview_records,
            'filename': str(file.filename)
        }
        
        # Final conversion to ensure everything is JSON serializable
        response = convert_to_native_types(response)
        
        return jsonify(response)
    
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
