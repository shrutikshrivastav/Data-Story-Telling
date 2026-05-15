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
    """Generate interactive Plotly charts"""
    numeric_cols, categorical_cols, date_cols = identify_columns(df)
    charts = []
    
    # Bar Chart - Top categories
    if categorical_cols and numeric_cols:
        try:
            cat_col = categorical_cols[0]
            num_col = numeric_cols[0]
            grouped = df.groupby(cat_col)[num_col].sum().sort_values(ascending=False).head(10)
            
            fig = px.bar(
                x=grouped.index.astype(str), 
                y=grouped.values,
                title=f'Top 10 {cat_col} by {num_col}',
                template='plotly_dark',
                color=grouped.values,
                color_continuous_scale='viridis'
            )
            fig.update_layout(
                showlegend=False,
                xaxis_tickangle=-45,
                margin=dict(t=50, l=50, r=50, b=100)
            )
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
            
            fig = px.line(
                df_sorted, 
                x=date_col, 
                y=num_col,
                title=f'{num_col} Trend Over Time',
                template='plotly_dark'
            )
            fig.update_traces(line_color='#00ff88', line_width=2)
            fig.update_layout(margin=dict(t=50, l=50, r=50, b=50))
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
            
            fig = px.pie(
                values=value_counts.values,
                names=value_counts.index.astype(str),
                title=f'{cat_col} Distribution',
                template='plotly_dark',
                hole=0.3
            )
            fig.update_traces(marker=dict(colors=px.colors.sequential.Viridis))
            fig.update_layout(margin=dict(t=50, l=50, r=50, b=50))
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
            fig = px.histogram(
                df, 
                x=num_col, 
                nbins=30,
                title=f'{num_col} Distribution',
                template='plotly_dark',
                color_discrete_sequence=['#00ff88']
            )
            fig.update_layout(margin=dict(t=50, l=50, r=50, b=50))
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
            fig = px.scatter(
                df, 
                x=numeric_cols[0], 
                y=numeric_cols[1],
                title=f'{numeric_cols[0]} vs {numeric_cols[1]}',
                template='plotly_dark',
                opacity=0.6
            )
            fig.update_traces(marker=dict(color='#00ff88', size=8))
            fig.update_layout(margin=dict(t=50, l=50, r=50, b=50))
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
            
            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns.tolist(),
                y=corr_matrix.columns.tolist(),
                colorscale='Viridis',
                text=np.round(corr_matrix.values, 2),
                texttemplate='%{text}',
                textfont={"size": 10}
            ))
            fig.update_layout(
                title='Correlation Heatmap',
                template='plotly_dark',
                margin=dict(t=50, l=50, r=50, b=100)
            )
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
        'title': '📊 Dataset Overview',
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
                if std_val > mean_val * 0.5 and mean_val != 0:
                    trend = "highly variable"
                elif mean_val != 0 and std_val < mean_val * 0.1:
                    trend = "consistent"
                
                insights.append({
                    'type': 'numeric',
                    'title': f'📈 {col.title()} Analysis',
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
                    'title': f'🏷️ {cat_col.title()} Breakdown',
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
                    'title': '🔗 Key Relationship Found',
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
                        'title': '📅 Growth Trend Detected',
                        'content': f'Comparing the first and last quarters, {num_col} shows a growth of {growth:.1f}%. '
                                  f'The average increased from {first_quarter:.2f} to {last_quarter:.2f}. '
                                  f'This upward trajectory suggests positive momentum in the business.'
                    })
                elif last_quarter < first_quarter and first_quarter != 0:
                    decline = float(((first_quarter - last_quarter) / first_quarter) * 100)
                    insights.append({
                        'type': 'trend',
                        'title': '📉 Decline Pattern Observed',
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
        'title': '🎯 Key Takeaways',
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
