from flask import Flask, render_template, request, send_file, jsonify
import pandas as pd
import numpy as np
import os
from datetime import datetime
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import plotly.express as px
import plotly.figure_factory as ff
import io
import json

app = Flask(__name__)

# Ensure the uploads directory exists
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def create_distribution_plot(df, column):
    fig = px.histogram(df, x=column, title=f'Distribution of {column}')
    return fig.to_json()

def create_correlation_heatmap(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 1:
        corr_matrix = df[numeric_cols].corr()
        fig = px.imshow(corr_matrix, 
                       title='Correlation Heatmap',
                       color_continuous_scale='RdBu')
        return fig.to_json()
    return None

def process_data(df, operations):
    processed_df = df.copy()
    transformed_cols = {}
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for op in operations:
        if op == 'standardization':
            scaler = StandardScaler()
            processed_df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
            transformed_cols['standardized'] = list(numeric_cols)
            
        elif op == 'normalization':
            scaler = MinMaxScaler()
            processed_df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
            transformed_cols['normalized'] = list(numeric_cols)
            
        elif op == 'log_transform':
            for col in numeric_cols:
                if (df[col] > 0).all():
                    processed_df[f'{col}_log'] = np.log(df[col])
                    if 'log_transformed' not in transformed_cols:
                        transformed_cols['log_transformed'] = []
                    transformed_cols['log_transformed'].append(col)
    
    return processed_df, transformed_cols

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file uploaded'
        
        file = request.files['file']
        if file.filename == '':
            return 'No file selected'
        
        if file and file.filename.endswith('.csv'):
            # Read the CSV file
            df = pd.read_csv(file)
            
            # Store the DataFrame in session
            df.to_csv(os.path.join(UPLOAD_FOLDER, 'temp.csv'), index=False)
            
            # Generate basic statistics
            stats = {
                'Total Rows': len(df),
                'Total Columns': len(df.columns),
                'Column Names': list(df.columns),
                'Missing Values': df.isnull().sum().to_dict(),
                'Data Types': df.dtypes.astype(str).to_dict()
            }
            
            # Generate summary statistics for numeric columns
            numeric_stats = df.describe().to_html(classes='table table-striped')
            
            # Create visualizations
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            plots = {}
            for col in numeric_cols[:5]:  # Limit to first 5 numeric columns
                plots[col] = create_distribution_plot(df, col)
            
            correlation_plot = create_correlation_heatmap(df)
            
            # Sample data preview
            preview = df.head().to_html(classes='table table-striped')
            
            return render_template('report.html', 
                                stats=stats,
                                numeric_stats=numeric_stats,
                                preview=preview,
                                plots=plots,
                                correlation_plot=correlation_plot,
                                numeric_columns=list(numeric_cols))
        
        return 'Invalid file format. Please upload a CSV file.'
    
    return render_template('upload.html')

@app.route('/process', methods=['POST'])
def process():
    operations = request.json.get('operations', [])
    
    # Read the stored CSV file
    df = pd.read_csv(os.path.join(UPLOAD_FOLDER, 'temp.csv'))
    
    # Process the data
    processed_df, transformed_cols = process_data(df, operations)
    
    # Save processed data
    output = io.BytesIO()
    processed_df.to_csv(output, index=False)
    output.seek(0)
    
    # Save the processed file
    processed_df.to_csv(os.path.join(UPLOAD_FOLDER, 'processed.csv'), index=False)
    
    return jsonify({
        'message': 'Data processed successfully',
        'transformed_columns': transformed_cols,
        'preview': processed_df.head().to_html(classes='table table-striped')
    })

@app.route('/download/<file_type>')
def download(file_type):
    if file_type == 'original':
        filename = 'temp.csv'
    else:
        filename = 'processed.csv'
    
    return send_file(
        os.path.join(UPLOAD_FOLDER, filename),
        as_attachment=True,
        download_name=f'data_{file_type}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    )

if __name__ == '__main__':
    app.run(debug=True)