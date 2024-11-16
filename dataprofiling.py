import streamlit as st
import pandas as pd
import sweetviz as sv
import codecs
import os

st.set_page_config(page_title="Data Profiling App", layout="wide")

st.title("Data Profiling App")

# File upload
uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])

if uploaded_file is not None:
    @st.cache_data
    def load_data():
        return pd.read_csv(uploaded_file)
    
    @st.cache_data
    def generate_profile_report(df):
        report = sv.analyze(df)
        report.show_html('report.html')
        return 'report.html'

    # Load data
    df = load_data()
    
    # Show sample of dataframe
    st.subheader("Data Preview")
    st.dataframe(df.head())
    
    # Generate profile report
    with st.spinner("Generating Profile Report..."):
        report_path = generate_profile_report(df)
        
        # Display the report
        with open(report_path, 'r', encoding='utf-8') as report_file:
            report_html = report_file.read()
        st.components.v1.html(report_html, width=1100, height=600, scrolling=True)
        
        # Cleanup
        if os.path.exists(report_path):
            os.remove(report_path)
