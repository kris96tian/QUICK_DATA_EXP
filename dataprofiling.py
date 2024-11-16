import streamlit as st
import pandas as pd
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report

st.set_page_config(page_title="Data Profiling App", layout="wide")

st.title("Data Profiling App")

uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])

if uploaded_file is not None:
    try:
        # Read the CSV file
        df = pd.read_csv(uploaded_file)
        
        # Show basic info
        st.subheader("Data Preview")
        st.write(df.head())
        
        # Generate profile report with minimal configuration
        with st.spinner("Generating Profile Report..."):
            profile = ProfileReport(df, 
                                 minimal=True,
                                 explorative=True,
                                 dark_mode=True)
            
            # Display the report
            st_profile_report(profile)
        
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.exception(e)
