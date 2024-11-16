import streamlit as st
import pandas as pd
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report

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
        return ProfileReport(df, 
                           explorative=True,
                           minimal=True)

    # Load data
    df = load_data()
    
    # Show sample of dataframe
    st.subheader("Data Preview")
    st.dataframe(df.head())
    
    # Generate profile report
    with st.spinner("Generating Profile Report..."):
        pr = generate_profile_report(df)
        st_profile_report(pr)
