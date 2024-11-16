import pandas as pd
import streamlit as st
from streamlit_pandas_profiling import st_profile_report
from pandas_profiling import ProfileReport

# Set up the Streamlit app
st.title('Data Profiling with Streamlit and Pandas Profiling')

# File uploader for CSV files
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    # Generate the profile report
    profile = ProfileReport(df)
    
    st_profile_report(profile)
else:
    st.write("Please upload a CSV file to generate the profile report.")
