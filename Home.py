# Home.py
import streamlit as st
from pathlib import Path

st.set_page_config(page_title="Home", layout="wide")

APP_VERSION = "home-2025-09-25b" 

with st.sidebar:
    st.write(f"App version: {APP_VERSION}")
    if st.button("🔄 Clear caches & rerun"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()

st.title("Home")
st.write("Use the links below to open pages:")

# Link to your multipage file
st.page_link("pages/01_Company_Search.py", label= "Company Search", icon="🔎")
st.page_link("pages/02_ETF_Mean_Reversion.py", label= "ETF Mean Reversion", icon="📈")
st.page_link("pages/03_Margin__Optimization.py", label= "Margin Optimization", icon="⚖️")

