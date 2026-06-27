import streamlit as st
import pandas as pd
import os
import sys
import plotly.express as px

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.config import PROCESSED_DATA_DIR
from src.visualization import plot_violin, plot_sunburst, plot_correlation_heatmap

st.set_page_config(page_title="Dashboard", layout="wide")
st.title("Exploratory Data Analysis Dashboard")

@st.cache_data
def load_data():
    return pd.read_csv(os.path.join(PROCESSED_DATA_DIR, "cleaned_orders.csv"))

try:
    df = load_data()
    
    st.subheader("Sales by Region & Ship Mode")
    fig_sunburst = plot_sunburst(df, path=['Region', 'Ship Mode'], values='Sales')
    st.plotly_chart(fig_sunburst, use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Lead Time Distribution")
        fig_violin = plot_violin(df, x_col='Ship Mode', y_col='Lead_Time_Days')
        st.plotly_chart(fig_violin, use_container_width=True)
        
    with col2:
        st.subheader("Feature Correlation")
        fig_corr = plot_correlation_heatmap(df[['Distance_KM', 'Sales', 'Units', 'Lead_Time_Days', 'Gross Profit']])
        st.plotly_chart(fig_corr, use_container_width=True)
        
except Exception as e:
    st.error(f"Error loading dashboard: {e}")
