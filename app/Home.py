import streamlit as st
import pandas as pd
import os
import sys

# Add project root to sys path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import PROCESSED_DATA_DIR

st.set_page_config(page_title="Factory Optimization Home", layout="wide")

st.markdown("""
# 🏭 Factory Reallocation & Shipping Optimization Recommendation System
### Nassau Candy Distributor

---

### 📌 Project Overview
Welcome to the Factory Reallocation & Shipping Optimization Recommendation System.

This application helps Nassau Candy optimize its manufacturing and shipping operations using Data Analytics, Machine Learning, and Decision Intelligence.

Instead of relying on fixed factory assignments, the system predicts shipping performance under different factory configurations and recommends the best manufacturing location for each product.

### 🎯 Business Objective
The main objective is to improve logistics efficiency by:
- Reducing shipping lead time
- Increasing operational efficiency
- Maintaining or improving profitability
- Supporting data-driven factory allocation decisions

---
""")

@st.cache_data
def load_data():
    path = os.path.join(PROCESSED_DATA_DIR, "cleaned_orders.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()

df = load_data()

st.markdown("### 📊 Executive KPIs")

if df.empty:
    st.error("No data found! Please run `python main.py` first.")
else:
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("📦 Total Orders", f"{len(df):,}")
    with col2:
        st.metric("💰 Total Sales", f"${df['Sales'].sum():,.2f}")
    with col3:
        st.metric("📈 Avg Lead Time", f"{df['Lead_Time_Days'].mean():.2f} Days")
    with col4:
        st.metric("🏭 Factories", "5")
    with col5:
        st.metric("🍬 Products", "15")

st.markdown("""
---

### 🚀 Dashboard Modules

- **📊 Dashboard**: Explore sales, profit, lead time, regional trends, shipping performance, and product insights.
- **🏭 Factory Optimization Simulator**: Test different factory assignments and predict their impact on shipping lead time and profitability.
- **🎯 Recommendation Dashboard**: View AI-powered Top-5 factory recommendations for each product.
- **📈 Scenario Analysis**: Compare the current factory with recommended alternatives and measure expected improvements.
- **⚠️ Risk Analysis**: Identify delayed shipments, low-profit products, and high-risk logistics routes.

### 🛠 Technologies Used
Python | Pandas | NumPy | Scikit-learn | Plotly | Streamlit | Joblib

### 🎯 Expected Outcome
This system enables Nassau Candy to make smarter factory allocation decisions by reducing delivery times, improving shipping efficiency, and providing actionable recommendations through an interactive dashboard.
""")
