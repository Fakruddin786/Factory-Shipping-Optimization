import streamlit as st
import joblib
import pandas as pd
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.config import MODELS_DIR, PRODUCT_DATA, FACTORIES, REGIONS, SHIP_MODES
from src.simulation import simulate_reallocation

st.set_page_config(page_title="Scenario Analysis", layout="wide")
st.title("Scenario Analysis")

@st.cache_resource
def load_model():
    model = joblib.load(os.path.join(MODELS_DIR, "best_model.joblib"))
    features = joblib.load(os.path.join(MODELS_DIR, "feature_cols.joblib"))
    return model, features

try:
    model, feature_cols = load_model()
    
    st.markdown("Compare the performance improvements of alternative scenarios.")
    
    product = st.selectbox("Select Product", list(PRODUCT_DATA.keys()))
    curr_factory = PRODUCT_DATA[product]["Factory"]
    target_factory = st.selectbox("Select Target Factory", [f for f in FACTORIES.keys() if f != curr_factory])
    
    col1, col2 = st.columns(2)
    with col1:
        ship_mode = st.selectbox("Ship Mode", SHIP_MODES)
        region = st.selectbox("Region", REGIONS)
    with col2:
        quantity = st.number_input("Units", value=500)
        unit_price = st.number_input("Unit Price ($)", value=15.0)
        unit_cost = st.number_input("Unit Cost ($)", value=7.0)
        
    cust_lat = st.number_input("Customer Latitude", value=35.0)
    cust_lon = st.number_input("Customer Longitude", value=-95.0)
    
    if st.button("Analyze Scenario"):
        results = simulate_reallocation(product, curr_factory, target_factory, cust_lat, cust_lon, 
                                        quantity, unit_price, unit_cost, ship_mode, region, 
                                        model, feature_cols, FACTORIES)
                                        
        st.subheader("Performance Improvement")
        c1, c2 = st.columns(2)
        c1.metric("Lead Time Improvement", f"{results.get('Improvement_Percentage_Speed', 0):.2f}%")
        c2.metric("Profit Improvement", f"{results.get('Improvement_Percentage_Profit', 0):.2f}%")
        
        st.write("---")
        st.write(f"**Current Factory ({curr_factory})** vs **Target Factory ({target_factory})**")
        
        df_comp = pd.DataFrame({
            "Metric": ["Distance (KM)", "Lead Time (Days)", "Profit ($)"],
            "Current": [results.get("New_Distance_KM", 0) - results.get("Distance_Difference_KM", 0),
                        results.get("New_Lead_Time", 0) + results.get("Predicted_Lead_Time_Difference", 0),
                        results.get("New_Profit", 0) - results.get("Profit_Impact", 0)],
            "Target": [results.get("New_Distance_KM", 0), 
                       results.get("New_Lead_Time", 0), 
                       results.get("New_Profit", 0)]
        })
        st.table(df_comp)
        
        st.markdown("---")
        csv_comp = df_comp.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Save Scenario Comparison (CSV)",
            data=csv_comp,
            file_name=f'scenario_comparison_{product.replace(" ", "_")}.csv',
            mime='text/csv',
        )
        
except Exception as e:
    st.error(f"Error loading Scenario Analysis: {e}")
