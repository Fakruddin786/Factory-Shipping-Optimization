import streamlit as st
import joblib
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.config import MODELS_DIR, PRODUCT_DATA, FACTORIES, REGIONS, SHIP_MODES
from src.simulation import simulate_reallocation

st.set_page_config(page_title="Simulator", layout="wide")
st.title("Factory Optimization Simulator")

@st.cache_resource
def load_model():
    model = joblib.load(os.path.join(MODELS_DIR, "best_model.joblib"))
    features = joblib.load(os.path.join(MODELS_DIR, "feature_cols.joblib"))
    return model, features

try:
    model, feature_cols = load_model()
    
    col1, col2 = st.columns(2)
    with col1:
        product = st.selectbox("Select Product", list(PRODUCT_DATA.keys()))
        curr_factory = PRODUCT_DATA[product]["Factory"]
        st.info(f"Current Factory: {curr_factory}")
        
        target_factory = st.selectbox("Select Recommended/New Factory", [f for f in FACTORIES.keys() if f != curr_factory])
        ship_mode = st.selectbox("Ship Mode", SHIP_MODES)
        
    with col2:
        region = st.selectbox("Region", REGIONS)
        cust_lat = st.number_input("Customer Latitude", value=40.0)
        cust_lon = st.number_input("Customer Longitude", value=-100.0)
        quantity = st.number_input("Units", value=100)
        unit_price = st.number_input("Unit Price ($)", value=25.0)
        unit_cost = st.number_input("Unit Cost ($)", value=10.0)
        
    if st.button("Run Simulation"):
        results = simulate_reallocation(product, curr_factory, target_factory, cust_lat, cust_lon, 
                                        quantity, unit_price, unit_cost, ship_mode, region, 
                                        model, feature_cols, FACTORIES)
        
        st.subheader("Simulation Results")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Distance Diff (KM)", f"{results.get('Distance_Difference_KM', 0):.2f}")
        c2.metric("Lead Time Diff (Days)", f"{results.get('Predicted_Lead_Time_Difference', 0):.2f}")
        c3.metric("Profit Impact ($)", f"{results.get('Profit_Impact', 0):.2f}")
        c4.metric("Risk Impact", results.get('Risk_Impact', 'Unknown'))
        
        if results.get('Profit_Impact', 0) > 0 and results.get('Predicted_Lead_Time_Difference', 0) < 0:
            st.success("Recommendation: High Confidence. The new assignment improves both speed and profit.")
        else:
            st.warning("Recommendation: Proceed with caution. Trade-offs exist.")
            
except Exception as e:
    st.error(f"Error loading Simulator: {e}. Please ensure models are trained.")
