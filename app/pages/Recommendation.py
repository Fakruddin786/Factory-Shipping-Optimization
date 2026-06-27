import streamlit as st
import joblib
import pandas as pd
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.config import MODELS_DIR, PRODUCT_DATA, FACTORIES, REGIONS, SHIP_MODES
from src.recommendation_engine import get_recommendations

st.set_page_config(page_title="Recommendations", layout="wide")
st.title("Recommendation Engine")

@st.cache_resource
def load_model():
    model = joblib.load(os.path.join(MODELS_DIR, "best_model.joblib"))
    features = joblib.load(os.path.join(MODELS_DIR, "feature_cols.joblib"))
    return model, features

try:
    model, feature_cols = load_model()
    
    product = st.selectbox("Product", list(PRODUCT_DATA.keys()))
    curr_factory = PRODUCT_DATA[product]["Factory"]
    
    col1, col2 = st.columns(2)
    with col1:
        region = st.selectbox("Region", REGIONS)
        ship_mode = st.selectbox("Ship Mode", SHIP_MODES)
        cust_lat = st.number_input("Cust Lat", value=40.0)
        cust_lon = st.number_input("Cust Lon", value=-100.0)
    with col2:
        qty = st.number_input("Units", value=100)
        price = st.number_input("Price ($)", value=25.0)
        cost = st.number_input("Cost ($)", value=10.0)
        
    st.markdown("---")
    st.markdown("### Optimization Priority")
    speed_priority = st.slider("Speed vs Profit Priority (Higher = Faster Shipping, Lower = Max Profit)", 
                               min_value=0, max_value=70, value=40, step=5)
    profit_priority = 70 - speed_priority
    st.write(f"Current Weights: **Speed {speed_priority}%** | **Profit {profit_priority}%** | Risk 20% | Confidence 10%")
        
    if st.button("Generate Recommendations"):
        recs = get_recommendations(curr_factory, cust_lat, cust_lon, qty, price, cost, 
                                   ship_mode, region, model, feature_cols, FACTORIES,
                                   speed_weight=speed_priority, profit_weight=profit_priority)
        
        st.subheader("Top-5 Recommendations")
        df_recs = pd.DataFrame(recs)
        if not df_recs.empty:
            st.dataframe(df_recs.style.highlight_max(subset=['Score', 'Confidence'], color='lightgreen'))
            
            csv = df_recs.to_csv(index=False).encode('utf-8')
            
            # Use columns for layout
            col_a, col_b = st.columns(2)
            with col_a:
                st.download_button(
                    label="Download Recommendations as CSV",
                    data=csv,
                    file_name='recommendations.csv',
                    mime='text/csv',
                )
            
            with col_b:
                from src.utils import df_to_pdf
                pdf_bytes = df_to_pdf(df_recs, title="Top 5 Factory Recommendations")
                st.download_button(
                    label="Download Recommendations as PDF",
                    data=pdf_bytes,
                    file_name='recommendations.pdf',
                    mime='application/pdf',
                )
        else:
            st.warning("No recommendations generated.")
            
except Exception as e:
    st.error(f"Error loading Recommendations: {e}")
