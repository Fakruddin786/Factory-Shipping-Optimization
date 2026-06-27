import streamlit as st
import pandas as pd
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.config import PROCESSED_DATA_DIR
from src.visualization import plot_gauge, traffic_light_indicator

st.set_page_config(page_title="Risk Analysis", layout="wide")
st.title("Risk Analysis Dashboard")

@st.cache_data
def load_data():
    return pd.read_csv(os.path.join(PROCESSED_DATA_DIR, "cleaned_orders.csv"))

try:
    df = load_data()
    
    st.markdown("Analyze risks related to low margins and shipment delays.")
    
    # Calculate Profit_Margin if it doesn't exist
    if 'Profit_Margin' not in df.columns:
        df['Profit_Margin'] = df['Gross Profit'] / df['Sales']
        
    # Simple risk modeling for dashboard
    df['Margin_Risk'] = (df['Profit_Margin'] < 0.1).astype(int)
    
    # Calculate overall risk score
    total_orders = len(df)
    high_risk_margin_pct = (df['Margin_Risk'].sum() / total_orders) * 100
    
    st.subheader("Network Risk Indicators")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(traffic_light_indicator(100 - high_risk_margin_pct), unsafe_allow_html=True)
        fig_gauge = plot_gauge(high_risk_margin_pct, "High Risk Margin Orders (%)", is_risk=True)
        st.plotly_chart(fig_gauge, use_container_width=True)
        
    with col2:
        st.markdown("### Risk Categories")
        st.write(f"- **Low Margin Orders**: {df['Margin_Risk'].sum()}")
        st.write(f"- **Total Orders**: {total_orders}")
        st.write(f"- **Network Health Score**: {100 - high_risk_margin_pct:.1f}/100")
        
    st.subheader("High Risk Routes (Low Profit)")
    risk_df = df[df['Margin_Risk'] == 1].groupby(['Factory', 'Region']).size().reset_index(name='High_Risk_Orders')
    st.dataframe(risk_df.sort_values(by='High_Risk_Orders', ascending=False))
        
except Exception as e:
    st.error(f"Error loading Risk Analysis: {e}")
