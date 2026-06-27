import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# pyrefly: ignore [missing-import]
from app_utils import FACTORIES, PRODUCT_DATA, REGIONS, SHIP_MODES, haversine
# pyrefly: ignore [missing-import]
from optimization import simulate_reallocation
# pyrefly: ignore [missing-import]
from recommendation_engine import get_recommendations

st.set_page_config(page_title="Factory Optimization System", layout="wide")

@st.cache_data
def load_data():
    data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "orders.csv")
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        df['Order Date'] = pd.to_datetime(df['Order Date'])
        df['Ship Date'] = pd.to_datetime(df['Ship Date'])
        return df.dropna()
    return pd.DataFrame()

@st.cache_data
def load_clusters():
    cluster_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "route_clusters.csv")
    if os.path.exists(cluster_path):
        return pd.read_csv(cluster_path)
    return pd.DataFrame()

@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "best_model.joblib")
    feat_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "feature_cols.joblib")
    if os.path.exists(model_path) and os.path.exists(feat_path):
        return joblib.load(model_path), joblib.load(feat_path)
    return None, None

df = load_data()
route_clusters = load_clusters()
model, feature_cols = load_model()

st.sidebar.title("Navigation")
page = st.sidebar.radio("Select a Page", [
    "Executive Overview", 
    "EDA Dashboard", 
    "Route & Product Clustering",
    "Factory Optimization Simulator", 
    "Recommendation Dashboard", 
    "Risk and Impact Dashboard", 
    "Geo-Spatial Factory Map"
])

if df.empty:
    st.error("Data not found! Please run the data generation script.")
    st.stop()

if page == "Executive Overview":
    st.title("Executive Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Orders", f"{len(df):,}")
    with col2:
        st.metric("Total Sales", f"${df['Sales'].sum():,.2f}")
    with col3:
        st.metric("Total Profit", f"${df['Gross Profit'].sum():,.2f}")
    with col4:
        st.metric("Avg Lead Time", f"{df['Lead_Time_Days'].mean():.2f} days")
        
    st.markdown("### High-Level Factory Performance")
    factory_perf = df.groupby('Factory')[['Sales', 'Gross Profit', 'Lead_Time_Days', 'Distance_KM']].mean().reset_index()
    st.dataframe(factory_perf.style.format({"Sales": "${:.2f}", "Gross Profit": "${:.2f}", "Lead_Time_Days": "{:.2f}", "Distance_KM": "{:.2f}"}))

elif page == "EDA Dashboard":
    st.title("Exploratory Data Analysis")
    
    st.subheader("Sales by Region")
    region_sales = df.groupby('Region', observed=False)['Sales'].sum().reset_index()
    fig1 = px.bar(region_sales, x='Region', y='Sales', title="Sales by Region", color='Sales')
    st.plotly_chart(fig1, use_container_width=True)
    
    st.subheader("Sales by Ship Mode")
    mode_sales = df.groupby('Ship Mode')['Sales'].sum().reset_index()
    fig2 = px.pie(mode_sales, values='Sales', names='Ship Mode', title="Sales by Ship Mode")
    st.plotly_chart(fig2, use_container_width=True)
    
    st.subheader("Product Sales Distribution")
    prod_sales = df.groupby('Product Name')['Sales'].sum().reset_index().sort_values('Sales')
    fig3 = px.bar(prod_sales, x='Sales', y='Product Name', orientation='h', title="Sales by Product")
    st.plotly_chart(fig3, use_container_width=True)

elif page == "Route & Product Clustering":
    st.title("Route & Product Clustering")
    st.write("Clustered region-product combinations based on performance similarity to identify inefficiencies.")
    
    if route_clusters.empty:
        st.warning("Clustering data not found. Please run clustering.py.")
    else:
        st.subheader("Cluster Insights")
        fig = px.scatter_3d(route_clusters, x='Distance_KM', y='Profit_Margin', z='Lead_Time_Days',
                            color='Cluster', hover_name='Factory', hover_data=['Region', 'Division'],
                            title="3D Cluster Visualization")
        st.plotly_chart(fig, use_container_width=True)
        
        slow_routes = route_clusters[route_clusters['Is_Slow_Route']]
        st.subheader("Consistently Slow Routes")
        st.dataframe(slow_routes[['Factory', 'Region', 'Division', 'Lead_Time_Days', 'Profit_Margin']])

elif page == "Factory Optimization Simulator":
    st.title("Factory Optimization Simulator")
    
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
        if model:
            results = simulate_reallocation(product, curr_factory, target_factory, cust_lat, cust_lon, quantity, unit_price, unit_cost, ship_mode, region, model, feature_cols, FACTORIES, haversine)
            
            st.subheader("Simulation Results")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Distance Diff (KM)", f"{results['Distance_Difference_KM']:.2f}")
            c2.metric("Lead Time Diff (Days)", f"{results['Predicted_Lead_Time_Difference']:.2f}")
            c3.metric("Profit Impact ($)", f"{results['Profit_Impact']:.2f}")
            c4.metric("Risk Impact", results['Risk_Impact'])
            
            if results['Profit_Impact'] > 0 and results['Predicted_Lead_Time_Difference'] < 0:
                st.success("Recommendation: High Confidence. The new assignment improves both speed and profit.")
            else:
                st.warning("Recommendation: Proceed with caution. Trade-offs exist.")
        else:
            st.error("Model not loaded.")

elif page == "Recommendation Dashboard":
    st.title("Recommendation Dashboard")
    st.write("Based on order parameters, here are the top alternative factory recommendations.")
    
    st.markdown("### Optimization Priority")
    speed_weight = st.slider("Focus on Speed vs Profit (0 = Max Profit, 1 = Max Speed)", 0.0, 1.0, 0.5)
    profit_weight = 1.0 - speed_weight
    
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
        
    if st.button("Generate Recommendations"):
        if model:
            recs = get_recommendations(curr_factory, cust_lat, cust_lon, qty, price, cost, ship_mode, region, model, feature_cols, FACTORIES, haversine, speed_weight, profit_weight)
            st.subheader("Top 3 Recommendations")
            for i, rec in enumerate(recs):
                st.markdown(f"#### {i+1}. {rec['Factory']}")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Score", f"{rec['Score']:.2f}%")
                c2.metric("Confidence", f"{rec['Confidence']:.2f}%")
                c3.metric("Lead Time Reduction", f"{rec['Lead_Time_Reduction']:.2f} days")
                c4.metric("Expected Savings", f"${rec['Expected_Savings']:.2f}")
                st.write("---")
        else:
            st.error("Model not loaded.")

elif page == "Risk and Impact Dashboard":
    st.title("Risk and Impact Dashboard")
    st.write("Analyzing potential risks with factory reassignments.")
    
    high_risk_df = df[df['Profit_Margin'] < 0.1]
    st.metric("Orders with < 10% Margin", f"{len(high_risk_df)} / {len(df)}")
    
    st.subheader("Profit Risks by Factory")
    risk_by_factory = high_risk_df.groupby('Factory').size().reset_index(name='Low_Margin_Count')
    fig = px.bar(risk_by_factory, x='Factory', y='Low_Margin_Count', title="Low Margin Orders by Factory", color='Low_Margin_Count', color_continuous_scale='Reds')
    st.plotly_chart(fig)

elif page == "Geo-Spatial Factory Map":
    st.title("Geo-Spatial Factory Map")
    
    factory_df = pd.DataFrame([
        {"Factory": name, "Lat": coords["Lat"], "Lon": coords["Lon"]}
        for name, coords in FACTORIES.items()
    ])
    
    fig = px.scatter_mapbox(
        factory_df, lat="Lat", lon="Lon", hover_name="Factory",
        color_discrete_sequence=["red"], zoom=3, height=600
    )
    
    fig.update_layout(mapbox_style="open-street-map")
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    st.plotly_chart(fig, use_container_width=True)
