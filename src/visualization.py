import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from typing import Any

def plot_correlation_heatmap(df: pd.DataFrame) -> Any:
    """Plots a correlation heatmap for numerical columns."""
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    corr = numeric_df.corr()
    fig = px.imshow(corr, text_auto=True, aspect="auto", 
                    title="Feature Correlation Heatmap", color_continuous_scale='RdBu_r')
    return fig

def plot_violin(df: pd.DataFrame, x_col: str, y_col: str) -> Any:
    """Plots a violin plot for distributions."""
    fig = px.violin(df, y=y_col, x=x_col, box=True, points="all",
                    title=f"Distribution of {y_col} by {x_col}")
    return fig

def plot_sunburst(df: pd.DataFrame, path: list, values: str) -> Any:
    """Plots a sunburst chart for hierarchical data."""
    fig = px.sunburst(df, path=path, values=values, 
                      title=f"Hierarchical View: {' -> '.join(path)}")
    return fig

def plot_gauge(value: float, title: str, max_val: float = 100, is_risk: bool = False) -> Any:
    """Plots a gauge chart. Color inversion if it's a risk metric."""
    colors = ['green', 'yellow', 'red'] if is_risk else ['red', 'yellow', 'green']
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = value,
        title = {'text': title},
        gauge = {
            'axis': {'range': [None, max_val]},
            'steps': [
                {'range': [0, max_val/3], 'color': "lightgray"},
                {'range': [max_val/3, 2*max_val/3], 'color': "gray"}],
            'bar': {'color': "darkblue"}
        }
    ))
    return fig

def traffic_light_indicator(score: float, max_score: float = 100) -> str:
    """Returns an HTML snippet for a traffic light indicator based on score."""
    pct = score / max_score
    if pct < 0.33:
        color = "red"
        text = "High Risk / Poor"
    elif pct < 0.66:
        color = "orange"
        text = "Medium Risk / Fair"
    else:
        color = "green"
        text = "Low Risk / Excellent"
        
    return f"""
    <div style="display:flex; align-items:center;">
        <div style="width:20px; height:20px; border-radius:50%; background-color:{color}; margin-right:10px;"></div>
        <b>{text}</b>
    </div>
    """
