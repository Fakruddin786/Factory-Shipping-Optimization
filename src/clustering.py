import os
import joblib
import pandas as pd
from typing import Tuple, Any
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler

from src.utils import setup_logger
from src.config import MODELS_DIR, PROCESSED_DATA_DIR, RANDOM_STATE

logger = setup_logger(__name__)

def run_clustering(df: pd.DataFrame) -> Tuple[pd.DataFrame, Any, Any]:
    """
    Performs clustering (KMeans, DBSCAN, Hierarchical) on the routes.
    
    Args:
        df: Cleaned and feature engineered DataFrame.
        
    Returns:
        DataFrame with cluster labels and the clustering model/scaler.
    """
    logger.info("Starting route clustering...")
    
    # We group by Factory, Region, and Product Division
    route_perf = df.groupby(['Factory', 'Region', 'Division']).agg({
        'Lead_Time_Days': 'mean',
        'Distance_KM': 'mean',
        'Profit_Margin': 'mean'
    }).reset_index()
    
    features = ['Lead_Time_Days', 'Distance_KM', 'Profit_Margin']
    X = route_perf[features]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 1. KMeans Clustering
    logger.info("Running KMeans clustering...")
    kmeans = KMeans(n_clusters=4, random_state=RANDOM_STATE, n_init=10)
    route_perf['Cluster_KMeans'] = kmeans.fit_predict(X_scaled)
    
    # 2. DBSCAN Clustering
    logger.info("Running DBSCAN clustering...")
    dbscan = DBSCAN(eps=0.5, min_samples=3)
    route_perf['Cluster_DBSCAN'] = dbscan.fit_predict(X_scaled)
    
    # 3. Hierarchical Clustering
    logger.info("Running Hierarchical clustering...")
    hierarchical = AgglomerativeClustering(n_clusters=4)
    route_perf['Cluster_Hierarchical'] = hierarchical.fit_predict(X_scaled)
    
    # We will use KMeans as the primary for dashboard simplicity, but we have the others
    cluster_means = route_perf.groupby('Cluster_KMeans')[features].mean()
    
    # Identify the "Slow/Congested" cluster (highest lead time)
    slow_cluster_id = cluster_means['Lead_Time_Days'].idxmax()
    route_perf['Is_Slow_Route'] = route_perf['Cluster_KMeans'] == slow_cluster_id
    
    # Save the clustered data
    output_path = os.path.join(PROCESSED_DATA_DIR, "route_clusters.csv")
    route_perf.to_csv(output_path, index=False)
    logger.info(f"Clustered routes saved to {output_path}")
    
    # Save the scaler and kmeans model
    model_path = os.path.join(MODELS_DIR, "clustering_model.joblib")
    joblib.dump((kmeans, scaler), model_path)
    logger.info(f"Clustering model saved to {model_path}")
    
    return route_perf, kmeans, scaler
