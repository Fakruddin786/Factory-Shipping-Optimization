import os
from src.utils import setup_logger
from src.data_generation import generate_synthetic_data
from src.preprocessing import run_preprocessing
from src.feature_engineering import generate_features, prepare_modeling_data
from src.model_training import train_and_evaluate_models, save_model
from src.clustering import run_clustering
from src.config import TARGET_COLUMN

logger = setup_logger(__name__)

def main():
    logger.info("--- Starting Factory Optimization Pipeline ---")
    
    # 1. Generate Raw Data
    logger.info("STEP 1: Generating Data")
    generate_synthetic_data(10000)
    
    # 2. Preprocess Data
    logger.info("STEP 2: Preprocessing Data")
    df_clean = run_preprocessing()
    
    # 3. Feature Engineering
    logger.info("STEP 3: Feature Engineering")
    df_feat = generate_features(df_clean)
    X, y, feature_cols = prepare_modeling_data(df_feat, TARGET_COLUMN)
    
    # 4. Model Training
    logger.info("STEP 4: Model Training")
    best_model, best_name, results_df = train_and_evaluate_models(X, y, feature_cols)
    save_model(best_model, feature_cols)
    
    # 5. Route Clustering
    logger.info("STEP 5: Route Clustering")
    # clustering expects the feature engineered df
    route_perf, kmeans_model, scaler = run_clustering(df_feat)
    
    logger.info("--- Pipeline Completed Successfully ---")
    logger.info("You can now run the Streamlit dashboard using: streamlit run app/Home.py")

if __name__ == "__main__":
    main()
