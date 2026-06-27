import os
import joblib
import pandas as pd
from typing import Tuple, Any, List
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

from src.utils import setup_logger
from src.config import MODELS_DIR, TEST_SIZE, RANDOM_STATE
from src.evaluation import evaluate_model

logger = setup_logger(__name__)

def train_and_evaluate_models(X: pd.DataFrame, y: pd.Series, feature_cols: List[str]) -> Tuple[Any, str, pd.DataFrame]:
    """
    Trains multiple regression models, evaluates them, and selects the best one.
    
    Args:
        X: Feature dataframe.
        y: Target series.
        feature_cols: List of feature names.
        
    Returns:
        Best model object, best model name, and evaluation results dataframe.
    """
    logger.info("Splitting data into train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(random_state=RANDOM_STATE),
        "Gradient Boosting": GradientBoostingRegressor(random_state=RANDOM_STATE),
        "XGBoost": XGBRegressor(random_state=RANDOM_STATE, objective='reg:squarederror'),
        "LightGBM": LGBMRegressor(random_state=RANDOM_STATE, verbose=-1),
        "CatBoost": CatBoostRegressor(random_state=RANDOM_STATE, verbose=0)
    }
    
    # Optional: We could do GridSearchCV here for the best model, but to save time 
    # we will use default/light parameters for the sweep, and tune the best if needed.
    
    results = []
    best_model = None
    best_r2 = -float('inf')
    best_name = ""
    
    for name, model in models.items():
        logger.info(f"Training {name}...")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        
        eval_metrics = evaluate_model(y_test, preds, name)
        results.append(eval_metrics)
        
        logger.info(f"{name} Results: R2={eval_metrics['R2']:.4f}, MAPE={eval_metrics['MAPE']:.2f}%")
        
        if eval_metrics['R2'] > best_r2:
            best_r2 = eval_metrics['R2']
            best_model = model
            best_name = name
            
    results_df = pd.DataFrame(results).sort_values(by="R2", ascending=False)
    logger.info(f"Best Model Selected: {best_name} with R2: {best_r2:.4f}")
    
    return best_model, best_name, results_df

def save_model(model: Any, feature_cols: List[str], model_name: str = "best_model.joblib"):
    """
    Saves the trained model and feature columns to the models directory.
    
    Args:
        model: Trained model object.
        feature_cols: List of feature names.
        model_name: Filename for the model.
    """
    model_path = os.path.join(MODELS_DIR, model_name)
    feat_path = os.path.join(MODELS_DIR, "feature_cols.joblib")
    
    joblib.dump(model, model_path)
    joblib.dump(feature_cols, feat_path)
    logger.info(f"Model saved to {model_path}")
