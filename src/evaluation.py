import pandas as pd
import numpy as np
from typing import Dict, Any

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def mean_absolute_percentage_error(y_true: pd.Series, y_pred: np.ndarray) -> float:
    """
    Calculates the Mean Absolute Percentage Error (MAPE).
    
    Args:
        y_true: Ground truth target values.
        y_pred: Predicted target values.
        
    Returns:
        MAPE score.
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # Avoid division by zero
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def evaluate_model(y_true: pd.Series, y_pred: np.ndarray, model_name: str) -> Dict[str, Any]:
    """
    Evaluates a regression model using multiple metrics.
    
    Args:
        y_true: Ground truth target values.
        y_pred: Predicted target values.
        model_name: Name of the evaluated model.
        
    Returns:
        Dictionary containing RMSE, MAE, R2, and MAPE.
    """
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    mape = float(mean_absolute_percentage_error(y_true, y_pred))
    
    return {
        "Model": model_name,
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
        "MAPE": mape
    }

def extract_feature_importance(model: Any, feature_cols: list) -> pd.DataFrame:
    """
    Extracts feature importance from tree-based models if available.
    
    Args:
        model: Trained machine learning model.
        feature_cols: List of feature names.
        
    Returns:
        DataFrame containing feature importances.
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        df_imp = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)
        return df_imp
    return pd.DataFrame()
