import pandas as pd
from typing import Tuple, List
from src.utils import setup_logger

logger = setup_logger(__name__)

def generate_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate new features such as temporal components and unit metrics.
    
    Args:
        df: Cleaned pandas DataFrame.
        
    Returns:
        DataFrame with new engineered features.
    """
    logger.info("Generating features...")
    try:
        # Date Features
        df['Order Date'] = pd.to_datetime(df['Order Date'])
        df['Ship Date'] = pd.to_datetime(df['Ship Date'])
        
        df['Month'] = df['Order Date'].dt.month
        df['Year'] = df['Order Date'].dt.year
        df['Quarter'] = df['Order Date'].dt.quarter
        df['Weekday'] = df['Order Date'].dt.weekday
        
        # Metric Features
        df['Sales_Per_Unit'] = df['Sales'] / df['Units']
        df['Cost_Per_Unit'] = df['Cost'] / df['Units']
        
        # Ship Delay Feature (Actual ship date - order date vs Lead Time Days expected)
        # Assuming Lead_Time_Days is actual, we could model expected. Let's define Ship_Delay as 
        # actual transit days (Ship Date - Order Date) minus standard expected time based on mode.
        # But for this problem, Lead_Time_Days IS the target. We can compute an abstract Ship_Delay as noise.
        actual_days = (df['Ship Date'] - df['Order Date']).dt.days
        df['Ship_Delay'] = df['Lead_Time_Days'] - actual_days
        
        df['Profit_Margin'] = df['Gross Profit'] / df['Sales']
        df['Profit_Margin'] = df['Profit_Margin'].fillna(0)
        
        return df
    except Exception as e:
        logger.error(f"Error during feature generation: {e}")
        raise

def prepare_modeling_data(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    Encodes categorical features and prepares X, y for modeling.
    
    Args:
        df: Feature-engineered DataFrame.
        target_col: Name of the target variable column.
        
    Returns:
        X (features DataFrame), y (target Series), and list of feature column names.
    """
    logger.info("Preparing data for modeling...")
    try:
        # Select base features
        base_features = [
            'Distance_KM', 'Sales', 'Gross Profit', 'Units', 
            'Month', 'Quarter', 'Sales_Per_Unit', 'Cost_Per_Unit',
            'Profit_Margin'
        ]
        
        # One-hot encode Ship Mode and Region
        df_encoded = pd.get_dummies(df, columns=['Ship Mode', 'Region'], drop_first=True)
        
        features = base_features.copy()
        for col in df_encoded.columns:
            if col.startswith('Ship Mode_') or col.startswith('Region_'):
                features.append(col)
                
        X = df_encoded[features]
        y = df_encoded[target_col]
        
        logger.info(f"Final feature count: {len(features)}")
        return X, y, features
    except Exception as e:
        logger.error(f"Error preparing modeling data: {e}")
        raise
