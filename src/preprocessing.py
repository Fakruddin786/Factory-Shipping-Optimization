import pandas as pd
import numpy as np
import os
from src.utils import setup_logger
from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR

logger = setup_logger(__name__)

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the raw dataframe by imputing missing values, removing duplicates,
    and removing outliers using IQR.
    
    Args:
        df: Raw pandas DataFrame.
        
    Returns:
        Cleaned pandas DataFrame.
    """
    try:
        logger.info(f"Initial shape: {df.shape}")
        
        # 1. Remove Duplicates
        df = df.drop_duplicates()
        logger.info(f"Shape after duplicate removal: {df.shape}")
        
        # 2. Handle Missing Values
        # Instead of dropping, we impute. For 'Gross Profit', we calculate it if Sales and Cost exist.
        if 'Gross Profit' in df.columns:
            missing_gp = df['Gross Profit'].isna()
            if missing_gp.any():
                logger.info(f"Imputing {missing_gp.sum()} missing Gross Profit values.")
                # Recalculate using Sales and Cost, approximating shipping cost if needed
                df.loc[missing_gp, 'Gross Profit'] = df.loc[missing_gp, 'Sales'] - df.loc[missing_gp, 'Cost']
        
        # Drop any remaining unhandled missing values
        df = df.dropna()
        logger.info(f"Shape after dropping remaining NAs: {df.shape}")
        
        # 3. Remove Outliers using IQR (e.g., on Lead_Time_Days)
        if 'Lead_Time_Days' in df.columns:
            Q1 = df['Lead_Time_Days'].quantile(0.25)
            Q3 = df['Lead_Time_Days'].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outlier_condition = (df['Lead_Time_Days'] < lower_bound) | (df['Lead_Time_Days'] > upper_bound)
            num_outliers = outlier_condition.sum()
            df = df[~outlier_condition]
            logger.info(f"Removed {num_outliers} outliers based on Lead_Time_Days. Final shape: {df.shape}")
            
        return df
    except Exception as e:
        logger.error(f"Error during data cleaning: {e}")
        raise

def run_preprocessing() -> pd.DataFrame:
    """
    Reads raw data, cleans it, and saves to processed folder.
    
    Returns:
        Cleaned dataframe.
    """
    raw_path = os.path.join(RAW_DATA_DIR, "orders.csv")
    if not os.path.exists(raw_path):
        # Fallback to base data dir
        raw_path = os.path.join(os.path.dirname(RAW_DATA_DIR), "orders.csv")
        
    logger.info(f"Reading raw data from {raw_path}")
    df = pd.read_csv(raw_path)
    
    df_clean = clean_data(df)
    
    processed_path = os.path.join(PROCESSED_DATA_DIR, "cleaned_orders.csv")
    df_clean.to_csv(processed_path, index=False)
    logger.info(f"Saved cleaned data to {processed_path}")
    
    return df_clean
