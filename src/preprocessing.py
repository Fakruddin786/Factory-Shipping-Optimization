import pandas as pd
import numpy as np
import os
from src.utils import setup_logger
from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, FACTORIES, PRODUCT_DATA, CITIES
from src.utils import haversine

logger = setup_logger(__name__)

def enrich_data(df: pd.DataFrame) -> pd.DataFrame:
    """Enrich the dataset with missing critical columns if needed."""
    logger.info("Enriching data with necessary pipeline columns if missing...")
    
    if 'Lead_Time_Days' not in df.columns and 'Order Date' in df.columns and 'Ship Date' in df.columns:
        logger.info("Calculating Lead_Time_Days...")
        order_dates = pd.to_datetime(df['Order Date'], format='mixed', dayfirst=True)
        ship_dates = pd.to_datetime(df['Ship Date'], format='mixed', dayfirst=True)
        df['Lead_Time_Days'] = (ship_dates - order_dates).dt.days

    if 'Factory' not in df.columns and 'Product Name' in df.columns:
        logger.info("Mapping Product Name to Factory...")
        def get_factory_info(prod_name):
            if prod_name in PRODUCT_DATA:
                factory = PRODUCT_DATA[prod_name]['Factory']
                return factory, FACTORIES[factory]['Lat'], FACTORIES[factory]['Lon']
            fallback = list(FACTORIES.keys())[0]
            return fallback, FACTORIES[fallback]['Lat'], FACTORIES[fallback]['Lon']
            
        res = df['Product Name'].apply(get_factory_info)
        df['Factory'] = res.apply(lambda x: x[0])
        df['Factory_Lat'] = res.apply(lambda x: x[1])
        df['Factory_Lon'] = res.apply(lambda x: x[2])
        
    if 'Customer_Lat' not in df.columns and 'City' in df.columns:
        logger.info("Mapping City to Customer_Lat/Lon...")
        city_coords = {c['City']: (c['Lat'], c['Lon']) for c in CITIES}
        def get_cust_coords(city):
            if city in city_coords:
                return city_coords[city]
            return 39.8283, -98.5795 # Default US center
        res = df['City'].apply(get_cust_coords)
        df['Customer_Lat'] = res.apply(lambda x: x[0])
        df['Customer_Lon'] = res.apply(lambda x: x[1])
        
    if 'Distance_KM' not in df.columns and 'Factory_Lat' in df.columns and 'Customer_Lat' in df.columns:
        logger.info("Calculating Distance_KM...")
        df['Distance_KM'] = df.apply(lambda row: haversine(row['Factory_Lat'], row['Factory_Lon'], row['Customer_Lat'], row['Customer_Lon']), axis=1)

    return df

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
    
    df = enrich_data(df)
    
    df_clean = clean_data(df)
    
    processed_path = os.path.join(PROCESSED_DATA_DIR, "cleaned_orders.csv")
    df_clean.to_csv(processed_path, index=False)
    logger.info(f"Saved cleaned data to {processed_path}")
    
    return df_clean
