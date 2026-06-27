import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import os
from src.config import FACTORIES, PRODUCT_DATA, CITIES, SHIP_MODES, RAW_DATA_DIR
from src.utils import setup_logger, haversine

logger = setup_logger(__name__)

def generate_synthetic_data(num_records=5000) -> pd.DataFrame:
    """Generates synthetic order data and saves it to raw directory."""
    logger.info(f"Generating {num_records} synthetic records...")
    np.random.seed(42)
    random.seed(42)
    
    data = []
    start_date = datetime(2023, 1, 1)
    
    for i in range(num_records):
        order_date = start_date + timedelta(days=random.randint(0, 365))
        product_name = random.choice(list(PRODUCT_DATA.keys()))
        prod_info = PRODUCT_DATA[product_name]
        factory = prod_info["Factory"]
        factory_lat = FACTORIES[factory]["Lat"]
        factory_lon = FACTORIES[factory]["Lon"]
        
        cust_info = random.choice(CITIES)
        cust_lat, cust_lon = cust_info["Lat"], cust_info["Lon"]
        
        distance_km = haversine(factory_lat, factory_lon, cust_lat, cust_lon)
        ship_mode = random.choice(SHIP_MODES)
        
        if ship_mode == "Same-Day":
            base_lead_time, multiplier = random.uniform(0.5, 1.0), 0.5
        elif ship_mode == "Express":
            base_lead_time, multiplier = random.uniform(1.0, 2.0), 0.8
        else: # Standard
            base_lead_time, multiplier = random.uniform(2.0, 4.0), 1.0
            
        distance_delay = (distance_km / 500.0) * multiplier
        noise = random.uniform(0.0, 1.5)
        
        lead_time_days = base_lead_time + distance_delay + noise
        ship_date = order_date + timedelta(days=lead_time_days)
        
        units = random.randint(50, 5000)
        unit_cost = random.uniform(5.0, 20.0)
        margin_mult = random.uniform(1.3, 2.0)
        unit_price = unit_cost * margin_mult
        
        cost = units * unit_cost
        sales = units * unit_price
        
        # Reduced shipping cost multiplier to ensure realistic positive gross margins
        shipping_cost = distance_km * 0.002 * units * multiplier
        gross_profit = sales - cost - shipping_cost
        
        data.append({
            "Order ID": f"ORD-{random.randint(100000, 999999)}",
            "Order Date": order_date.strftime("%Y-%m-%d"),
            "Ship Date": ship_date.strftime("%Y-%m-%d"),
            "Ship Mode": ship_mode,
            "Customer ID": f"CUST-{random.randint(1000, 9999)}",
            "City": cust_info["City"],
            "State/Province": cust_info["State"],
            "Region": cust_info["Region"],
            "Division": prod_info["Division"],
            "Product ID": prod_info["ID"],
            "Product Name": product_name,
            "Sales": round(sales, 2),
            "Units": units,
            "Cost": round(cost, 2),
            "Gross Profit": round(gross_profit, 2),
            "Factory": factory,
            "Distance_KM": round(distance_km, 2),
            "Lead_Time_Days": round(lead_time_days, 2),
            "Factory_Lat": factory_lat,
            "Factory_Lon": factory_lon,
            "Customer_Lat": cust_lat,
            "Customer_Lon": cust_lon
        })
        
    df = pd.DataFrame(data)
    
    # Missing values logic
    for _ in range(int(num_records * 0.01)):
        df.loc[random.randint(0, num_records-1), 'Gross Profit'] = np.nan
        
    # Duplicates logic
    df = pd.concat([df, df.sample(int(num_records * 0.005))], ignore_index=True)
    
    output_path = os.path.join(RAW_DATA_DIR, "orders.csv")
    df.to_csv(output_path, index=False)
    logger.info(f"Dataset generated and saved to {output_path}")
    return df
