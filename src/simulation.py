import pandas as pd
from typing import Dict, Any, List
from src.utils import setup_logger, haversine

logger = setup_logger(__name__)

def simulate_reallocation(product: str, current_factory: str, new_factory: str, 
                          customer_lat: float, customer_lon: float, quantity: int, 
                          unit_price: float, unit_cost: float, ship_mode: str, 
                          region: str, model: Any, feature_cols: List[str], factories_dict: Dict) -> Dict[str, Any]:
    """
    Simulates the reallocation of an order from a current factory to a new factory.
    Returns impact metrics.
    """
    logger.info(f"Simulating reallocation from {current_factory} to {new_factory}")
    try:
        multiplier = 0.5 if ship_mode == "Same-Day" else (0.8 if ship_mode == "Express" else 1.0)
            
        def prep_features(dist, sales, gp):
            d = {'Distance_KM': dist, 'Sales': sales, 'Gross Profit': gp, 
                 'Units': quantity, 'Month': 1, 'Quarter': 1,
                 'Sales_Per_Unit': unit_price, 'Cost_Per_Unit': unit_cost,
                 'Profit_Margin': gp / sales if sales > 0 else 0}
            for col in feature_cols:
                if col not in d:
                    if col == f"Ship Mode_{ship_mode}":
                        d[col] = 1
                    elif col == f"Region_{region}":
                        d[col] = 1
                    else:
                        d[col] = 0
            df = pd.DataFrame([d])
            return df[feature_cols]

        # Current scenario
        curr_lat = factories_dict[current_factory]["Lat"]
        curr_lon = factories_dict[current_factory]["Lon"]
        curr_dist = haversine(curr_lat, curr_lon, customer_lat, customer_lon)
        
        sales = quantity * unit_price
        cost = quantity * unit_cost
        curr_shipping_cost = curr_dist * 0.05 * quantity * multiplier
        curr_gross_profit = sales - cost - curr_shipping_cost
        
        curr_features = prep_features(curr_dist, sales, curr_gross_profit)
        curr_lead_time = model.predict(curr_features)[0]
        
        # New scenario
        new_lat = factories_dict[new_factory]["Lat"]
        new_lon = factories_dict[new_factory]["Lon"]
        new_dist = haversine(new_lat, new_lon, customer_lat, customer_lon)
        
        new_shipping_cost = new_dist * 0.05 * quantity * multiplier
        new_gross_profit = sales - cost - new_shipping_cost
        
        new_features = prep_features(new_dist, sales, new_gross_profit)
        new_lead_time = model.predict(new_features)[0]
        
        profit_diff = new_gross_profit - curr_gross_profit
        lead_time_diff = curr_lead_time - new_lead_time
        
        return {
            "Distance_Difference_KM": new_dist - curr_dist,
            "Predicted_Lead_Time_Difference": -lead_time_diff, # negative means faster
            "Profit_Impact": profit_diff,
            "Risk_Impact": "Low" if profit_diff > 0 and lead_time_diff > 0 else "High",
            "New_Distance_KM": new_dist,
            "New_Lead_Time": new_lead_time,
            "New_Profit": new_gross_profit,
            "Improvement_Percentage_Speed": (lead_time_diff / curr_lead_time) * 100 if curr_lead_time > 0 else 0,
            "Improvement_Percentage_Profit": (profit_diff / abs(curr_gross_profit)) * 100 if curr_gross_profit != 0 else 0
        }
    except Exception as e:
        logger.error(f"Error in simulation: {e}")
        return {}
