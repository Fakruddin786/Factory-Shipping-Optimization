import pandas as pd
from typing import Dict, Any, List, Tuple
from src.utils import setup_logger, haversine

logger = setup_logger(__name__)

def calculate_recommendation_score(dist_diff_km: float, orig_dist_km: float, 
                                   lead_time_diff: float, orig_lead_time: float, 
                                   profit_diff: float, orig_profit: float,
                                   speed_weight: float = 40.0, profit_weight: float = 30.0) -> Tuple[float, float, float]:
    """
    Calculates the multi-objective recommendation score dynamically based on user priority.
    
    Returns:
        Score, Confidence, Risk Score
    """
    # Speed (Lead Time) Improvement
    lead_time_improvement_pct = lead_time_diff / orig_lead_time if orig_lead_time > 0 else 0
    lead_time_score = min(max(lead_time_improvement_pct, -1), 1) * speed_weight
    
    # Profit Impact
    profit_impact_pct = profit_diff / abs(orig_profit) if orig_profit != 0 else 0
    profit_score = min(max(profit_impact_pct, -1), 1) * profit_weight
    
    # Risk calculation based on margins and delays
    margin_risk = 10 if profit_diff < 0 else 0
    delay_risk = 10 if lead_time_diff < 0 else 0
    risk_score = 20 - (margin_risk + delay_risk) # 20% max for low risk
    
    # Confidence Score (10% max)
    # Simple heuristic: higher if both profit and speed improve
    confidence_pts = 10 if (profit_diff > 0 and lead_time_diff > 0) else 5
    
    total_score = lead_time_score + profit_score + risk_score + confidence_pts
    
    # Scale to 0-100 overall percentage
    final_score = max(0, min(100, 50 + total_score)) # 50 is baseline
    confidence = (confidence_pts / 10) * 100
    
    return final_score, confidence, risk_score

def get_recommendations(current_factory: str, customer_lat: float, customer_lon: float, 
                        quantity: int, unit_price: float, unit_cost: float, ship_mode: str, 
                        region: str, model: Any, feature_cols: List[str], factories_dict: Dict,
                        speed_weight: float = 40.0, profit_weight: float = 30.0) -> List[Dict]:
    """
    Generates Top-5 alternative factory recommendations.
    """
    try:
        recommendations = []
        base_lat = factories_dict[current_factory]["Lat"]
        base_lon = factories_dict[current_factory]["Lon"]
        base_dist = haversine(base_lat, base_lon, customer_lat, customer_lon)
        
        multiplier = 0.5 if ship_mode == "Same-Day" else (0.8 if ship_mode == "Express" else 1.0)
        
        base_sales = quantity * unit_price
        base_cost = quantity * unit_cost
        base_shipping_cost = base_dist * 0.05 * quantity * multiplier
        base_gross_profit = base_sales - base_cost - base_shipping_cost
        
        def prep_features(dist, sales, gp):
            d = {'Distance_KM': dist, 'Sales': sales, 'Gross Profit': gp, 
                 'Units': quantity, 'Month': 1, 'Quarter': 1, # Dummy temporal for simulation
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
            
        base_features = prep_features(base_dist, base_sales, base_gross_profit)
        base_lead_time = model.predict(base_features)[0]
        
        for factory_name, coords in factories_dict.items():
            if factory_name == current_factory:
                continue
                
            new_dist = haversine(coords["Lat"], coords["Lon"], customer_lat, customer_lon)
            new_shipping_cost = new_dist * 0.05 * quantity * multiplier
            new_gross_profit = base_sales - base_cost - new_shipping_cost
            
            new_features = prep_features(new_dist, base_sales, new_gross_profit)
            new_lead_time = model.predict(new_features)[0]
            
            dist_diff = base_dist - new_dist
            lead_time_diff = base_lead_time - new_lead_time
            profit_diff = new_gross_profit - base_gross_profit
            
            score, confidence, risk_score = calculate_recommendation_score(
                dist_diff, base_dist, lead_time_diff, base_lead_time, profit_diff, base_gross_profit,
                speed_weight, profit_weight
            )
            
            recommendations.append({
                "Factory": factory_name,
                "Distance_KM": new_dist,
                "Distance_Reduction_KM": dist_diff,
                "Predicted_Lead_Time": new_lead_time,
                "Lead_Time_Reduction": lead_time_diff,
                "Expected_Savings": profit_diff,
                "Risk_Score": risk_score,
                "Score": score,
                "Confidence": confidence
            })
            
        recommendations.sort(key=lambda x: x["Score"], reverse=True)
        return recommendations[:5] # Return Top 5
    except Exception as e:
        logger.error(f"Error generating recommendations: {e}")
        return []
