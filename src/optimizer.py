import numpy as np
from scipy.optimize import linprog
from typing import Dict, List, Any
from src.utils import setup_logger

logger = setup_logger(__name__)

def optimize_factory_allocation(demand: List[float], capacities: List[float], costs: np.ndarray) -> Dict[str, Any]:
    """
    Globally optimizes the allocation of products to factories minimizing total cost
    subject to capacity and demand constraints using SciPy's linprog.
    
    Args:
        demand: List of demand quantities for each region/product.
        capacities: List of capacity limits for each factory.
        costs: 2D array of costs where rows are factories and columns are demand nodes.
        
    Returns:
        Dictionary containing the optimal assignment matrix and total cost.
    """
    logger.info("Starting global factory allocation optimization...")
    try:
        num_factories = len(capacities)
        num_demands = len(demand)
        
        # Flatten the cost matrix to 1D for linprog
        c = costs.flatten()
        
        # 1. Supply Constraints (Sum of allocations from factory <= capacity)
        # We need num_factories constraints. Each row in A_ub corresponds to a factory.
        A_ub = np.zeros((num_factories, num_factories * num_demands))
        for i in range(num_factories):
            A_ub[i, i*num_demands : (i+1)*num_demands] = 1
        b_ub = np.array(capacities)
        
        # 2. Demand Constraints (Sum of allocations to a demand node == demand)
        # We need num_demands constraints. Each row in A_eq corresponds to a demand node.
        A_eq = np.zeros((num_demands, num_factories * num_demands))
        for j in range(num_demands):
            for i in range(num_factories):
                A_eq[j, i*num_demands + j] = 1
        b_eq = np.array(demand)
        
        # Bounds: Variables must be >= 0
        bounds = [(0, None) for _ in range(num_factories * num_demands)]
        
        # Run linear programming optimization
        logger.info("Solving linear program...")
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
        
        if res.success:
            allocation = res.x.reshape((num_factories, num_demands))
            logger.info("Optimization successful.")
            return {
                "success": True,
                "total_cost": res.fun,
                "allocation": allocation,
                "message": res.message
            }
        else:
            logger.warning(f"Optimization failed: {res.message}")
            return {
                "success": False,
                "total_cost": None,
                "allocation": None,
                "message": res.message
            }
            
    except Exception as e:
        logger.error(f"Error during optimization: {e}")
        raise
