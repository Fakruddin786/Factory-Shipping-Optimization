# Factory Reallocation & Shipping Optimization Recommendation System

## 1. Background
Nassau Candy distributes a wide range of confectionery products across multiple regions. Currently, each product is permanently assigned to a specific manufacturing factory based on traditional business rules. Although this method has worked historically, it often results in inefficient shipping routes, higher logistics costs, and increased delivery times.

Business leaders need a system that can answer questions such as:
- Can shipping be improved by changing the manufacturing factory?
- Which factory is best suited for a specific product?
- How will factory reassignment affect shipping time and profitability?

This project transforms historical shipping data into an intelligent decision-support system capable of recommending optimal factory assignments.

## 2. Problem Statement
The current manufacturing strategy relies on static factory assignments, creating several operational challenges:
- Products may be manufactured far from customer demand.
- Shipping distances become unnecessarily long.
- Delivery lead times increase.
- Logistics costs reduce overall profitability.
- Factory assignment decisions cannot be evaluated before implementation.

The organization requires a recommendation engine capable of:
- Simulating different factory allocation scenarios.
- Predicting shipping performance for each scenario.
- Measuring business impact before making operational changes.
- Recommending the most efficient factory for every product.

## 3. Project Structure
- `app/`: Multi-page Streamlit dashboard providing executive overviews, simulations, and recommendations.
- `src/`: Core backend modules for data engineering, model training (XGBoost/LightGBM/CatBoost), clustering, and optimization.
- `data/`: Raw and processed dataset storage, including the newly added `orders.csv` (raw historical orders) and `route_clusters.csv` for clustered route performance.
- `models/`: Saved machine learning pipelines.
- `main.py`: Orchestrator script to generate data, train models, and run the optimization pipeline.

## 4. How to Run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the full data and machine learning pipeline:
   ```bash
   python main.py
   ```
3. Launch the Executive Dashboard:
   ```bash
   streamlit run app/Home.py
   ```
