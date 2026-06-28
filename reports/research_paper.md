# Research Report: Optimization of Factory Reallocation and Shipping for Nassau Candy Distributor

## Abstract
This paper presents the development and evaluation of an intelligent decision-support system designed for Nassau Candy Distributor. The objective of the research is to overcome the inefficiencies associated with static factory-product assignment by implementing a multi-modal machine learning and operations research approach. By generating synthetic but operationally realistic historical data, we applied a comprehensive methodology comprising data preprocessing, feature engineering, predictive modeling (Linear Regression, Random Forest, Gradient Boosting, XGBoost, LightGBM, CatBoost), unsupervised clustering, and linear programming-based optimization. The result is a dynamic recommendation engine that simulates reallocation scenarios and predicts improvements in shipping lead time and profitability, accessible via a user-friendly Streamlit web interface.

## 1. Introduction
Historically, Nassau Candy has relied on static, rule-based systems to assign the production of specific confectionery items to designated factories. While simple to administer, this approach ignores geographic fluctuations in customer demand, resulting in suboptimal shipping routes, extended lead times, and inflated logistics costs. 

To address these challenges, we proposed a data-driven recommendation engine capable of simulating alternative manufacturing scenarios, predicting the logistical and financial impacts of those scenarios, and generating mathematically sound reallocation recommendations.

## 2. Methodology

### 2.1 Data Generation and Preprocessing
The research utilized a synthetic dataset of 10,000 order records (stored in `data/raw/orders.csv`) mapped to 5 distinct factories and 15 products. The preprocessing pipeline included:
- **Missing Value Imputation**: Gross Profit was derived using Sales and Manufacturing Cost rather than relying on listwise deletion.
- **Outlier Removal**: Extreme shipping lead times were removed using the Interquartile Range (IQR) method to ensure models were trained on representative operational data.
- **Feature Engineering**: Temporal features (month, quarter) and unit economics (cost per unit, sales per unit) were generated to provide the predictive models with rich context.

### 2.2 Predictive Modeling
The core objective of the predictive phase was to accurately estimate shipping lead times based on distance, order characteristics, and factory location. We evaluated six distinct algorithms:
1. Linear Regression (Baseline)
2. Random Forest Regressor
3. Gradient Boosting Regressor
4. XGBoost
5. LightGBM
6. CatBoost

**Evaluation Metrics**: Models were evaluated using Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), R-Squared (R²), and Mean Absolute Percentage Error (MAPE). 
**Results**: The Gradient Boosting Regressor consistently outperformed other models, achieving an R² of approximately 0.9267 and a MAPE of 10.42%, indicating highly reliable predictions suitable for operational deployment.

### 2.3 Unsupervised Clustering
To understand intrinsic network behaviors, we employed clustering algorithms (K-Means, DBSCAN, Hierarchical) to segment delivery routes. This analysis successfully identified high-density, high-efficiency shipping corridors versus isolated, high-cost routes, providing qualitative context to the quantitative predictions. The output of this stage is stored in the newly added `data/route_clusters.csv` dataset, which includes cluster assignments and slow-route identifiers.

### 2.4 Optimization and Recommendation Engine
The final phase translated predictive insights into actionable business logic. The recommendation algorithm scores alternative factory assignments using a weighted multi-objective function:
- **Speed (Lead Time Reduction)**: User-adjustable (Default 40%)
- **Profit (Expected Savings)**: User-adjustable (Default 30%)
- **Risk Avoidance**: 20% penalty for negative margins or extreme delays.
- **Confidence**: 10% based on the density of historical data near the destination.

Furthermore, a global linear programming optimizer (`scipy.optimize.linprog`) was implemented to solve the broader factory capacity allocation problem, ensuring that recommended reallocations do not violate manufacturing limits.

## 3. Results and Impact
The deployment of the system via the Streamlit dashboard successfully met all project deliverables. The What-If Scenario Analysis module demonstrated that dynamic reallocation can frequently reduce shipping distances by hundreds of kilometers, directly correlating to multi-day reductions in lead time and significant reductions in shipping costs.

By exposing the dynamic weights (Speed vs. Profit) to the end-user, the system affords supply chain managers the flexibility to adapt to changing macroeconomic conditions (e.g., prioritizing profit during high fuel-cost periods, or prioritizing speed during holiday rushes).

## 4. Conclusion
The Factory Reallocation & Shipping Optimization Recommendation System successfully transitions Nassau Candy from a static, reactive logistics model to a dynamic, predictive one. The integration of advanced gradient boosting algorithms with a flexible scoring heuristic provides a robust framework for continuous supply chain optimization. Future research should focus on integrating real-time weather and traffic APIs to further refine the lead time predictions.
