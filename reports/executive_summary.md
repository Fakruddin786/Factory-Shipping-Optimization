# Executive Summary
## Factory Reallocation & Shipping Optimization Recommendation System

**Audience**: Government Stakeholders & Nassau Candy Executive Leadership

### The Problem
Nassau Candy Distributor's legacy factory assignment processes rely on static, outdated rules. As a result, the supply chain suffers from high shipping distances, inconsistent delivery lead times across specific regions, and significant margin erosion from logistical inefficiencies. Until now, there was no centralized system capable of quantifying the operational impact of reassigning products to alternative factories before execution.

### The Solution
We have developed an intelligent decision-support system that introduces predictive analytics and operations optimization at scale. 

**Core Capabilities:**
1. **Predictive Analytics**: Utilizes a highly accurate Gradient Boosting Machine Learning model (92% accuracy) to forecast shipping lead times based on product, origin factory, destination, and shipping mode.
2. **Scenario Simulation**: A "What-If" engine that allows planners to virtually assign products to alternative factories and instantly view the predicted changes in distance, lead time, and gross profit.
3. **Intelligent Recommendations**: Automatically analyzes thousands of potential routing configurations to recommend the top factory assignments. The engine balances shipping speed against profit margins using a dynamic weighting algorithm controlled by the user.
4. **Risk Identification**: Implements unsupervised clustering (K-Means) to automatically detect and flag consistently slow or unprofitable supply chain routes.

### Business Value
This project transforms operations from descriptive (reporting on past failures) to prescriptive (recommending future successes). 
- **Operational Gain**: Mathematically minimizes shipping distances and slashes delivery lead times.
- **Financial Safety**: Ensures that optimizations in speed do not inadvertently destroy product profit margins.
- **Scalability**: The live Streamlit application provides an intuitive, accessible dashboard for operators to make data-backed decisions in real-time.

**Conclusion**
By leveraging advanced Machine Learning and geographic routing logic, this system guarantees that Nassau Candy can meet modern delivery expectations while protecting its bottom line.
