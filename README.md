SmartCart: AI-Powered Customer Segmentation
Live Application: https://e-commerce-customer-segmentation-systemgit-qcc6eblw3h86qjkinpo.streamlit.app/

Built with: Python, Scikit-Learn, Streamlit, and PCA.

Project Vision
This project transforms raw e-commerce transactional data into high-level business intelligence. By leveraging Unsupervised Machine Learning, the system automatically categorizes customers into distinct groups based on their spending habits, income levels, and demographic profiles.

This allows businesses to stop "guessing" and start targeting the right customers with the right offers.

Key Features
1. Interactive Dashboard
Dynamic File Upload: Users can upload any CSV following the SmartCart schema.

Real-Time K-Selection: An interactive slider to adjust the number of clusters and instantly see the impact on data distribution.

2. Dual-Metric Model Evaluation
To ensure the mathematical validity of the clusters, the app provides a side-by-side comparison of:

The Elbow Method (WCSS): Identifying the point where adding more clusters yields diminishing returns.

Silhouette Score: Measuring how similar an object is to its own cluster compared to other clusters.

3. High-Dimensional Visualization
PCA (Principal Component Analysis): Reduces complex data dimensions into 2D and 3D space for human interpretation.

Combined Analysis Graph: A custom scatter plot with Total Spending (X-axis) and Income (Y-axis) using a specific Red-Green-Yellow color palette to identify "Premium," "Average," and "Budget" segments.

Tech Stack & Methodology
Data Engineering
Feature Creation: Derived Age, Customer_Tenure, and Total_Spending.

Simplification: Grouped complex Education and Marital Status categories into logical business segments (e.g., "Partner" vs "Alone").

Pipeline: Implemented Median Imputation, Standard Scaling, and One-Hot Encoding.

Machine Learning
Algorithm: K-Means Clustering,Hierarchichal Clustering

Dimensionality Reduction: PCA (3 Components).

Metric Tracking: Silhouette Analysis and Inertia (WCSS).

 Strategic Business Insights
The model identifies four primary segments:

The Premium Segment: High Income, High Spending. High loyalty potential.

The Value Segment: Moderate Income, Moderate Spending. The bulk of the customer base.

The Growth Segment: High Income, Low Spending. High potential for upselling.

The Budget Segment: Low Income, Low Spending. Best for clearance and discount offers.
