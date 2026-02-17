# 🛒 E-commerce Customer Segmentation System using Machine Learning

This project focuses on segmenting e-commerce customers based on purchasing behavior, income, and demographics using unsupervised machine learning techniques.

The goal is to identify meaningful customer groups that can help businesses design targeted marketing strategies and improve customer retention.

---

## 📌 Project Overview

Customer segmentation was performed using clustering algorithms after extensive data preprocessing and feature engineering.

The workflow includes:

- Data Cleaning
- Feature Engineering
- Outlier Removal
- Encoding Categorical Variables
- Feature Scaling
- Dimensionality Reduction (PCA)
- Optimal Cluster Selection
- Clustering & Business Interpretation

---

## 🛠️ Tech Stack

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Kneed

---

## ⚙️ Feature Engineering

The following new features were created:

- **Age** (calculated from Year of Birth)
- **Customer Tenure** (days since joining)
- **Total Spending** (sum of all product categories)
- **Total Children** (Kidhome + Teenhome)

Education and Marital Status were simplified into broader categories to improve clustering performance.

---

## 📊 Data Preprocessing

- Missing values in Income were filled using median.
- Outliers were removed (Age > 90, Income > 600,000).
- Categorical variables were encoded using OneHotEncoder.
- Features were scaled using StandardScaler.

---

## 📉 Dimensionality Reduction

Principal Component Analysis (PCA) was applied to reduce dimensionality and visualize clusters in 3D space.

---

## 🔎 Finding Optimal Clusters

Two techniques were used:

1. Elbow Method (WCSS)
2. Silhouette Score

Both methods suggested an optimal number of clusters.

---

## 🤖 Clustering Algorithms Used

- K-Means Clustering
- Agglomerative (Hierarchical) Clustering

Clusters were visualized using 3D PCA projection.

---

## 📈 Key Insights

The model identified distinct customer segments such as:

- High Income – High Spending (Premium Customers)
- Low Income – Low Spending
- Family-oriented Moderate Buyers
- High Income – Low Spending (Potential Growth Segment)

---

## 🎯 Business Impact

This segmentation can help:

- Design targeted marketing campaigns
- Improve customer retention
- Personalize offers
- Increase overall revenue




