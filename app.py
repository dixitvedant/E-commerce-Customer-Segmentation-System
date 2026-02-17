
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.impute import SimpleImputer

# Page Configuration
st.set_page_config(page_title="Smart Clustering Project", layout="wide")

st.title("🛒 SmartClustering: E-Commerce Customer Segmentation")
st.write("Clustering Analysis with custom visualizations for Income, Spending, and Model Performance.")

uploaded_file = st.sidebar.file_uploader("Upload your dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("📊 Raw Dataset Preview")
    st.dataframe(df.head())

    # ======================================================
    # DATA PREPROCESSING
    # ======================================================
    # 1. Handle Missing Income
    if "Income" in df.columns:
        df["Income"] = df["Income"].fillna(df["Income"].median())

    # 2. Feature Engineering
    if "Year_Birth" in df.columns:
        df["Age"] = 2026 - df["Year_Birth"]

    if "Dt_Customer" in df.columns:
        df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"], dayfirst=True)
        reference_date = df["Dt_Customer"].max()
        df["Customer_Tenure_Days"] = (reference_date - df["Dt_Customer"]).dt.days

    # 3. Total Spending Calculation
    spending_cols = ["MntWines", "MntFruits", "MntMeatProducts", "MntFishProducts", "MntSweetProducts", "MntGoldProds"]
    available_spending = [col for col in spending_cols if col in df.columns]
    if available_spending:
        df["Total_Spending"] = df[available_spending].sum(axis=1)

    # 4. Total Children
    if "Kidhome" in df.columns and "Teenhome" in df.columns:
        df["Total_Children"] = df["Kidhome"] + df["Teenhome"]

    # 5. Categorical Cleaning
    if "Education" in df.columns:
        df["Education"] = df["Education"].replace({
            "Basic": "Undergraduate", "2n Cycle": "Undergraduate",
            "Graduation": "Graduate", "Master": "Postgraduate", "PhD": "Postgraduate"
        })

    if "Marital_Status" in df.columns:
        df["Living_With"] = df["Marital_Status"].replace({
            "Married": "Partner", "Together": "Partner", "Single": "Alone",
            "Divorced": "Alone", "Widow": "Alone", "Absurd": "Alone", "YOLO": "Alone"
        })

    # Create a copy for analysis before encoding/scaling
    df_analysis = df.copy()

    # Drop Columns not used in Scaling
    cols_to_drop = ["ID", "Year_Birth", "Marital_Status", "Kidhome", "Teenhome", "Dt_Customer"] + spending_cols
    df_for_model = df.drop(columns=[col for col in cols_to_drop if col in df.columns])

    # ======================================================
    # ENCODING & SCALING
    # ======================================================
    cat_cols = [c for c in ["Education", "Living_With"] if c in df_for_model.columns]
    if cat_cols:
        ohe = OneHotEncoder(sparse_output=False)
        encoded = ohe.fit_transform(df_for_model[cat_cols])
        encoded_df = pd.DataFrame(encoded, columns=ohe.get_feature_names_out(cat_cols), index=df_for_model.index)
        df_for_model = pd.concat([df_for_model.drop(columns=cat_cols), encoded_df], axis=1)

    # Filter Outliers
    if "Age" in df_for_model.columns:
        df_for_model = df_for_model[df_for_model["Age"] < 90]
    if "Income" in df_for_model.columns:
        df_for_model = df_for_model[df_for_model["Income"] < 600000]

    # Impute and Scale
    imputer = SimpleImputer(strategy="median")
    imputed_data = imputer.fit_transform(df_for_model)
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(imputed_data)

    # PCA
    pca = PCA(n_components=3)
    pca_data = pca.fit_transform(scaled_data)

    st.success("Preprocessing & PCA Complete!")

    # ======================================================
    # 📈 ELBOW METHOD VS SILHOUETTE SCORE GRAPH
    # ======================================================
    st.subheader("📉 Model Evaluation: Elbow vs Silhouette")
    
    k_range = range(2, 11)
    wcss = []
    sil_scores = []

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=20)
        labels = km.fit_predict(pca_data)
        wcss.append(km.inertia_)
        sil_scores.append(silhouette_score(pca_data, labels))

    fig_eval, ax1 = plt.subplots(figsize=(10, 5))

    # Plot WCSS (Elbow)
    color1 = 'tab:blue'
    ax1.set_xlabel('Number of Clusters (K)')
    ax1.set_ylabel('WCSS (Elbow)', color=color1)
    ax1.plot(k_range, wcss, marker='o', color=color1, linewidth=2, label='WCSS')
    ax1.tick_params(axis='y', labelcolor=color1)

    # Create twin axis for Silhouette
    ax2 = ax1.twinx()
    color2 = 'tab:red'
    ax2.set_ylabel('Silhouette Score', color=color2)
    ax2.plot(k_range, sil_scores, marker='s', color=color2, linewidth=2, label='Silhouette')
    ax2.tick_params(axis='y', labelcolor=color2)

    plt.title("Elbow Method vs Silhouette Score Comparison")
    st.pyplot(fig_eval)

    # ======================================================
    # CLUSTER SELECTION
    # ======================================================
    k_selected = st.slider("Select K (Number of Clusters)", 2, 10, 3)
    kmeans = KMeans(n_clusters=k_selected, random_state=42, n_init=20)
    clusters = kmeans.fit_predict(pca_data)
    
    # Map clusters back to analysis dataframe
    # Ensuring indices match after potential outlier removal
    df_for_model["Cluster"] = clusters
    
    # ======================================================
    # 🔗 COMBINED GRAPH: SPENDING vs INCOME
    # ======================================================
    st.subheader("🔗 Combined Graph: Total Spending vs Income")
    st.write(f"Visualizing clusters using Red, Green, and Yellow for the first 3 groups.")

    # Custom Color Palette (Requested: Red, Green, Yellow)
    # We add more colors in case K > 3
    custom_colors = ['red', 'green', 'yellow', 'blue', 'purple', 'orange', 'cyan', 'magenta', 'brown', 'black']
    cmap_custom = ListedColormap(custom_colors[:k_selected])

    fig_comb = plt.figure(figsize=(10, 6))
    
    # Scatter plot: X = Total Spending, Y = Income
    scatter = plt.scatter(
        df_for_model["Total_Spending"], 
        df_for_model["Income"], 
        c=df_for_model["Cluster"], 
        cmap=cmap_custom,
        alpha=0.7,
        edgecolors='w'
    )
    
    plt.xlabel("Total Spending")
    plt.ylabel("Annual Income")
    plt.title("Customer Segments: Total Spending vs Income")
    plt.legend(handles=scatter.legend_elements()[0], labels=[f"Cluster {i}" for i in range(k_selected)])
    plt.grid(True, linestyle='--', alpha=0.6)
    st.pyplot(fig_comb)

    # ======================================================
    # PCA VISUALIZATIONS
    # ======================================================
    col_pca1, col_pca2 = st.columns(2)
    
    with col_pca1:
        st.write("#### 2D PCA View")
        fig_2d = plt.figure()
        plt.scatter(pca_data[:, 0], pca_data[:, 1], c=clusters, cmap='viridis')
        plt.xlabel("PCA 1")
        plt.ylabel("PCA 2")
        st.pyplot(fig_2d)

    with col_pca2:
        st.write("#### 3D PCA View")
        fig_3d = plt.figure()
        ax = fig_3d.add_subplot(111, projection='3d')
        ax.scatter(pca_data[:, 0], pca_data[:, 1], pca_data[:, 2], c=clusters, cmap='viridis')
        ax.set_xlabel("PCA 1")
        ax.set_ylabel("PCA 2")
        ax.set_zlabel("PCA 3")
        st.pyplot(fig_3d)

    # Cluster Summary Table
    st.subheader("📊 Cluster Profiling")
    summary = df_for_model.groupby("Cluster").mean()
    st.dataframe(summary.style.background_gradient(cmap='YlGn'))

else:
    st.info("Please upload your CSV file in the sidebar to start.")
