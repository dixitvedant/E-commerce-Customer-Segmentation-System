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
st.set_page_config(page_title="Smart Clustering System", layout="wide")

st.title("SmartCart: E-Commerce Customer Segmentation System")
st.write("Generating precise customer personas using multi-dimensional behavioral analysis.")

uploaded_file = st.sidebar.file_uploader("Upload your dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader(" Raw Dataset Preview")
    st.dataframe(df.head())

    # ======================================================
    # DATA PREPROCESSING
    # ======================================================
    if "Income" in df.columns:
        df["Income"] = df["Income"].fillna(df["Income"].median())

    if "Year_Birth" in df.columns:
        df["Age"] = 2026 - df["Year_Birth"]

    if "Dt_Customer" in df.columns:
        df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"], dayfirst=True)
        reference_date = df["Dt_Customer"].max()
        df["Customer_Tenure_Days"] = (reference_date - df["Dt_Customer"]).dt.days

    spending_cols = ["MntWines", "MntFruits", "MntMeatProducts", "MntFishProducts", "MntSweetProducts", "MntGoldProds"]
    available_spending = [col for col in spending_cols if col in df.columns]
    if available_spending:
        df["Total_Spending"] = df[available_spending].sum(axis=1)

    if "Kidhome" in df.columns and "Teenhome" in df.columns:
        df["Total_Children"] = df["Kidhome"] + df["Teenhome"]

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

    if "Age" in df_for_model.columns:
        df_for_model = df_for_model[df_for_model["Age"] < 90]
    if "Income" in df_for_model.columns:
        df_for_model = df_for_model[df_for_model["Income"] < 600000]

    imputer = SimpleImputer(strategy="median")
    imputed_data = imputer.fit_transform(df_for_model)
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(imputed_data)

    pca = PCA(n_components=3)
    pca_data = pca.fit_transform(scaled_data)

    # ======================================================
    #  EVALUATION SECTION (Step 1)
    # ======================================================
    st.subheader(" Step 1: Model Evaluation")
    col_eval_1, col_eval_2 = st.columns([2, 1])

    with col_eval_1:
        k_range = range(2, 11)
        wcss, sil_scores = [], []
        for k in k_range:
            km = KMeans(n_clusters=k, random_state=42, n_init=20)
            labels = km.fit_predict(pca_data)
            wcss.append(km.inertia_)
            sil_scores.append(silhouette_score(pca_data, labels))

        fig_eval, ax1 = plt.subplots(figsize=(10, 5))
        ax1.set_xlabel('Number of Clusters (K)')
        ax1.set_ylabel('WCSS (Elbow)', color='tab:blue')
        ax1.plot(k_range, wcss, marker='o', color='tab:blue', linewidth=2)
        
        ax2 = ax1.twinx()
        ax2.set_ylabel('Silhouette Score', color='tab:red')
        ax2.plot(k_range, sil_scores, marker='s', color='tab:red', linewidth=2)
        st.pyplot(fig_eval)

    with col_eval_2:
        st.write("### Select Clusters")
        # Defaulting to 4 as per your requirement
        k_selected = st.slider("Number of Clusters (K)", 2, 10, 4)
        st.info("The Elbow method helps visualize the point where adding more clusters provides diminishing returns.")

    # ======================================================
    # APPLY CLUSTERING & VISUALIZATION (Steps 2 & 3)
    # ======================================================
    kmeans = KMeans(n_clusters=k_selected, random_state=42, n_init=20)
    clusters = kmeans.fit_predict(pca_data)
    df_for_model["Cluster"] = clusters

    st.subheader("Step 2: Customer Segment Visualizations")
    custom_colors = ['red', 'green', 'yellow', 'blue', 'purple', 'orange', 'cyan', 'magenta', 'brown', 'black']
    cmap_custom = ListedColormap(custom_colors[:k_selected])

    col_plot1, col_plot2 = st.columns(2)
    with col_plot1:
        st.write("####  Spending vs Income")
        fig_comb, ax_comb = plt.subplots(figsize=(10, 7))
        scatter = ax_comb.scatter(df_for_model["Total_Spending"], df_for_model["Income"],
        c=df_for_model["Cluster"], cmap=cmap_custom, alpha=0.7, edgecolors='w')
        ax_comb.set_xlabel("Total Spending ($)")
        ax_comb.set_ylabel("Annual Income ($)")
        ax_comb.legend(handles=scatter.legend_elements()[0], labels=[f"Cluster {i}" for i in range(k_selected)])
        st.pyplot(fig_comb)

    with col_plot2:
        st.write("####  2D PCA View")
        fig_2d = plt.figure(figsize=(10, 7))
        plt.scatter(pca_data[:, 0], pca_data[:, 1], c=clusters, cmap=cmap_custom, alpha=0.7, edgecolors='w')
        plt.xlabel("PC 1")
        plt.ylabel("PC 2")
        st.pyplot(fig_2d)

    st.write("#### 3D PCA Segment Visualization")
    fig_3d = plt.figure(figsize=(10, 7))
    ax = fig_3d.add_subplot(111, projection='3d')
    ax.scatter(pca_data[:, 0], pca_data[:, 1], pca_data[:, 2], c=clusters, cmap=cmap_custom)
    st.pyplot(fig_3d)

    # Cluster Summary Table
    st.subheader(" Cluster Profiling")
    summary = df_for_model.groupby("Cluster").mean()
    st.dataframe(summary.style.background_gradient(cmap='YlGn'))

    # ======================================================
    # PERSONA INTERPRETATION
    # ======================================================
    st.subheader(" Step 3: Detailed Cluster Personas")
    summary = df_for_model.groupby("Cluster").mean()
    
    for i in range(k_selected):
        row = summary.loc[i]
        income_lvl = "High 💰" if row['Income'] > summary['Income'].mean() else "Moderate/Low 📉"
        spend_lvl = "Big Spender 🔥" if row['Total_Spending'] > summary['Total_Spending'].mean() else "Conservative 🏷️"
        
        living_status = "Living Alone 🏠"
        if "Living_With_Partner" in summary.columns:
            if row['Living_With_Partner'] > 0.5:
                living_status = "Living with Partner 👨‍👩‍👦"
        
        edu_cols = [c for c in summary.columns if "Education" in c]
        best_edu = summary.columns[summary.loc[i, edu_cols].argmax()].replace("Education_", "") if edu_cols else "N/A"
        kids_info = "Family with Kids" if row.get('Total_Children', 0) >= 1 else "No Kids at Home"
        ratio = row['Total_Spending'] / (row['Income'] + 1)
        spending_efficiency = "High Value/Income Ratio" if ratio > (summary['Total_Spending'] / summary['Income']).mean() else "Low Value/Income Ratio"

        with st.expander(f"Identify: Cluster {i} (Color: {custom_colors[i].capitalize()})"):
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown("**Financial Profile**")
                st.write(f"Income: {income_lvl}")
                st.write(f"Spending: {spend_lvl}")
                st.write(f"Efficiency: {spending_efficiency}")
            with c2:
                st.markdown("**Lifestyle**")
                st.write(f"Status: {living_status}")
                st.write(f"Family: {kids_info}")
                st.write(f"Age: ~{int(row['Age'])} yrs")
            with c3:
                st.markdown("**Background**")
                st.write(f"Education: {best_edu}")
                st.write(f"Avg Tenure: {int(row.get('Customer_Tenure_Days', 0))} days")

            if row['Total_Spending'] > summary['Total_Spending'].mean():
                if living_status == "Living with Partner 👨‍👩‍👦":
                    st.info("**Strategy:** 'Premium Family Packages'. Focus on luxury household items.")
                else:
                    st.info("**Strategy:** 'Exclusive Elite'. Focus on individual luxury and premium travel.")
            elif row['Income'] > summary['Income'].mean() and row['Total_Spending'] < summary['Total_Spending'].mean():
                st.warning("**Strategy:** 'Untapped Wealth'. High income but low engagement.")
            else:
                st.success("**Strategy:** 'Budget & Volume'. Focus on discount bundles and daily essentials.")
                
    


else:
    st.info("👋 Upload your CSV to generate professional customer segments.")
