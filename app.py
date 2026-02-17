import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

st.set_page_config(page_title="E-Commerce Customer Segmentation", layout="wide")

st.title("🛒 E-Commerce Customer Segmentation System")
st.write("Customer Segmentation using Unsupervised Machine Learning (KMeans + PCA)")

# -------------------------------
# File Upload
# -------------------------------
uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("📊 Raw Dataset")
    st.dataframe(df.head())

    # -------------------------------
    # Data Preprocessing
    # -------------------------------
    st.subheader("⚙️ Data Preprocessing")

    # Drop non-numeric columns automatically
    numeric_df = df.select_dtypes(include=np.number)

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(numeric_df)

    st.success("Data Scaled Successfully!")

    # -------------------------------
    # Elbow Method
    # -------------------------------
    st.subheader("📈 Elbow Method")

    wcss = []
    k_range = range(2, 11)

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(scaled_data)
        wcss.append(kmeans.inertia_)

    fig1 = plt.figure()
    plt.plot(k_range, wcss, marker='o')
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("WCSS")
    plt.title("Elbow Method")
    st.pyplot(fig1)

    # -------------------------------
    # Select K
    # -------------------------------
    st.subheader("🔢 Select Number of Clusters")
    k_selected = st.slider("Choose K", 2, 10, 4)

    # Apply KMeans
    kmeans = KMeans(n_clusters=k_selected, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(scaled_data)

    df["Cluster"] = clusters

    # -------------------------------
    # Silhouette Score
    # -------------------------------
    score = silhouette_score(scaled_data, clusters)
    st.write(f"📌 Silhouette Score: **{score:.3f}**")

    # -------------------------------
    # PCA Visualization
    # -------------------------------
    st.subheader("📉 PCA 2D Visualization")

    pca = PCA(n_components=2)
    pca_components = pca.fit_transform(scaled_data)

    pca_df = pd.DataFrame(data=pca_components, columns=["PC1", "PC2"])
    pca_df["Cluster"] = clusters

    fig2 = plt.figure()
    for cluster in np.unique(clusters):
        subset = pca_df[pca_df["Cluster"] == cluster]
        plt.scatter(subset["PC1"], subset["PC2"], label=f"Cluster {cluster}")

    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend()
    plt.title("Customer Segments (PCA)")
    st.pyplot(fig2)

    # -------------------------------
    # Cluster Summary
    # -------------------------------
    st.subheader("📊 Cluster Summary")

    summary = df.groupby("Cluster").mean()
    st.dataframe(summary)

    st.success("Segmentation Completed Successfully!")

else:
    st.info("Please upload a CSV file to begin.")
