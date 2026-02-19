import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering

from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.impute import SimpleImputer

# ======================================================
# PAGE CONFIGURATION & PREMIUM THEME
# ======================================================
st.set_page_config(page_title="SmartCart AI | Enterprise Analytics", layout="wide", page_icon="🚀")

# Ultra-Premium CSS with Animations
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700;900&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main { background-color: #05070a; }
    
    @keyframes gradientBG {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    .hero-container {
        font-size: 26px;
        background: linear-gradient(-45deg, #00d4ff, #004e92, #9d50bb, #6e48aa);
        background-size: 400% 400%;
        animation: gradientBG 15s ease infinite;
        padding: 60px;
        border-radius: 25px;
        text-align: center;
        margin-bottom: 40px;
        box-shadow: 0 20px 40px rgba(0,0,0,0.4);
    }

    .hero-title {
        font-size: 70px;
        font-weight: 900;
        color: white;
        margin: 0;
        letter-spacing: -0px;
        text-shadow: 2px 2px 10px rgba(0,0,0,0.3);
    }

    .persona-card {
        background: rgba(23, 28, 36, 0.8);
        backdrop-filter: blur(10px);
        padding: 30px;
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 25px;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    }

    .persona-card:hover {
        transform: scale(1.02);
        border-color: #00d4ff;
        box-shadow: 0 15px 30px rgba(0, 212, 255, 0.2);
    }

    .stMetric {
        background: #11151c;
        border-radius: 15px;
        padding: 20px;
        border-bottom: 4px solid #00d4ff;
    }

    .strategy-pill {
        background: linear-gradient(90deg, #00d4ff, #004e92);
        color: white;
        padding: 6px 18px;
        border-radius: 30px;
        font-size: 13px;
        font-weight: 700;
        text-transform: uppercase;
    }
    </style>
    """, unsafe_allow_html=True)

# Impressive Hero Header
st.markdown("""
    <div class="hero-container">
        <p class="hero-title">SMARTCART AI</p>
        <p class="hero-subtitle">Next-Generation E-Commerce Segmentation System</p>
    </div>
    """, unsafe_allow_html=True)

# Sidebar UI
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=100)
    st.title("Control Center")
    uploaded_file = st.file_uploader(" Import Dataset (CSV)", type=["csv"])
    st.markdown("---")
    st.markdown("### Model Tuning")
    k_selected = st.slider("Target Segments (K)", 2, 10, 4)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    # ======================================================
    # DATA PREPROCESSING (Notebook Logic)
    # ======================================================
    # Handle Missing Values
    if "Income" in df.columns:
        df["Income"] = df["Income"].fillna(df["Income"].median())
    
    # Feature Engineering
    if "Year_Birth" in df.columns:
        df["Age"] = 2026 - df["Year_Birth"]
    
    if "Dt_Customer" in df.columns:
        df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"], dayfirst=True)
        reference_date = df["Dt_Customer"].max()
        df["Customer_tenure_days"] = (reference_date - df["Dt_Customer"]).dt.days
    
    spending_cols = ["MntWines", "MntFruits", "MntMeatProducts", "MntFishProducts", "MntSweetProducts", "MntGoldProds"]
    available_spending = [col for col in spending_cols if col in df.columns]
    df["Total_spendings"] = df[available_spending].sum(axis=1)
    
    if "Kidhome" in df.columns and "Teenhome" in df.columns:
        df["Total_children"] = df["Kidhome"] + df["Teenhome"]
    
    if "Education" in df.columns:
        df["Education"] = df["Education"].replace({
            "Basic": "Undergraduate", "2n Cycle": "Undergraduate", 
            "Graduation": "Graduate", "Master": "Postgraduate", "PhD": "Postgraduate"
        })
    
    if "Marital_Status" in df.columns:
        df["Living_with"] = df["Marital_Status"].replace({
            "Married": "Partner", "Together": "Partner", "Single": "Alone", 
            "Divorced": "Alone", "Widow": "Alone", "Absurd": "Alone", "YOLO": "Alone"
        })

    # Columns to drop for modeling
    drop_cols = ["ID", "Year_Birth", "Marital_Status", "Kidhome", "Teenhome", "Dt_Customer"] + spending_cols
    df_cleaned = df.drop(columns=[c for c in drop_cols if c in df.columns])
    
    # Ensure Income is numeric
    if "Income" in df_cleaned.columns:
        df_cleaned["Income"] = pd.to_numeric(df_cleaned["Income"], errors="coerce")
    
    # Remove Outliers (Crucial Notebook Logic)
    if "Age" in df_cleaned.columns:
        df_cleaned = df_cleaned[df_cleaned["Age"] < 90]
    if "Income" in df_cleaned.columns:
        df_cleaned = df_cleaned[df_cleaned["Income"] < 600000]
    
    # Synchronize original df with cleaned df for visualization
    df = df.loc[df_cleaned.index]

    # KPIs
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1: st.metric("Users", f"{len(df)} customers")
    with kpi2: st.metric("Features(CLEANED)", len(df_cleaned.columns))
    with kpi3: st.metric("Engine(ML ALGORITHM)", "KMeans+Agg")
    with kpi4: st.metric("Clusters", k_selected)

    # ======================================================
    # ML PIPELINE (PCA & KMeans)
    # ======================================================
    # One-Hot Encoding
    cat_cols = ["Education", "Living_with"]
    cat_cols_exist = [c for c in cat_cols if c in df_cleaned.columns]
    
    df_encoded = df_cleaned.copy()
    if cat_cols_exist:
        ohe = OneHotEncoder(sparse_output=False)
        enc_cols = ohe.fit_transform(df_cleaned[cat_cols_exist])
        enc_df = pd.DataFrame(enc_cols, columns=ohe.get_feature_names_out(cat_cols_exist), index=df_cleaned.index)
        df_encoded = pd.concat([df_cleaned.drop(columns=cat_cols_exist), enc_df], axis=1)

    # Impute remaining NaNs (if any) and Scale
    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(df_encoded)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    # PCA (3D)
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X_scaled)

    # Clustering(KMEANS)
    kmeans = KMeans(n_clusters=k_selected, random_state=42, n_init=20)
    df["Cluster"] = kmeans.fit_predict(X_pca)
    df_encoded["Cluster"] = df["Cluster"]
    #Clustering(AgglomerativeClustering)
    agg = AgglomerativeClustering(n_clusters=k_selected)
    df["Agg_Cluster"] = agg.fit_predict(X_pca)

    # Premium Color Palette
    custom_colors = ['#00d4ff', '#ff007f', '#7cfc00', '#ffea00', '#9d50bb', '#ff8c00', '#00ffa3', '#ff0000', '#ffffff', '#607d8b']
    cmap_custom = ListedColormap(custom_colors[:k_selected])

    # UI WORKFLOW
    tab_data, tab_eval, tab_viz, tab_profile, tab_persona = st.tabs([" DATASET", " DIAGNOSTICS", " SPATIAL MAPS", " CLUSTER PROFILES", " STRATEGY PERSONAS"])

    with tab_data:
        st.subheader("Enterprise Data Intelligence")
        st.dataframe(df.head(15), use_container_width=True)

    with tab_eval:
        st.subheader("Model Validation Metrics")
        k_range = range(2, 11)
        wcss, sil_scores = [], []
        for k in k_range:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(X_pca)
            wcss.append(km.inertia_)
            sil_scores.append(silhouette_score(X_pca, labels))

        fig_eval, ax1 = plt.subplots(figsize=(10, 4))
        plt.style.use('dark_background')
        fig_eval.patch.set_facecolor('#05070a')
        ax1.set_facecolor('#05070a')
        ax1.plot(k_range, wcss, marker='o', color='#00d4ff', linewidth=3, markersize=8)
        ax1.set_ylabel("Inertia (WCSS)", color='#00d4ff')
        ax1.set_title("Elbow Method and Silhouette score")
        ax2 = ax1.twinx()
        ax2.plot(k_range, sil_scores, marker='s', color='#ff007f', linewidth=3, markersize=8)
        ax2.set_ylabel("Silhouette Score", color='#ff007f')
        st.pyplot(fig_eval)

    with tab_viz:
        st.subheader("High-Dimensional Spatial Mapping")
        col_a, col_b = st.columns(2)
        with col_a:
            fig_is, ax_is = plt.subplots()
            fig_is.patch.set_facecolor('#05070a')
            ax_is.set_facecolor('#05070a')
            ax_is.scatter(df["Total_spendings"], df["Income"], c=df["Cluster"], cmap=cmap_custom, alpha=0.7, edgecolors='#000', s=50)
            ax_is.set_xlabel("Spending Strength", color='white')
            ax_is.set_ylabel("Income Tier", color='white')
            st.pyplot(fig_is)
        with col_b:
            fig_2d, ax_2d = plt.subplots()
            fig_2d.patch.set_facecolor('#05070a')
            ax_2d.set_facecolor('#05070a')
            ax_2d.scatter(X_pca[:, 0], X_pca[:, 1], c=df["Cluster"], cmap=cmap_custom, alpha=0.7, edgecolors='#000', s=50)
            ax_2d.set_xlabel("PCA component 1", color="white")
            ax_2d.set_ylabel("PCA component 2", color="white")
            st.pyplot(fig_2d)

        fig_3d = plt.figure(figsize=(10, 7))
        fig_3d.patch.set_facecolor('#05070a')
        ax_3d = fig_3d.add_subplot(111, projection='3d')
        ax_3d.set_facecolor('#05070a')
        ax_3d.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=df["Cluster"], cmap=cmap_custom, s=100, edgecolors='white', alpha=0.8)
        ax_3d.set_xlabel("PCA 1")
        ax_3d.set_ylabel("PCA 2")
        ax_3d.set_zlabel("PCA 3")
        ax_3d.set_title("3D PCA Cluster Visualization(Hierarchichal)", color='white')
        st.pyplot(fig_3d)

    with tab_profile:
        st.subheader("Categorical Cluster Distribution")
        p_col1, p_col2 = st.columns(2)
        with p_col1:
            st.write("#### Education Profile by Cluster")
            fig_edu, ax_edu = plt.subplots()
            fig_edu.patch.set_facecolor('#05070a')
            ax_edu.set_facecolor('#05070a')
            sns.countplot(data=df, x='Cluster', hue='Education', palette='magma', ax=ax_edu)
            st.pyplot(fig_edu)
        with p_col2:
            st.write("#### Living Status Profile by Cluster")
            fig_liv, ax_liv = plt.subplots()
            fig_liv.patch.set_facecolor('#05070a')
            ax_liv.set_facecolor('#05070a')
            sns.countplot(data=df, x='Cluster', hue='Living_with', palette='viridis', ax=ax_liv)
            st.pyplot(fig_liv)
    
    with tab_persona:
        # Cluster Summary Table
        st.subheader("Cluster Profiling")
        summary_table = df_encoded.groupby("Cluster").mean()
        st.dataframe(summary_table.style.background_gradient(cmap='YlGn'), use_container_width=True)

        st.markdown("---")
        st.subheader("Deep Behavioral Persona Playbook")
        summary = df.groupby("Cluster").mean(numeric_only=True)
        
       # Ensure 'Living_with' is included in your summary logic before the loop:
        # summary = df.groupby("Cluster").agg({ ... 'Living_with': lambda x: x.mode()[0], ... })

        for i in range(k_selected):
            row = summary.loc[i]
            cluster_color = custom_colors[i]
            
            # Fetch Living Status mode for this cluster
            living_status = df[df["Cluster"] == i]["Living_with"].mode()[0]
            
            st.markdown(f"""
                <div class="persona-card">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <h2 style="color:{cluster_color}; margin:0; font-size:32px;">SEGMENT {i}</h2>
                        <span class="strategy-pill">Cluster ID: {i}</span>
                    </div>
                    <hr style="opacity:0.1; margin: 20px 0;">
                    <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 20px;">
                        <div>
                            <p style="color:#00d4ff; font-weight:bold; margin-bottom:5px;">FINANCIAL DNA</p>
                            <p style="font-size:18px; margin:0;">Income: {int(row['Income']):,}</p>
                            <p style="font-size:18px; margin:0;">Spending: {int(row['Total_spendings']):,}</p>
                        </div>
                        <div>
                            <p style="color:#ff007f; font-weight:bold; margin-bottom:5px;">HOUSEHOLD & CHANNELS</p>
                            <p style="font-size:16px; margin:0; color:#fff; font-weight:bold;">Status: {living_status}</p>
                            <p style="font-size:15px; margin:0;">🌐 Web: {row.get('NumWebPurchases', 0):.1f}</p>
                            <p style="font-size:15px; margin:0;">📖 Catalog: {row.get('NumCatalogPurchases', 0):.1f}</p>
                            <p style="font-size:15px; margin:0;">🏬 Store: {row.get('NumStorePurchases', 0):.1f}</p>
                        </div>
                        <div>
                            <p style="color:#7cfc00; font-weight:bold; margin-bottom:5px;">ENGAGEMENT</p>
                            <p style="font-size:18px; margin:0;">Response Rate: {row.get('Total_Promos', 0):.2f}</p>
                            <p style="font-size:18px; margin:0;">Tenure: {int(row.get('Customer_tenure_days', 0))} Days</p>
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

            # --- ADVANCED DYNAMIC STRATEGY LOGIC ---
            s1, s2 = st.columns(2)
            avg_spend = summary['Total_spendings'].mean()
            avg_income = summary['Income'].mean()
            
            # Fetch Living Status mode for this specific cluster
            living_status = df[df["Cluster"] == i]["Living_with"].mode()[0]

            if living_status == "Partner":
                if row['Total_spendings'] > avg_spend:
                    with s1: st.info(f" **Family Elite Strategy:** High-value couples. Deploy 'Couple’s Retreat' bundles and premium wine tasting invites. Focus on high-margin household luxury.")
                    with s2: st.success(f" **Expansion:** Launch a family-tier loyalty program. Use catalog marketing for high-end home appliances or decor.")
                else:
                    with s1: st.warning(f" **Household Stability:** Value-conscious partners. Focus on 'Stock Up & Save' campaigns for daily essentials and kitchen staples.")
                    with s2: st.info(f" **Promotion:** Implement 'Buy 2 Get 1' on high-volume products. Use email automation for weekly family meal-planning deals.")
            
            else: # Alone
                if row['Income'] > avg_income:
                    with s1: st.info(f" **Solo Luxury Strategy:** High-income singles. Focus on 'Treat Yourself' messaging. Showcase gold products and premium individual-sized gourmet meat.")
                    with s2: st.success(f" **Digital VIP:** High engagement potential. Offer exclusive web-only early access to new tech or fashion launches.")
                elif row['Total_spendings'] < avg_spend:
                    with s1: st.error(f" **Efficiency Strategy:** Price-sensitive singles. Maximize inventory turnover via flash sales and heavy discount push notifications.")
                    with s2: st.info(f"**Channel:** Focus on Mobile/Web-only coupons. Minimize catalog costs and redirect to digital-first daily deals.")
                else:
                    with s1: st.warning(f" **Niche Engagement:** Average income singles with steady spending. Use social proof and influencer-style lifestyle marketing.")
                    with s2: st.info(f" **Campaign:** Referral programs ('Invite a friend') to lower acquisition costs while increasing basket frequency.")

            st.markdown("<br>", unsafe_allow_html=True)

else:
    st.markdown("""
        <div style="text-align: center; padding: 120px; color: #30363d;">
            <img src="https://cdn-icons-png.flaticon.com/512/3081/3081840.png" width="100" style="filter: grayscale(1); opacity: 0.3;">
            <h2 style="margin-top:20px;">SYSTEM STANDBY</h2>
            <p>Awaiting enterprise dataset to initialize AI clustering engine.</p>
        </div>
    """, unsafe_allow_html=True)
