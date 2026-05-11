import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from preprocessing import prepare_customer_level_data

KMEANS_FEATURES = [
    "Age",
    "Quantity",
    "Total_Amount",
    "Avg_Order_Value",
    "Transactions",
    "Avg_Price_Per_Unit",
]


def _safe_mode(series):
    modes = series.dropna().mode()
    return modes.iloc[0] if not modes.empty else "Unknown"


def _dominant_share(series):
    counts = series.dropna().value_counts(normalize=True)
    return float(counts.iloc[0] * 100) if not counts.empty else 0.0


def evaluate_k_values(df, min_k=2, max_k=8):
    customer_df = prepare_customer_level_data(df)
    if len(customer_df) < 3:
        raise ValueError("K-Means needs at least 3 customers.")

    X = customer_df[KMEANS_FEATURES].fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    max_k = min(max_k, len(customer_df) - 1)
    rows = []

    for k in range(min_k, max_k + 1):
        model = KMeans(n_clusters=k, random_state=42, n_init=30, max_iter=500)
        labels = model.fit_predict(X_scaled)
        rows.append({
            "K": k,
            "Inertia": float(model.inertia_),
            "Silhouette Score": float(silhouette_score(X_scaled, labels)),
            "Calinski Harabasz Score": float(calinski_harabasz_score(X_scaled, labels)),
            "Davies Bouldin Score": float(davies_bouldin_score(X_scaled, labels)),
        })

    result = pd.DataFrame(rows)
    result["Rank Silhouette"] = result["Silhouette Score"].rank(ascending=False)
    result["Rank Calinski"] = result["Calinski Harabasz Score"].rank(ascending=False)
    result["Rank Davies"] = result["Davies Bouldin Score"].rank(ascending=True)
    result["Overall Rank"] = result["Rank Silhouette"] + result["Rank Calinski"] + result["Rank Davies"]
    best_k = int(result.sort_values(["Overall Rank", "K"]).iloc[0]["K"])
    return result, best_k


def train_kmeans(df, n_clusters=3):
    customer_df = prepare_customer_level_data(df)
    if len(customer_df) < 3:
        raise ValueError("K-Means needs at least 3 customers.")

    X = customer_df[KMEANS_FEATURES].fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    n_clusters = int(max(2, min(n_clusters, len(customer_df) - 1)))
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=30, max_iter=500)
    clusters = model.fit_predict(X_scaled)

    clustered_df = customer_df.copy()
    clustered_df["Cluster"] = clusters.astype(str)

    metrics = {
        "Model Type": "K-Means Clustering",
        "Learning Type": "Unsupervised Learning",
        "Goal": "Customer Segmentation",
        "Number of Customers": int(len(customer_df)),
        "Number of Clusters": int(n_clusters),
        "Features Used": ", ".join(KMEANS_FEATURES),
        "Scaling Used": "StandardScaler",
        "Inertia": float(model.inertia_),
        "Silhouette Score": float(silhouette_score(X_scaled, clusters)),
        "Calinski Harabasz Score": float(calinski_harabasz_score(X_scaled, clusters)),
        "Davies Bouldin Score": float(davies_bouldin_score(X_scaled, clusters)),
    }

    summary = build_cluster_summary(clustered_df)
    recommendations = get_kmeans_recommendations(summary)
    return model, scaler, clustered_df, summary, metrics, KMEANS_FEATURES, recommendations


def _segment_name(row, spending_low, spending_high, quantity_low, quantity_high):
    if row["Avg Spending"] >= spending_high and row["Avg Quantity"] >= quantity_high:
        return "Premium High-Volume Customers"
    if row["Avg Spending"] >= spending_high:
        return "Premium High-Spending Customers"
    if row["Avg Quantity"] >= quantity_high:
        return "Bulk Quantity Customers"
    if row["Avg Spending"] <= spending_low and row["Avg Quantity"] <= quantity_low:
        return "Low-Value Light Buyers"
    return "Regular Mid-Value Customers"


def build_cluster_summary(clustered_df):
    rows = []
    for cluster_id, group in clustered_df.groupby("Cluster"):
        rows.append({
            "Cluster": cluster_id,
            "Customers": int(group["Customer ID"].nunique()),
            "Customer Share %": round(len(group) / len(clustered_df) * 100, 2),
            "Age Range": f"{int(group['Age_Min'].min())} - {int(group['Age_Max'].max())}",
            "Avg Age": round(group["Age"].mean(), 2),
            "Main Gender": _safe_mode(group["Gender"]),
            "Main Gender %": round(_dominant_share(group["Gender"]), 2),
            "Main Category": _safe_mode(group["Product_Category"]),
            "Main Category %": round(_dominant_share(group["Product_Category"]), 2),
            "Avg Quantity": round(group["Quantity"].mean(), 2),
            "Quantity Range": f"{int(group['Quantity'].min())} - {int(group['Quantity'].max())}",
            "Avg Spending": round(group["Total_Amount"].mean(), 2),
            "Min Spending": round(group["Total_Amount"].min(), 2),
            "Max Spending": round(group["Total_Amount"].max(), 2),
            "Total Revenue": round(group["Total_Amount"].sum(), 2),
            "Avg Order Value": round(group["Avg_Order_Value"].mean(), 2),
            "Avg Transactions": round(group["Transactions"].mean(), 2),
            "Avg Price Per Unit": round(group["Avg_Price_Per_Unit"].mean(), 2),
        })

    summary = pd.DataFrame(rows).sort_values("Cluster").reset_index(drop=True)
    spending_low = summary["Avg Spending"].quantile(0.33)
    spending_high = summary["Avg Spending"].quantile(0.66)
    quantity_low = summary["Avg Quantity"].quantile(0.33)
    quantity_high = summary["Avg Quantity"].quantile(0.66)

    summary["Segment Name"] = summary.apply(
        lambda row: _segment_name(row, spending_low, spending_high, quantity_low, quantity_high),
        axis=1,
    )

    def describe(row):
        return (
            f"{row['Segment Name']}: Cluster {row['Cluster']} contains {row['Customers']} customers "
            f"({row['Customer Share %']}% of all customers). "
            f"Average age is {row['Avg Age']} and age range is {row['Age Range']}. "
            f"Dominant gender is {row['Main Gender']} ({row['Main Gender %']}%). "
            f"Strongest product category is {row['Main Category']} ({row['Main Category %']}%). "
            f"Average quantity is {row['Avg Quantity']}, average spending is {row['Avg Spending']}, "
            f"and total revenue is {row['Total Revenue']}."
        )

    summary["Business Interpretation"] = summary.apply(describe, axis=1)
    return summary


def get_kmeans_recommendations(cluster_summary):
    rows = []
    for _, row in cluster_summary.iterrows():
        segment = row["Segment Name"]
        if "Premium" in segment:
            action = "VIP offers, premium bundles, early access, and personalized recommendations."
        elif "Bulk" in segment:
            action = "Quantity discounts, bundle offers, and inventory planning for high-volume demand."
        elif "Low-Value" in segment:
            action = "Small discounts, entry-level offers, and retargeting campaigns to increase value."
        else:
            action = "Regular campaigns, cross-selling, and category-based recommendations."
        rows.append({"Cluster": row["Cluster"], "Segment Name": segment, "Recommended Business Action": action})
    return pd.DataFrame(rows)


def predict_customer_cluster(model, scaler, customer_values):
    input_df = pd.DataFrame([customer_values], columns=KMEANS_FEATURES)
    input_scaled = scaler.transform(input_df)
    return str(model.predict(input_scaled)[0])
