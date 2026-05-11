
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


def train_kmeans(df, n_clusters=3):
    customer_df = prepare_customer_level_data(df)
    X = customer_df[KMEANS_FEATURES].fillna(0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    n_clusters = int(max(2, min(n_clusters, len(customer_df) - 1)))
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
    clusters = model.fit_predict(X_scaled)

    clustered_df = customer_df.copy()
    clustered_df["Cluster"] = clusters.astype(str)

    metrics = {
        "Number of Customers": len(customer_df),
        "Number of Clusters": n_clusters,
        "Inertia": float(model.inertia_),
        "Silhouette Score": float(silhouette_score(X_scaled, clusters)) if len(set(clusters)) > 1 else np.nan,
        "Calinski Harabasz Score": float(calinski_harabasz_score(X_scaled, clusters)) if len(set(clusters)) > 1 else np.nan,
        "Davies Bouldin Score": float(davies_bouldin_score(X_scaled, clusters)) if len(set(clusters)) > 1 else np.nan,
    }

    summary = build_cluster_summary(clustered_df)
    return model, scaler, clustered_df, summary, metrics, KMEANS_FEATURES


def build_cluster_summary(clustered_df):
    rows = []
    for cluster_id, group in clustered_df.groupby("Cluster"):
        rows.append(
            {
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
            }
        )

    summary = pd.DataFrame(rows).sort_values("Cluster")
    spending_median = summary["Avg Spending"].median()
    quantity_median = summary["Avg Quantity"].median()

    def describe(row):
        spending = "high-value" if row["Avg Spending"] >= spending_median else "lower-value"
        quantity = "higher-quantity" if row["Avg Quantity"] >= quantity_median else "lower-quantity"
        return (
            f"Cluster {row['Cluster']} is a {spending}, {quantity} segment. "
            f"Age range: {row['Age Range']}. Main gender: {row['Main Gender']} "
            f"({row['Main Gender %']}%). Main category: {row['Main Category']} "
            f"({row['Main Category %']}%)."
        )

    summary["Business Interpretation"] = summary.apply(describe, axis=1)
    return summary


def predict_customer_cluster(model, scaler, customer_values):
    input_df = pd.DataFrame([customer_values], columns=KMEANS_FEATURES)
    input_scaled = scaler.transform(input_df)
    return str(model.predict(input_scaled)[0])
