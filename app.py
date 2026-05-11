
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import streamlit as st

from preprocessing import load_and_preprocess_data
from kmeans_model import KMEANS_FEATURES, predict_customer_cluster, train_kmeans
from random_forest_model import train_random_forest_category, train_random_forest_high_spender
from linear_regression_model import train_linear_regression

st.set_page_config(page_title="Retail Sales Data Mining Dashboard", page_icon="📊", layout="wide")

st.markdown(
    """
    <style>
        .main-title {font-size: 38px; font-weight: 900; color: #111827; margin-bottom: 4px;}
        .sub-text {color: #4b5563; font-size: 16px; margin-bottom: 20px;}
        .section-card, .insight-box, .success-box, .warning-box, .dark-box {line-height: 1.55;}
        .section-card {padding: 20px; border-radius: 18px; background: #ffffff; border: 1px solid #e5e7eb; box-shadow: 0 8px 24px rgba(15,23,42,0.08); margin-bottom: 18px;}
        .insight-box {padding: 16px 18px; border-radius: 14px; background: #eff6ff; border: 1px solid #bfdbfe; border-left: 6px solid #2563eb; margin-bottom: 12px;}
        .success-box {padding: 16px 18px; border-radius: 14px; background: #ecfdf5; border: 1px solid #bbf7d0; border-left: 6px solid #16a34a; margin-bottom: 12px;}
        .warning-box {padding: 16px 18px; border-radius: 14px; background: #fff7ed; border: 1px solid #fed7aa; border-left: 6px solid #f97316; margin-bottom: 12px;}
        .dark-box {padding: 16px 18px; border-radius: 14px; background: #111827; border: 1px solid #334155; color: #ffffff; margin-bottom: 12px;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def get_data():
    return load_and_preprocess_data()


@st.cache_resource
def get_kmeans_model(df, n_clusters):
    return train_kmeans(df, n_clusters=n_clusters)


@st.cache_resource
def get_rf_high_model(df):
    return train_random_forest_high_spender(df)


@st.cache_resource
def get_rf_category_model(df):
    return train_random_forest_category(df)


@st.cache_resource
def get_lr_model(df):
    return train_linear_regression(df)


def metric_cards(items):
    cols = st.columns(len(items))
    for col, (label, value) in zip(cols, items):
        col.metric(label, value)


def apply_filters(df):
    with st.sidebar:
        st.header("Global Filters")
        categories = sorted(df["Product Category"].dropna().unique())
        genders = sorted(df["Gender"].dropna().unique())
        selected_categories = st.multiselect("Product Category", categories, default=categories)
        selected_genders = st.multiselect("Gender", genders, default=genders)
        selected_age = st.slider("Age Range", int(df["Age"].min()), int(df["Age"].max()), (int(df["Age"].min()), int(df["Age"].max())))

    return df[
        df["Product Category"].isin(selected_categories)
        & df["Gender"].isin(selected_genders)
        & df["Age"].between(selected_age[0], selected_age[1])
    ].copy()


def show_metrics_dict(metrics):
    metric_df = pd.DataFrame({"Metric": list(metrics.keys()), "Value": list(metrics.values())})
    st.dataframe(metric_df, use_container_width=True, hide_index=True)


data_output = get_data()
df = data_output["df"]
filtered = apply_filters(df)

st.markdown('<div class="main-title">Retail Sales Data Mining Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-text">Separate files version: preprocessing + K-Means + Random Forest + Linear Regression + Streamlit App.</div>', unsafe_allow_html=True)

if filtered.empty:
    st.error("No data available for the selected filters.")
    st.stop()

overview_tab, preprocessing_tab, owner_tab, kmeans_tab, rf_tab, lr_tab, insights_tab = st.tabs(
    [
        "Overview & EDA",
        "Preprocessing Details",
        "Owner Dashboard",
        "K-Means Model",
        "Random Forest Model",
        "Linear Regression Model",
        "CRISP-DM & Insights",
    ]
)

with overview_tab:
    st.subheader("Dataset Overview")
    metric_cards([
        ("Total Revenue", f"${filtered['Total Amount'].sum():,.0f}"),
        ("Transactions", f"{filtered['Transaction ID'].nunique():,}"),
        ("Customers", f"{filtered['Customer ID'].nunique():,}"),
        ("Average Order Value", f"${filtered['Total Amount'].mean():,.2f}"),
    ])

    st.markdown("### Data Preview")
    st.dataframe(filtered.head(100), use_container_width=True, hide_index=True)

    col1, col2 = st.columns(2)
    with col1:
        revenue_by_category = filtered.groupby("Product Category", as_index=False)["Total Amount"].sum().sort_values("Total Amount", ascending=False)
        st.plotly_chart(px.bar(revenue_by_category, x="Product Category", y="Total Amount", title="Revenue by Product Category", text_auto=".2s"), use_container_width=True)
    with col2:
        monthly_sales = filtered.groupby(["Month Number", "Month"], as_index=False)["Total Amount"].sum().sort_values("Month Number")
        st.plotly_chart(px.line(monthly_sales, x="Month", y="Total Amount", markers=True, title="Monthly Revenue Trend"), use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        gender_sales = filtered.groupby("Gender", as_index=False)["Total Amount"].mean()
        st.plotly_chart(px.bar(gender_sales, x="Gender", y="Total Amount", title="Average Spending by Gender", text_auto=".2f"), use_container_width=True)
    with col4:
        st.plotly_chart(px.scatter(filtered, x="Age", y="Total Amount", size="Quantity", color="Product Category", hover_data=["Customer ID", "Gender"], title="Age vs Total Spending"), use_container_width=True)

with preprocessing_tab:
    st.subheader("Full Preprocessing Details")
    st.markdown("### Data Source")
    st.info(data_output["data_path"])

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Raw Data Details")
        show_metrics_dict(data_output["raw_details"])
    with col2:
        st.markdown("### Cleaning Report")
        show_metrics_dict(data_output["cleaning_report"])

    st.markdown("### Raw Data Quality")
    st.dataframe(data_output["raw_quality"], use_container_width=True, hide_index=True)

    st.markdown("### Cleaned Data Quality")
    st.dataframe(data_output["clean_quality"], use_container_width=True, hide_index=True)

    st.markdown("### Cleaned Numeric Summary")
    st.dataframe(data_output["clean_numeric_summary"], use_container_width=True, hide_index=True)

    st.markdown("### Cleaned Categorical Summary")
    st.dataframe(data_output["clean_categorical_summary"], use_container_width=True, hide_index=True)

with owner_tab:
    st.subheader("Owner Dashboard")
    category_summary = filtered.groupby("Product Category").agg(
        Revenue=("Total Amount", "sum"),
        Transactions=("Transaction ID", "nunique"),
        Customers=("Customer ID", "nunique"),
        Quantity_Sold=("Quantity", "sum"),
        Avg_Order_Value=("Total Amount", "mean"),
    ).reset_index().sort_values("Revenue", ascending=False)
    category_summary["Revenue Share %"] = category_summary["Revenue"] / category_summary["Revenue"].sum() * 100

    metric_cards([
        ("Best Category", category_summary.iloc[0]["Product Category"]),
        ("Best Category Revenue", f"${category_summary.iloc[0]['Revenue']:,.0f}"),
        ("Top Revenue Share", f"{category_summary.iloc[0]['Revenue Share %']:.1f}%"),
        ("Highest Quantity Category", category_summary.sort_values("Quantity_Sold", ascending=False).iloc[0]["Product Category"]),
    ])

    st.dataframe(category_summary.round(2), use_container_width=True, hide_index=True)
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(px.pie(category_summary, names="Product Category", values="Revenue", title="Revenue Share by Category", hole=0.45), use_container_width=True)
    with col2:
        st.plotly_chart(px.bar(category_summary, x="Product Category", y="Quantity_Sold", title="Quantity Sold by Category", text_auto=True), use_container_width=True)

    gender_category = filtered.groupby(["Product Category", "Gender"], as_index=False).agg(Revenue=("Total Amount", "sum"), Transactions=("Transaction ID", "nunique"), Quantity=("Quantity", "sum"))
    st.markdown("### Gender Analysis by Category")
    st.dataframe(gender_category, use_container_width=True, hide_index=True)
    st.plotly_chart(px.bar(gender_category, x="Product Category", y="Revenue", color="Gender", barmode="group", title="Male vs Female Revenue per Category"), use_container_width=True)

with kmeans_tab:
    st.subheader("K-Means Customer Segmentation")
    max_clusters = min(6, max(2, filtered["Customer ID"].nunique() - 1))
    n_clusters = st.slider("Number of Clusters", 2, max_clusters, min(3, max_clusters))
    model, scaler, clustered_df, cluster_summary, km_metrics, km_features = get_kmeans_model(filtered, n_clusters)

    metric_cards([
        ("Silhouette Score", f"{km_metrics['Silhouette Score']:.3f}"),
        ("Inertia", f"{km_metrics['Inertia']:,.2f}"),
        ("Calinski Harabasz", f"{km_metrics['Calinski Harabasz Score']:,.2f}"),
        ("Davies Bouldin", f"{km_metrics['Davies Bouldin Score']:.3f}"),
    ])

    st.markdown("### All K-Means Metrics")
    show_metrics_dict(km_metrics)

    st.markdown("### Cluster Summary")
    st.dataframe(cluster_summary, use_container_width=True, hide_index=True)

    for _, row in cluster_summary.iterrows():
        st.markdown(f"<div class='insight-box'><b>Cluster {row['Cluster']}:</b> {row['Business Interpretation']}</div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(px.scatter(clustered_df, x="Age", y="Total_Amount", color="Cluster", size="Quantity", hover_data=["Customer ID", "Gender", "Product_Category"], title="Customer Segments: Age vs Spending"), use_container_width=True)
    with col2:
        cluster_category = clustered_df.groupby(["Cluster", "Product_Category"], as_index=False)["Customer ID"].count().rename(columns={"Customer ID": "Customers"})
        st.plotly_chart(px.bar(cluster_category, x="Cluster", y="Customers", color="Product_Category", title="Category Distribution by Cluster"), use_container_width=True)

    st.markdown("### Live Customer Segmentation Test")
    live_cols = st.columns(6)
    live_values = []
    defaults = [30, 3, 500.0, 250.0, 1, 150.0]
    labels = ["Age", "Quantity", "Total Amount", "Avg Order Value", "Transactions", "Avg Price Per Unit"]
    for col, label, default in zip(live_cols, labels, defaults):
        with col:
            live_values.append(st.number_input(label, value=default, min_value=1.0 if isinstance(default, float) else 1))
    live_cluster = predict_customer_cluster(model, scaler, live_values)
    st.success(f"Predicted Cluster: {live_cluster}")

with rf_tab:
    st.subheader("Random Forest Models")
    rf_high = get_rf_high_model(filtered)
    rf_cat = get_rf_category_model(filtered)

    st.markdown("### High-Spender Classification Metrics")
    metric_cards([
        ("Accuracy", f"{rf_high['metrics']['Accuracy']:.3f}"),
        ("Balanced Accuracy", f"{rf_high['metrics']['Balanced Accuracy']:.3f}"),
        ("Precision", f"{rf_high['metrics']['Precision']:.3f}"),
        ("Recall", f"{rf_high['metrics']['Recall']:.3f}"),
        ("F1", f"{rf_high['metrics']['F1 Score']:.3f}"),
    ])
    show_metrics_dict(rf_high["metrics"])

    col1, col2 = st.columns(2)
    with col1:
        fig = ff.create_annotated_heatmap(z=rf_high["confusion_matrix"], x=["Pred Normal/Low", "Pred High"], y=["Actual Normal/Low", "Actual High"], colorscale="Blues", showscale=True)
        fig.update_layout(title="High-Spender Confusion Matrix")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.dataframe(rf_high["classification_report"].round(3), use_container_width=True)

    # st.markdown("### Product Category Classification Metrics")
    # show_metrics_dict(rf_cat["metrics"])
    # st.dataframe(rf_cat["classification_report"].round(3), use_container_width=True)

    st.markdown("### Live Random Forest Prediction")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        rf_gender = st.selectbox("Gender", sorted(filtered["Gender"].dropna().unique()), key="rf_gender")
        rf_age = st.number_input("Age", 18, 90, 30, key="rf_age")
    with col_b:
        rf_quantity = st.number_input("Quantity", 1, 20, 2, key="rf_quantity")
        rf_price = st.number_input("Price per Unit", 1.0, 5000.0, 100.0, step=10.0, key="rf_price")
    with col_c:
        rf_category = st.selectbox("Product Category", sorted(filtered["Product Category"].dropna().unique()), key="rf_category")

    category_input = pd.DataFrame([[rf_gender, rf_age, rf_quantity, rf_price]], columns=["Gender", "Age", "Quantity", "Price per Unit"])
    predicted_category = rf_cat["model"].predict(category_input)[0]

    spend_input = pd.DataFrame([[rf_gender, rf_age, rf_category, rf_quantity, rf_price]], columns=["Gender", "Age", "Product Category", "Quantity", "Price per Unit"])
    spend_probability = rf_high["model"].predict_proba(spend_input)[0][1]
    spend_label = "High Spender" if spend_probability >= rf_high["metrics"]["Best Probability Cutoff"] else "Normal/Low Spender"

    result_col1, result_col2 = st.columns(2)
    result_col1.metric("Predicted Product Category", predicted_category)
    result_col2.metric("High-Spending Probability", f"{spend_probability:.1%}")
    st.success(f"Spending Class: {spend_label}")

    with st.expander("Show High-Spender Test Rows"):
        st.dataframe(rf_high["result_df"].round(3), use_container_width=True, hide_index=True)
    with st.expander("Show Category Test Rows"):
        st.dataframe(rf_cat["result_df"], use_container_width=True, hide_index=True)

with lr_tab:
    st.subheader("Linear Regression Model")
    lr = get_lr_model(filtered)

    metric_cards([
        ("R² Score", f"{lr['metrics']['R2 Score']:.3f}"),
        ("MSE", f"{lr['metrics']['MSE']:,.2f}"),
        ("RMSE", f"{lr['metrics']['RMSE']:,.2f}"),
        ("MAE", f"{lr['metrics']['MAE']:,.2f}"),
    ])
    st.markdown("### All Regression Metrics")
    show_metrics_dict(lr["metrics"])

    col1, col2 = st.columns(2)
    with col1:
        fig = px.scatter(lr["result_df"], x="Actual Total Amount", y="Predicted Total Amount", color="Product Category", title="Actual vs Predicted Total Amount")
        fig.add_shape(type="line", x0=lr["result_df"]["Actual Total Amount"].min(), y0=lr["result_df"]["Actual Total Amount"].min(), x1=lr["result_df"]["Actual Total Amount"].max(), y1=lr["result_df"]["Actual Total Amount"].max(), line=dict(dash="dash"))
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.plotly_chart(px.histogram(lr["result_df"], x="Error", nbins=30, title="Prediction Error Distribution"), use_container_width=True)

    st.markdown("### Model Coefficients")
    st.dataframe(lr["coefficients"].round(4), use_container_width=True, hide_index=True)

    st.markdown("### Live Revenue Prediction")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        lr_gender = st.selectbox("Gender", sorted(filtered["Gender"].dropna().unique()), key="lr_gender")
        lr_age = st.number_input("Age", 18, 90, 30, key="lr_age")
    with col_b:
        lr_category = st.selectbox("Product Category", sorted(filtered["Product Category"].dropna().unique()), key="lr_category")
        lr_quantity = st.number_input("Quantity", 1, 20, 2, key="lr_quantity")
    with col_c:
        lr_price = st.number_input("Price per Unit", 1.0, 5000.0, 100.0, step=10.0, key="lr_price")

    lr_input = pd.DataFrame([[lr_gender, lr_age, lr_category, lr_quantity, lr_price]], columns=["Gender", "Age", "Product Category", "Quantity", "Price per Unit"])
    lr_prediction = lr["model"].predict(lr_input)[0]
    st.success(f"Predicted Total Amount: ${lr_prediction:,.2f}")

    with st.expander("Show Regression Test Results"):
        st.dataframe(lr["result_df"].round(2), use_container_width=True, hide_index=True)

with insights_tab:
    st.subheader("CRISP-DM & Business Insights")
    crisp = pd.DataFrame([
        ["Business Understanding", "Analyze retail sales behavior, segment customers, predict high spending, predict likely category, and estimate revenue."],
        ["Data Understanding", "Dataset includes transaction ID, date, customer ID, gender, age, category, quantity, unit price, and total amount."],
        ["Data Preparation", "Cleaned duplicates/missing values, converted dates, created month/year/age group/high spender, encoded categories, and scaled numeric features."],
        ["Modeling", "Used K-Means, Random Forest classification, and Linear Regression."],
        ["Evaluation", "Used Silhouette/Inertia/Calinski/Davies for clustering, Accuracy/Precision/Recall/F1/ROC AUC for classification, and R2/MSE/RMSE/MAE for regression."],
        ["Deployment", "Streamlit app provides filters, charts, model metrics, and live prediction tools."],
    ], columns=["CRISP-DM Phase", "Implementation"])
    st.dataframe(crisp, use_container_width=True, hide_index=True)

    top_category = filtered.groupby("Product Category")["Total Amount"].sum().sort_values(ascending=False)
    best_gender = filtered.groupby("Gender")["Total Amount"].mean().sort_values(ascending=False)
    st.markdown(f"""
    <div class='success-box'><b>Top category:</b> {top_category.index[0]} generated the highest total revenue: ${top_category.iloc[0]:,.0f}.</div>
    <div class='insight-box'><b>Best average spending gender:</b> {best_gender.index[0]} has the highest average transaction value.</div>
    <div class='warning-box'><b>Important note:</b> High accuracy is achieved when the dataset has strong relationships, especially because Total Amount is strongly related to Quantity and Price per Unit. Category prediction may be lower if category has no strong pattern in the available features.</div>
    """, unsafe_allow_html=True)
