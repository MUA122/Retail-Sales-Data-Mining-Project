import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import streamlit as st

from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


st.set_page_config(
    page_title="Retail Sales Data Mining Dashboard",
    page_icon="📊",
    layout="wide",
)

DATA_PATH_OPTIONS = [
    "retail_sales_dataset.csv",
    "retail_sales_dataset(1).csv",
    "/mnt/data/retail_sales_dataset(1).csv",
]


# =============================
# Styling
# =============================
st.markdown(
    """
    <style>
        .main-title {
            font-size: 38px;
            font-weight: 900;
            margin-bottom: 4px;
            color: #111827;
        }
        .sub-text {
            color: #4b5563;
            font-size: 16px;
            margin-bottom: 20px;
        }
        .section-card,
        .insight-box,
        .success-box,
        .warning-box,
        .dark-box {
            color: #111827 !important;
            line-height: 1.55;
        }
        .section-card {
            padding: 20px;
            border-radius: 18px;
            background: #ffffff;
            border: 1px solid #e5e7eb;
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.08);
            margin-bottom: 18px;
        }
        .insight-box {
            padding: 16px 18px;
            border-radius: 14px;
            background: #eff6ff;
            border: 1px solid #bfdbfe;
            border-left: 6px solid #2563eb;
            margin-bottom: 12px;
        }
        .success-box {
            padding: 16px 18px;
            border-radius: 14px;
            background: #ecfdf5;
            border: 1px solid #bbf7d0;
            border-left: 6px solid #16a34a;
            margin-bottom: 12px;
        }
        .warning-box {
            padding: 16px 18px;
            border-radius: 14px;
            background: #fff7ed;
            border: 1px solid #fed7aa;
            border-left: 6px solid #f97316;
            margin-bottom: 12px;
        }
        .dark-box {
            padding: 16px 18px;
            border-radius: 14px;
            background: #111827;
            border: 1px solid #334155;
            color: #ffffff !important;
            margin-bottom: 12px;
        }
        .dark-box b, .dark-box li, .dark-box span, .dark-box p {
            color: #ffffff !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# =============================
# Helpers
# =============================
def find_data_path():
    for path in DATA_PATH_OPTIONS:
        if os.path.exists(path):
            return path
    return DATA_PATH_OPTIONS[0]


@st.cache_data
def load_data():
    data_path = find_data_path()
    df = pd.read_csv(data_path)

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Month"] = df["Date"].dt.strftime("%b")
    df["Month Number"] = df["Date"].dt.month
    df["Year"] = df["Date"].dt.year
    df["Age Group"] = pd.cut(
        df["Age"],
        bins=[17, 24, 34, 44, 54, 64, 100],
        labels=["18-24", "25-34", "35-44", "45-54", "55-64", "65+"],
        include_lowest=True,
    )
    df["High Spender"] = np.where(
        df["Total Amount"] >= df["Total Amount"].median(), "High", "Low"
    )
    return df


def apply_filters(dataframe):
    with st.sidebar:
        st.header("Global Dashboard Filters")

        categories = sorted(dataframe["Product Category"].dropna().unique().tolist())
        genders = sorted(dataframe["Gender"].dropna().unique().tolist())

        selected_categories = st.multiselect(
            "Product Category",
            categories,
            default=categories,
        )

        selected_genders = st.multiselect(
            "Gender",
            genders,
            default=genders,
        )

        min_age = int(dataframe["Age"].min())
        max_age = int(dataframe["Age"].max())
        selected_age = st.slider("Age Range", min_age, max_age, (min_age, max_age))

    filtered_df = dataframe[
        dataframe["Product Category"].isin(selected_categories)
        & dataframe["Gender"].isin(selected_genders)
        & dataframe["Age"].between(selected_age[0], selected_age[1])
    ].copy()

    return filtered_df


def metric_cards(values):
    cols = st.columns(len(values))
    for col, item in zip(cols, values):
        col.metric(item[0], item[1], item[2] if len(item) > 2 else None)


def safe_mode(series):
    modes = series.dropna().mode()
    return modes.iloc[0] if not modes.empty else "N/A"


def dominant_share(series):
    counts = series.dropna().value_counts(normalize=True)
    return float(counts.iloc[0] * 100) if not counts.empty else 0.0


def money(value):
    return f"${float(value):,.0f}"


def percent(value):
    return f"{float(value):.1f}%"


def build_owner_text_insights(dataframe):
    total_revenue = dataframe["Total Amount"].sum()
    total_transactions = dataframe["Transaction ID"].nunique()
    total_customers = dataframe["Customer ID"].nunique()
    avg_order_value = dataframe["Total Amount"].mean()

    category_revenue = dataframe.groupby("Product Category")["Total Amount"].sum().sort_values(ascending=False)
    category_transactions = dataframe.groupby("Product Category")["Transaction ID"].nunique().sort_values(ascending=False)
    gender_revenue = dataframe.groupby("Gender")["Total Amount"].sum().sort_values(ascending=False)
    gender_avg = dataframe.groupby("Gender")["Total Amount"].mean().sort_values(ascending=False)
    customer_revenue = dataframe.groupby("Customer ID")["Total Amount"].sum().sort_values(ascending=False)
    customer_transactions = dataframe.groupby("Customer ID")["Transaction ID"].nunique().sort_values(ascending=False)

    top_category = category_revenue.index[0]
    top_category_share = category_revenue.iloc[0] / total_revenue * 100 if total_revenue else 0
    top_transaction_category = category_transactions.index[0]
    top_gender_revenue = gender_revenue.index[0]
    top_gender_avg = gender_avg.index[0]
    top_customer = customer_revenue.index[0]
    top_customer_value = customer_revenue.iloc[0]
    top_customer_orders = customer_transactions.loc[top_customer]

    if "Month" in dataframe.columns and "Month Number" in dataframe.columns:
        monthly_revenue = dataframe.groupby(["Month Number", "Month"])["Total Amount"].sum().sort_values(ascending=False)
        strongest_month = monthly_revenue.index[0][1]
        strongest_month_value = monthly_revenue.iloc[0]
    else:
        strongest_month = "N/A"
        strongest_month_value = 0

    return {
        "total_revenue": total_revenue,
        "total_transactions": total_transactions,
        "total_customers": total_customers,
        "avg_order_value": avg_order_value,
        "top_category": top_category,
        "top_category_share": top_category_share,
        "top_transaction_category": top_transaction_category,
        "top_gender_revenue": top_gender_revenue,
        "top_gender_avg": top_gender_avg,
        "top_customer": top_customer,
        "top_customer_value": top_customer_value,
        "top_customer_orders": top_customer_orders,
        "strongest_month": strongest_month,
        "strongest_month_value": strongest_month_value,
    }


@st.cache_data
def prepare_customer_level_data(dataframe):
    # In this dataset each customer has one transaction, but this aggregation also works
    # if future datasets contain repeated customers.
    customer_df = (
        dataframe.groupby("Customer ID", as_index=False)
        .agg(
            Age_Min=("Age", "min"),
            Age_Max=("Age", "max"),
            Age=("Age", "mean"),
            Gender=("Gender", safe_mode),
            Product_Category=("Product Category", safe_mode),
            Quantity=("Quantity", "sum"),
            Total_Amount=("Total Amount", "sum"),
            Avg_Order_Value=("Total Amount", "mean"),
            Transactions=("Transaction ID", "nunique"),
            Avg_Price_Per_Unit=("Price per Unit", "mean"),
        )
    )
    return customer_df


# =============================
# Model Builders
# =============================
def run_kmeans(customer_df, n_clusters):
    features = [
        "Age",
        "Quantity",
        "Total_Amount",
        "Avg_Order_Value",
        "Transactions",
        "Avg_Price_Per_Unit",
    ]
    X = customer_df[features].fillna(0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = model.fit_predict(X_scaled)

    result = customer_df.copy()
    result["Cluster"] = clusters.astype(str)

    sil = silhouette_score(X_scaled, clusters) if n_clusters > 1 and len(result) > n_clusters else np.nan
    return result, model, scaler, features, sil


def build_cluster_summary(clustered_df):
    rows = []
    for cluster_id, group in clustered_df.groupby("Cluster"):
        main_gender = safe_mode(group["Gender"])
        main_category = safe_mode(group["Product_Category"])
        rows.append(
            {
                "Cluster": cluster_id,
                "Customers": int(group["Customer ID"].nunique()),
                "Age Range": f"{int(group['Age_Min'].min())} - {int(group['Age_Max'].max())}",
                "Main Gender": main_gender,
                "Main Gender %": f"{dominant_share(group['Gender']):.1f}%",
                "Main Category": main_category,
                "Main Category %": f"{dominant_share(group['Product_Category']):.1f}%",
                "Avg Quantity": round(group["Quantity"].mean(), 2),
                "Quantity Range": f"{int(group['Quantity'].min())} - {int(group['Quantity'].max())}",
                "Avg Spending": round(group["Total_Amount"].mean(), 2),
                "Total Revenue": round(group["Total_Amount"].sum(), 2),
            }
        )
    return pd.DataFrame(rows).sort_values("Cluster")


def describe_cluster(row):
    spending = "high-value" if row["Avg Spending"] >= row["Avg Spending Median"] else "lower-value"
    quantity = "higher-quantity" if row["Avg Quantity"] >= row["Avg Quantity Median"] else "lower-quantity"
    return (
        f"Cluster {row['Cluster']} is a {spending}, {quantity} customer segment. "
        f"The age range is {row['Age Range']}, the dominant gender is {row['Main Gender']} "
        f"({row['Main Gender %']}), and the strongest product category is {row['Main Category']} "
        f"({row['Main Category %']})."
    )


@st.cache_resource
def train_random_forest_spend(dataframe):
    model_df = dataframe.copy().dropna(
        subset=["Gender", "Product Category", "Age", "Quantity", "Price per Unit", "Total Amount"]
    )

    high_threshold = model_df["Total Amount"].quantile(0.65)
    model_df["High Spender Binary"] = (model_df["Total Amount"] >= high_threshold).astype(int)

    X = model_df[["Gender", "Age", "Product Category", "Quantity", "Price per Unit"]]
    y = model_df["High Spender Binary"]

    categorical_features = ["Gender", "Product Category"]
    numeric_features = ["Age", "Quantity", "Price per Unit"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("num", StandardScaler(), numeric_features),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=7,
                    random_state=42,
                    class_weight="balanced",
                ),
            ),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, predictions)
    cm = confusion_matrix(y_test, predictions)
    report = classification_report(
        y_test, predictions, target_names=["Normal/Low", "High"], output_dict=True, zero_division=0
    )

    result_df = X_test.copy()
    result_df["Actual High Spender"] = np.where(y_test.values == 1, "High", "Normal/Low")
    result_df["Predicted High Spender"] = np.where(predictions == 1, "High", "Normal/Low")
    result_df["High Spending Probability"] = probabilities

    return model, X_test, y_test, predictions, probabilities, accuracy, cm, report, high_threshold, result_df


@st.cache_resource
def train_random_forest_category(dataframe):
    model_df = dataframe.copy().dropna(
        subset=["Gender", "Age", "Quantity", "Price per Unit", "Product Category"]
    )

    X = model_df[["Gender", "Age", "Quantity", "Price per Unit"]]
    y = model_df["Product Category"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["Gender"]),
            ("num", StandardScaler(), ["Age", "Quantity", "Price per Unit"]),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=8,
                    random_state=42,
                    class_weight="balanced",
                ),
            ),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    cm = confusion_matrix(y_test, predictions, labels=model.classes_)
    report = classification_report(y_test, predictions, output_dict=True, zero_division=0)

    return model, X_test, y_test, predictions, accuracy, cm, report


@st.cache_resource
def train_linear_regression(dataframe):
    model_df = dataframe.copy().dropna(
        subset=["Gender", "Product Category", "Age", "Quantity", "Price per Unit", "Total Amount"]
    )

    X = model_df[["Gender", "Age", "Product Category", "Quantity", "Price per Unit"]]
    y = model_df["Total Amount"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["Gender", "Product Category"]),
            ("num", StandardScaler(), ["Age", "Quantity", "Price per Unit"]),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", LinearRegression()),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    result_df = X_test.copy()
    result_df["Actual Total Amount"] = y_test.values
    result_df["Predicted Total Amount"] = predictions
    result_df["Error"] = result_df["Actual Total Amount"] - result_df["Predicted Total Amount"]

    return model, X_test, y_test, predictions, mae, rmse, r2, result_df


# =============================
# App Starts
# =============================
df = load_data()
filtered = apply_filters(df)

st.markdown('<div class="main-title">Retail Sales Data Mining Dashboard</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-text">Phase 2 implementation: customer segmentation, high-spender prediction, product-category prediction, linear regression, evaluation metrics, and live testing.</div>',
    unsafe_allow_html=True,
)

if filtered.empty:
    st.error("No data available for the selected filters. Please change the sidebar filters.")
    st.stop()

overview_tab, owner_tab, kmeans_tab, rf_tab, lr_tab, crisp_tab = st.tabs(
    [
        "Overview & EDA",
        "Owner Dashboard",
        "K-Means Clustering",
        "Random Forest Prediction",
        "Linear Regression",
        "CRISP-DM & Business Insights",
    ]
)

# =============================
# Overview Tab
# =============================
with overview_tab:
    st.subheader("Dataset Overview")

    total_revenue = filtered["Total Amount"].sum()
    transactions = filtered["Transaction ID"].nunique()
    avg_order_value = filtered["Total Amount"].mean()
    avg_quantity = filtered["Quantity"].mean()

    metric_cards(
        [
            ("Total Revenue", f"${total_revenue:,.0f}"),
            ("Transactions", f"{transactions:,}"),
            ("Average Order Value", f"${avg_order_value:,.2f}"),
            ("Average Quantity", f"{avg_quantity:.2f}"),
        ]
    )

    st.markdown("### Data Preview")
    st.dataframe(filtered.head(100), use_container_width=True, hide_index=True)

    col1, col2 = st.columns(2)

    with col1:
        revenue_by_category = (
            filtered.groupby("Product Category", as_index=False)["Total Amount"]
            .sum()
            .sort_values("Total Amount", ascending=False)
        )
        fig = px.bar(
            revenue_by_category,
            x="Product Category",
            y="Total Amount",
            title="Revenue by Product Category",
            text_auto=".2s",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        monthly_sales = (
            filtered.groupby(["Month Number", "Month"], as_index=False)["Total Amount"]
            .sum()
            .sort_values("Month Number")
        )
        fig = px.line(
            monthly_sales,
            x="Month",
            y="Total Amount",
            markers=True,
            title="Monthly Revenue Trend",
        )
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        gender_sales = filtered.groupby("Gender", as_index=False)["Total Amount"].mean()
        fig = px.bar(
            gender_sales,
            x="Gender",
            y="Total Amount",
            title="Average Spending by Gender",
            text_auto=".2f",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        fig = px.scatter(
            filtered,
            x="Age",
            y="Total Amount",
            size="Quantity",
            color="Product Category",
            hover_data=["Customer ID", "Gender", "Price per Unit"],
            title="Customer Spending Pattern",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Data Quality Check")
    quality_df = pd.DataFrame(
        {
            "Column": filtered.columns,
            "Missing Values": filtered.isna().sum().values,
            "Missing %": (filtered.isna().mean().values * 100).round(2),
            "Unique Values": filtered.nunique().values,
        }
    )
    st.dataframe(quality_df, use_container_width=True, hide_index=True)


# =============================
# Owner Dashboard Tab
# =============================
with owner_tab:
    st.subheader("Owner Dashboard")
    st.write(
        "This tab summarizes the most important business information an owner needs: total sales, best categories, gender behavior, top customers, quantities, and direct text insights."
    )

    owner = build_owner_text_insights(filtered)

    metric_cards(
        [
            ("Total Sales", money(owner["total_revenue"])),
            ("Transactions", f"{owner['total_transactions']:,}"),
            ("Customers", f"{owner['total_customers']:,}"),
            ("Avg Order Value", f"${owner['avg_order_value']:,.2f}"),
        ]
    )

    st.markdown("### Executive Summary for the Owner")
    st.markdown(
        f"""
        <div class="success-box">
            <b>Overall performance:</b> The selected data generated <b>{money(owner['total_revenue'])}</b> from <b>{owner['total_transactions']:,}</b> transactions and <b>{owner['total_customers']:,}</b> unique customers.
        </div>
        <div class="insight-box">
            <b>Best category:</b> <b>{owner['top_category']}</b> is the strongest revenue category, contributing around <b>{owner['top_category_share']:.1f}%</b> of total sales in the selected view.
        </div>
        <div class="insight-box">
            <b>Customer behavior:</b> <b>{owner['top_gender_revenue']}</b> customers generated the highest total revenue, while <b>{owner['top_gender_avg']}</b> customers have the highest average spending per transaction.
        </div>
        <div class="warning-box">
            <b>Top customer:</b> Customer ID <b>{owner['top_customer']}</b> is the highest-value customer in this view with total spending of <b>{money(owner['top_customer_value'])}</b> across <b>{owner['top_customer_orders']}</b> transaction(s).
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Category Performance")
    category_summary = (
        filtered.groupby("Product Category")
        .agg(
            Revenue=("Total Amount", "sum"),
            Transactions=("Transaction ID", "nunique"),
            Customers=("Customer ID", "nunique"),
            Quantity_Sold=("Quantity", "sum"),
            Avg_Order_Value=("Total Amount", "mean"),
            Avg_Quantity=("Quantity", "mean"),
        )
        .reset_index()
        .sort_values("Revenue", ascending=False)
    )
    category_summary["Revenue Share %"] = category_summary["Revenue"] / category_summary["Revenue"].sum() * 100

    category_display = category_summary.copy()
    category_display["Revenue"] = category_display["Revenue"].map(lambda x: f"${x:,.0f}")
    category_display["Avg_Order_Value"] = category_display["Avg_Order_Value"].map(lambda x: f"${x:,.2f}")
    category_display["Avg_Quantity"] = category_display["Avg_Quantity"].round(2)
    category_display["Revenue Share %"] = category_display["Revenue Share %"].map(lambda x: f"{x:.1f}%")
    st.dataframe(category_display, use_container_width=True, hide_index=True)

    col1, col2 = st.columns(2)
    with col1:
        fig = px.pie(
            category_summary,
            names="Product Category",
            values="Revenue",
            title="Revenue Share by Product Category",
            hole=0.45,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.bar(
            category_summary,
            x="Product Category",
            y="Quantity_Sold",
            title="Quantity Sold by Category",
            text_auto=True,
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Gender Analysis by Category")
    gender_category = (
        filtered.groupby(["Product Category", "Gender"], as_index=False)
        .agg(
            Revenue=("Total Amount", "sum"),
            Transactions=("Transaction ID", "nunique"),
            Quantity_Sold=("Quantity", "sum"),
            Avg_Order_Value=("Total Amount", "mean"),
        )
    )
    category_totals = gender_category.groupby("Product Category")["Revenue"].transform("sum")
    gender_category["Category Revenue Share %"] = np.where(category_totals > 0, gender_category["Revenue"] / category_totals * 100, 0)

    gender_display = gender_category.copy().sort_values(["Product Category", "Revenue"], ascending=[True, False])
    gender_display["Revenue"] = gender_display["Revenue"].map(lambda x: f"${x:,.0f}")
    gender_display["Avg_Order_Value"] = gender_display["Avg_Order_Value"].map(lambda x: f"${x:,.2f}")
    gender_display["Category Revenue Share %"] = gender_display["Category Revenue Share %"].map(lambda x: f"{x:.1f}%")
    st.dataframe(gender_display, use_container_width=True, hide_index=True)

    col3, col4 = st.columns(2)
    with col3:
        fig = px.bar(
            gender_category,
            x="Product Category",
            y="Revenue",
            color="Gender",
            barmode="group",
            title="Male vs Female Revenue per Category",
            text_auto=".2s",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        fig = px.sunburst(
            gender_category,
            path=["Product Category", "Gender"],
            values="Revenue",
            title="Category → Gender Revenue Breakdown",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Top Customers")
    top_customers = (
        filtered.groupby("Customer ID")
        .agg(
            Total_Spending=("Total Amount", "sum"),
            Transactions=("Transaction ID", "nunique"),
            Quantity_Sold=("Quantity", "sum"),
            Avg_Order_Value=("Total Amount", "mean"),
            Age_Min=("Age", "min"),
            Age_Max=("Age", "max"),
            Most_Common_Gender=("Gender", safe_mode),
            Most_Common_Category=("Product Category", safe_mode),
        )
        .reset_index()
        .sort_values("Total_Spending", ascending=False)
        .head(10)
    )
    top_customers["Age Range"] = top_customers["Age_Min"].astype(int).astype(str) + " - " + top_customers["Age_Max"].astype(int).astype(str)
    top_customers_display = top_customers.drop(columns=["Age_Min", "Age_Max"]).copy()
    top_customers_display["Total_Spending"] = top_customers_display["Total_Spending"].map(lambda x: f"${x:,.0f}")
    top_customers_display["Avg_Order_Value"] = top_customers_display["Avg_Order_Value"].map(lambda x: f"${x:,.2f}")
    st.dataframe(top_customers_display, use_container_width=True, hide_index=True)

    st.markdown("### Owner Text Analysis")
    best_category_row = category_summary.iloc[0]
    lowest_category_row = category_summary.iloc[-1]
    highest_quantity_category = category_summary.sort_values("Quantity_Sold", ascending=False).iloc[0]

    st.markdown(
        f"""
        <div class="section-card">
            <b>1. Sales focus:</b> The owner should focus first on <b>{best_category_row['Product Category']}</b> because it generated the highest revenue: <b>${best_category_row['Revenue']:,.0f}</b>.<br><br>
            <b>2. Quantity movement:</b> The category with the highest sold quantity is <b>{highest_quantity_category['Product Category']}</b>, with <b>{int(highest_quantity_category['Quantity_Sold'])}</b> units sold. This category is important for stock planning.<br><br>
            <b>3. Weak category:</b> <b>{lowest_category_row['Product Category']}</b> currently has the lowest revenue at <b>${lowest_category_row['Revenue']:,.0f}</b>. The owner can review its pricing, promotion, or product availability.<br><br>
            <b>4. Gender targeting:</b> The dashboard compares male and female performance inside each category. This helps the owner decide whether a category should be promoted more to male customers, female customers, or both.<br><br>
            <b>5. Customer targeting:</b> The top customer table helps identify valuable customers who can receive loyalty offers, premium bundles, or personalized recommendations.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Recommended Owner Actions")
    st.write(
        """
- Increase stock and marketing attention for the strongest revenue category.
- Use gender/category analysis to create more focused campaigns.
- Give loyalty offers to top customers with the highest spending.
- Watch the lowest-performing category and test discounts, bundles, or better product placement.
- Use quantity sold to support inventory planning, not only revenue.
        """
    )


# =============================
# K-Means Tab
# =============================
with kmeans_tab:
    st.subheader("K-Means Customer Segmentation")
    st.write(
        "This model divides customers into business segments using spending behavior, quantity, age, transactions, and price level. The output is explained as customer groups, not only cluster numbers."
    )

    customer_df = prepare_customer_level_data(filtered)

    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        max_clusters = min(6, max(2, len(customer_df) - 1))
        n_clusters = st.slider("Number of Clusters", min_value=2, max_value=max_clusters, value=min(3, max_clusters))
    with col2:
        st.metric("Customers Used", f"{len(customer_df):,}")

    clustered_df, kmeans_model, scaler, cluster_features, sil_score = run_kmeans(customer_df, n_clusters)

    with col3:
        st.metric("Silhouette Score", f"{sil_score:.3f}" if not np.isnan(sil_score) else "N/A")

    cluster_summary = build_cluster_summary(clustered_df)
    cluster_summary["Avg Spending Median"] = cluster_summary["Avg Spending"].median()
    cluster_summary["Avg Quantity Median"] = cluster_summary["Avg Quantity"].median()
    cluster_summary["Segment Interpretation"] = cluster_summary.apply(describe_cluster, axis=1)
    display_summary = cluster_summary.drop(columns=["Avg Spending Median", "Avg Quantity Median"])

    st.markdown("### Business Cluster Summary")
    st.dataframe(display_summary, use_container_width=True, hide_index=True)

    st.markdown("### Cluster Interpretation")
    for _, row in display_summary.iterrows():
        st.markdown(
            f"""
            <div class="insight-box">
                <b>Cluster {row['Cluster']}:</b> {row['Segment Interpretation']}
            </div>
            """,
            unsafe_allow_html=True,
        )

    col_a, col_b = st.columns(2)

    with col_a:
        fig = px.scatter(
            clustered_df,
            x="Age",
            y="Total_Amount",
            color="Cluster",
            size="Quantity",
            hover_data=["Customer ID", "Gender", "Product_Category", "Avg_Order_Value", "Transactions"],
            title="Customer Segments: Age vs Total Spending",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        cluster_category = (
            clustered_df.groupby(["Cluster", "Product_Category"], as_index=False)["Customer ID"]
            .count()
            .rename(columns={"Customer ID": "Customers"})
        )
        fig = px.bar(
            cluster_category,
            x="Cluster",
            y="Customers",
            color="Product_Category",
            title="Product Category Distribution inside Each Cluster",
        )
        st.plotly_chart(fig, use_container_width=True)

    col_c, col_d = st.columns(2)
    with col_c:
        cluster_gender = (
            clustered_df.groupby(["Cluster", "Gender"], as_index=False)["Customer ID"]
            .count()
            .rename(columns={"Customer ID": "Customers"})
        )
        fig = px.bar(
            cluster_gender,
            x="Cluster",
            y="Customers",
            color="Gender",
            title="Gender Distribution inside Each Cluster",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_d:
        fig = px.box(
            clustered_df,
            x="Cluster",
            y="Quantity",
            color="Cluster",
            title="Quantity Range by Cluster",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Live Customer Segmentation Test")
    st.write("Enter a customer profile and the model will assign the closest business cluster.")

    live_col1, live_col2, live_col3 = st.columns(3)
    with live_col1:
        live_age = st.number_input("Customer Age", min_value=18, max_value=90, value=30, key="km_age")
        live_quantity = st.number_input("Total Quantity", min_value=1, max_value=50, value=3, key="km_quantity")
    with live_col2:
        live_total = st.number_input("Total Spending", min_value=1.0, max_value=10000.0, value=500.0, step=50.0, key="km_total")
        live_aov = st.number_input("Average Order Value", min_value=1.0, max_value=5000.0, value=250.0, step=25.0, key="km_aov")
    with live_col3:
        live_transactions = st.number_input("Transactions", min_value=1, max_value=30, value=1, key="km_transactions")
        live_avg_price = st.number_input("Average Price per Unit", min_value=1.0, max_value=5000.0, value=150.0, step=25.0, key="km_price")

    live_customer = pd.DataFrame(
        [[live_age, live_quantity, live_total, live_aov, live_transactions, live_avg_price]],
        columns=cluster_features,
    )
    live_scaled = scaler.transform(live_customer)
    live_cluster = str(kmeans_model.predict(live_scaled)[0])

    matched_cluster = display_summary[display_summary["Cluster"] == live_cluster].iloc[0]
    st.success(f"This customer belongs to Cluster {live_cluster}.")
    st.markdown(
        f"""
        <div class="success-box">
            <b>Segment meaning:</b> {matched_cluster['Segment Interpretation']}<br>
            <b>Main category:</b> {matched_cluster['Main Category']} | <b>Main gender:</b> {matched_cluster['Main Gender']} | <b>Quantity range:</b> {matched_cluster['Quantity Range']}
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Clustered Customers")
    st.dataframe(clustered_df, use_container_width=True, hide_index=True)


# =============================
# Random Forest Tab
# =============================
with rf_tab:
    st.subheader("Random Forest Prediction")
    st.write(
        "This section uses Random Forest in two practical ways: first to predict if a transaction is likely to be high spending, and second to estimate the most likely product category for the next purchase."
    )

    (
        rf_spend_model,
        X_test_rf,
        y_test_rf,
        rf_predictions,
        rf_probabilities,
        rf_accuracy,
        rf_cm,
        rf_report,
        high_threshold,
        rf_result_df,
    ) = train_random_forest_spend(filtered)

    rf_category_model, X_test_cat, y_test_cat, cat_predictions, cat_accuracy, cat_cm, cat_report = train_random_forest_category(filtered)

    metric_cards(
        [
            ("High-Spender Accuracy", f"{rf_accuracy:.3f}"),
            ("Category Accuracy", f"{cat_accuracy:.3f}"),
            ("High-Spender Threshold", f"${high_threshold:,.0f}"),
            ("Test Samples", f"{len(y_test_rf):,}"),
        ]
    )

    st.markdown("### What the Random Forest is answering")
    st.markdown(
        f"""
        <div class="dark-box">
            <b>Business question 1:</b> Is this customer/transaction likely to become a high-spending purchase?<br>
            <b>Business question 2:</b> Which product category is the customer most likely to buy?<br>
            <b>High spender definition:</b> A transaction is treated as High Spender when Total Amount is around ${high_threshold:,.0f} or more.
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### High-Spender Confusion Matrix")
        fig = ff.create_annotated_heatmap(
            z=rf_cm,
            x=["Predicted Normal/Low", "Predicted High"],
            y=["Actual Normal/Low", "Actual High"],
            colorscale="Blues",
            showscale=True,
        )
        fig.update_layout(title="Random Forest High-Spender Confusion Matrix")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### High-Spender Classification Report")
        report_df = pd.DataFrame(rf_report).transpose().round(3)
        st.dataframe(report_df, use_container_width=True)

    st.markdown("### Live Next-Purchase Prediction")
    st.write(
        "Enter a customer profile. The dashboard will predict the likely category and calculate the probability of high spending for each category scenario."
    )

    rf_col1, rf_col2, rf_col3 = st.columns(3)
    with rf_col1:
        rf_gender = st.selectbox("Gender", sorted(filtered["Gender"].dropna().unique()), key="rf_gender")
        rf_age = st.number_input("Age", min_value=18, max_value=90, value=30, key="rf_age")
    with rf_col2:
        rf_quantity = st.number_input("Expected Quantity", min_value=1, max_value=20, value=2, key="rf_quantity")
        rf_price = st.number_input("Expected Price per Unit", min_value=1.0, max_value=5000.0, value=100.0, step=10.0, key="rf_price")
    with rf_col3:
        rf_manual_category = st.selectbox(
            "Test Specific Category",
            sorted(filtered["Product Category"].dropna().unique()),
            key="rf_manual_category",
        )

    category_input = pd.DataFrame(
        [[rf_gender, rf_age, rf_quantity, rf_price]],
        columns=["Gender", "Age", "Quantity", "Price per Unit"],
    )
    predicted_category = rf_category_model.predict(category_input)[0]
    category_probabilities = rf_category_model.predict_proba(category_input)[0]
    category_probability_df = pd.DataFrame(
        {
            "Product Category": rf_category_model.classes_,
            "Purchase Category Probability": category_probabilities,
        }
    ).sort_values("Purchase Category Probability", ascending=False)

    scenario_rows = []
    for category in sorted(filtered["Product Category"].dropna().unique()):
        scenario_df = pd.DataFrame(
            [[rf_gender, rf_age, category, rf_quantity, rf_price]],
            columns=["Gender", "Age", "Product Category", "Quantity", "Price per Unit"],
        )
        high_probability = rf_spend_model.predict_proba(scenario_df)[0][1]
        predicted_label = rf_spend_model.predict(scenario_df)[0]
        scenario_rows.append(
            {
                "Product Category": category,
                "High Spending Probability": high_probability,
                "Predicted Spending Class": "High Spender" if predicted_label == 1 else "Normal/Low Spender",
            }
        )

    scenario_df = pd.DataFrame(scenario_rows).sort_values("High Spending Probability", ascending=False)
    best_spend_category = scenario_df.iloc[0]

    selected_scenario = pd.DataFrame(
        [[rf_gender, rf_age, rf_manual_category, rf_quantity, rf_price]],
        columns=["Gender", "Age", "Product Category", "Quantity", "Price per Unit"],
    )
    selected_prediction = rf_spend_model.predict(selected_scenario)[0]
    selected_probability = rf_spend_model.predict_proba(selected_scenario)[0][1]
    selected_label = "High Spender" if selected_prediction == 1 else "Normal/Low Spender"

    result_col1, result_col2, result_col3 = st.columns(3)
    result_col1.metric("Most Likely Category", predicted_category)
    result_col2.metric("Best High-Spend Category", best_spend_category["Product Category"])
    result_col3.metric("Selected Category High-Spend Probability", f"{selected_probability:.1%}")

    if selected_label == "High Spender":
        st.success(f"For {rf_manual_category}, prediction: {selected_label} | Probability: {selected_probability:.1%}")
    else:
        st.warning(f"For {rf_manual_category}, prediction: {selected_label} | Probability: {selected_probability:.1%}")

    col3, col4 = st.columns(2)
    with col3:
        st.markdown("### Category Purchase Probability")
        fig = px.bar(
            category_probability_df,
            x="Product Category",
            y="Purchase Category Probability",
            title="Most Likely Product Category for This Customer",
            text_auto=".1%",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        st.markdown("### High-Spending Probability by Category")
        fig = px.bar(
            scenario_df,
            x="Product Category",
            y="High Spending Probability",
            color="Predicted Spending Class",
            title="If This Customer Buys from Each Category, How Likely Is High Spending?",
            text_auto=".1%",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Scenario Table")
    display_scenario = scenario_df.copy()
    display_scenario["High Spending Probability"] = display_scenario["High Spending Probability"].map(lambda x: f"{x:.1%}")
    st.dataframe(display_scenario, use_container_width=True, hide_index=True)

    st.markdown("### Model Interpretation")
    st.markdown(
        """
        <div class="insight-box">
            <b>Business meaning:</b> Random Forest is now used as a decision-support tool. It does not only say High or Low; it also shows the most likely product category and compares high-spending probability across Beauty, Clothing, and Electronics. This helps the business decide which product category is more suitable for offers, recommendations, and targeting.
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("Show high-spender prediction test rows"):
        st.dataframe(rf_result_df.round(3), use_container_width=True, hide_index=True)


# =============================
# Linear Regression Tab
# =============================
with lr_tab:
    st.subheader("Linear Regression")
    st.write(
        "This regression model predicts the expected Total Amount for a transaction based on age, gender, product category, quantity, and price per unit."
    )

    lr_model, X_test_lr, y_test_lr, lr_predictions, mae, rmse, r2, regression_results = train_linear_regression(filtered)

    metric_cards(
        [
            ("MAE", f"${mae:,.2f}"),
            ("RMSE", f"${rmse:,.2f}"),
            ("R² Score", f"{r2:.3f}"),
        ]
    )

    col1, col2 = st.columns(2)

    with col1:
        fig = px.scatter(
            regression_results,
            x="Actual Total Amount",
            y="Predicted Total Amount",
            color="Product Category",
            title="Actual vs Predicted Total Amount",
        )
        fig.add_shape(
            type="line",
            x0=regression_results["Actual Total Amount"].min(),
            y0=regression_results["Actual Total Amount"].min(),
            x1=regression_results["Actual Total Amount"].max(),
            y1=regression_results["Actual Total Amount"].max(),
            line=dict(dash="dash"),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.histogram(
            regression_results,
            x="Error",
            nbins=30,
            title="Prediction Error Distribution",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Live Revenue Prediction Test")
    st.write("Enter a transaction profile and the model will estimate the Total Amount.")

    lr_col1, lr_col2, lr_col3 = st.columns(3)
    with lr_col1:
        lr_gender = st.selectbox("Gender", sorted(filtered["Gender"].dropna().unique()), key="lr_gender")
        lr_age = st.number_input("Age", min_value=18, max_value=90, value=30, key="lr_age")
    with lr_col2:
        lr_category = st.selectbox("Product Category", sorted(filtered["Product Category"].dropna().unique()), key="lr_category")
        lr_quantity = st.number_input("Quantity", min_value=1, max_value=20, value=2, key="lr_quantity")
    with lr_col3:
        lr_price = st.number_input("Price per Unit", min_value=1.0, max_value=5000.0, value=100.0, step=10.0, key="lr_price")

    lr_live_df = pd.DataFrame(
        [[lr_gender, lr_age, lr_category, lr_quantity, lr_price]],
        columns=["Gender", "Age", "Product Category", "Quantity", "Price per Unit"],
    )

    lr_live_prediction = lr_model.predict(lr_live_df)[0]
    st.success(f"Predicted Total Amount: ${lr_live_prediction:,.2f}")

    st.markdown("### Regression Test Results")
    st.dataframe(regression_results.round(2), use_container_width=True, hide_index=True)


# =============================
# CRISP-DM & Insights Tab
# =============================
with crisp_tab:
    st.subheader("CRISP-DM Compliance & Business Insights")

    st.markdown("### CRISP-DM Documentation")

    crisp_data = pd.DataFrame(
        [
            ["Business Understanding", "Analyze retail sales behavior, segment customers, predict high-spending transactions, and estimate expected revenue."],
            ["Data Understanding", "The dataset includes transaction date, customer ID, gender, age, product category, quantity, price per unit, and total amount."],
            ["Data Preparation", "Converted dates, created month/year fields, created age groups, handled missing values, encoded categorical variables, and scaled numerical features."],
            ["Modeling", "Applied K-Means clustering, Random Forest high-spender classification, Random Forest category prediction, and Linear Regression."],
            ["Evaluation", "Used Silhouette Score for clustering, Accuracy and confusion matrix for classification, and MAE/RMSE/R² for regression."],
            ["Deployment", "The Streamlit dashboard acts as the deployment layer, allowing users to filter data, view insights, and test live predictions."],
        ],
        columns=["CRISP-DM Phase", "Project Implementation"],
    )
    st.dataframe(crisp_data, use_container_width=True, hide_index=True)

    st.markdown("### Business Insights Panel")

    top_category = filtered.groupby("Product Category")["Total Amount"].sum().sort_values(ascending=False).index[0]
    top_category_value = filtered.groupby("Product Category")["Total Amount"].sum().max()
    best_gender = filtered.groupby("Gender")["Total Amount"].mean().sort_values(ascending=False).index[0]
    best_month = filtered.groupby(["Month Number", "Month"])["Total Amount"].sum().sort_values(ascending=False).index[0][1]

    st.markdown(
        f"""
        <div class="success-box">
            <b>Top Revenue Category:</b> {top_category} generated the highest total revenue with approximately ${top_category_value:,.0f}.
        </div>
        <div class="insight-box">
            <b>Best Average Spending Segment:</b> {best_gender} customers currently show the strongest average transaction value.
        </div>
        <div class="warning-box">
            <b>Strongest Sales Month:</b> {best_month} appears as the strongest month in the selected data view.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Recommended Business Actions")
    st.write(
        """
- Use K-Means clusters to create customer segments based on age range, gender, product category, quantity, and spending.
- Use Random Forest to identify likely high-spending purchases and the category most suitable for recommendations.
- Use Linear Regression predictions to estimate expected transaction value and support sales planning.
- Focus marketing campaigns on the highest revenue product category and the customer segments with the strongest spending behavior.
- Improve future model quality by adding more customer history, product names, discounts, and repeated transaction behavior.
        """
    )

    st.markdown("### Final Project Notes")
    st.info(
        "This dashboard includes the required Phase 2 mining techniques, interactive filters, model outputs, evaluation metrics, live prediction tools, and CRISP-DM documentation."
    )
