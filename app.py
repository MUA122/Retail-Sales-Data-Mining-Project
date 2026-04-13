import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="Retail Sales Data Mining Review",
    page_icon="📊",
    layout="wide",
)

DATA_PATH = "retail_sales_dataset.csv"


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Month"] = df["Date"].dt.strftime("%b")
    df["Month Number"] = df["Date"].dt.month
    df["Year"] = df["Date"].dt.year
    df["Age Group"] = pd.cut(
        df["Age"],
        bins=[17, 24, 34, 44, 54, 64],
        labels=["18-24", "25-34", "35-44", "45-54", "55-64"]
    )
    return df


def dataset_description_block(dataframe):
    numerical_cols = dataframe.select_dtypes(include=["number"]).columns.tolist()
    datetime_cols = dataframe.select_dtypes(include=["datetime64[ns]"]).columns.tolist()
    text_cols = dataframe.select_dtypes(include=["object", "category"]).columns.tolist()

    st.markdown("### Dataset Preview")
    st.dataframe(dataframe.head(100), use_container_width=True, hide_index=True)

    st.markdown("### Dataset Description")
    st.write(
        f"""
- **Dataset name:** Retail Sales Dataset  
- **Total rows:** {dataframe.shape[0]:,}  
- **Total columns:** {dataframe.shape[1]}  
- **Numerical columns:** {", ".join(numerical_cols) if numerical_cols else "None"}  
- **Text / categorical columns:** {", ".join(text_cols) if text_cols else "None"}  
- **Date / time columns:** {", ".join(datetime_cols) if datetime_cols else "None"}  
- **Grain:** Transaction-level retail sales records  
- **Main purpose:** Review customer behavior, product categories, quantity, pricing, and sales performance
"""
    )


df = load_data()

st.title("Retail Sales Dataset Review & Data Mining Project")
st.caption(
    "An interactive Streamlit preview for Phase 1 dataset review and Phase 2 project direction."
)

tab1, tab2 = st.tabs(["Phase 1 — Dataset Review", "Phase 2 — Data Mining Project"])

with tab1:
    st.subheader("Retail Sales Dataset — Exploratory Review")
    st.write(
        "This tab presents a clean Phase 1 review of the selected dataset, including business context, "
        "descriptive insights, filters, and visual analysis to support project planning."
    )

    with st.sidebar:
        st.header("Filters")

        all_categories = sorted(df["Product Category"].dropna().unique().tolist())
        all_genders = sorted(df["Gender"].dropna().unique().tolist())

        selected_categories = st.multiselect(
            "Product Category",
            options=all_categories,
            default=all_categories,
        )

        selected_genders = st.multiselect(
            "Gender",
            options=all_genders,
            default=all_genders,
        )

        age_min, age_max = int(df["Age"].min()), int(df["Age"].max())
        selected_age = st.slider("Age Range", age_min, age_max, (age_min, age_max))

    filtered = df[
        df["Product Category"].isin(selected_categories)
        & df["Gender"].isin(selected_genders)
        & df["Age"].between(selected_age[0], selected_age[1])
    ].copy()

    no_active_filter = (
        set(selected_categories) == set(all_categories)
        and set(selected_genders) == set(all_genders)
        and selected_age == (age_min, age_max)
    )

    if filtered.empty or no_active_filter:
        st.info(
            "No specific filter is currently applied, so the app is showing a general dataset preview."
        )
        dataset_description_block(df)

    else:
        total_revenue = float(filtered["Total Amount"].sum())
        total_transactions = int(filtered["Transaction ID"].nunique())
        avg_order_value = float(filtered["Total Amount"].mean())
        avg_quantity = float(filtered["Quantity"].mean())

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Revenue", f"${total_revenue:,.0f}")
        col2.metric("Transactions", f"{total_transactions:,}")
        col3.metric("Average Order Value", f"${avg_order_value:,.2f}")
        col4.metric("Average Quantity", f"{avg_quantity:.2f}")

        st.markdown("### Business Understanding")
        st.info(
            "This retail dataset can help analyze customer behavior, product performance, and sales patterns. "
            "It is useful for identifying strong product categories, customer segments, and revenue trends."
        )

        summary_col1, summary_col2 = st.columns([1.15, 1])

        with summary_col1:
            st.markdown("### Filtered Data Snapshot")
            st.dataframe(filtered.head(20), use_container_width=True, hide_index=True)

        with summary_col2:
            st.markdown("### Data Quality Notes")
            st.write(
                "- The dataset contains transaction-level retail sales records.\n"
                "- Main useful fields: date, customer demographics, category, quantity, unit price, and total amount.\n"
                "- The date column is converted into a proper datetime field.\n"
                "- Additional derived fields are created for month and age group analysis."
            )

        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            revenue_by_category = (
                filtered.groupby("Product Category", as_index=False)["Total Amount"]
                .sum()
                .sort_values("Total Amount", ascending=False)
            )
            fig_bar = px.bar(
                revenue_by_category,
                x="Product Category",
                y="Total Amount",
                title="Revenue by Product Category",
                text_auto=".2s",
            )
            fig_bar.update_layout(xaxis_title="", yaxis_title="Revenue")
            st.plotly_chart(fig_bar, use_container_width=True)

        with chart_col2:
            category_share = filtered["Product Category"].value_counts().reset_index()
            category_share.columns = ["Product Category", "Count"]
            fig_pie = px.pie(
                category_share,
                names="Product Category",
                values="Count",
                title="Transaction Share by Category",
                hole=0.45,
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        chart_col3, chart_col4 = st.columns(2)

        with chart_col3:
            monthly_sales = (
                filtered.groupby(["Month Number", "Month"], as_index=False)["Total Amount"]
                .sum()
                .sort_values("Month Number")
            )
            fig_line = px.line(
                monthly_sales,
                x="Month",
                y="Total Amount",
                markers=True,
                title="Monthly Revenue Trend",
            )
            fig_line.update_layout(xaxis_title="", yaxis_title="Revenue")
            st.plotly_chart(fig_line, use_container_width=True)

        with chart_col4:
            gender_spending = (
                filtered.groupby("Gender", as_index=False)["Total Amount"]
                .mean()
                .sort_values("Total Amount", ascending=False)
            )
            fig_gender = px.bar(
                gender_spending,
                x="Gender",
                y="Total Amount",
                title="Average Spending by Gender",
                text_auto=".2f",
            )
            fig_gender.update_layout(xaxis_title="", yaxis_title="Average Total Amount")
            st.plotly_chart(fig_gender, use_container_width=True)

        bottom_col1, bottom_col2 = st.columns(2)

        with bottom_col1:
            age_group_revenue = (
                filtered.groupby("Age Group", as_index=False)["Total Amount"]
                .sum()
            )
            fig_age = px.bar(
                age_group_revenue,
                x="Age Group",
                y="Total Amount",
                title="Revenue by Age Group",
                text_auto=".2s",
            )
            fig_age.update_layout(xaxis_title="", yaxis_title="Revenue")
            st.plotly_chart(fig_age, use_container_width=True)

        with bottom_col2:
            fig_scatter = px.scatter(
                filtered,
                x="Age",
                y="Total Amount",
                size="Quantity",
                color="Product Category",
                hover_data=["Customer ID", "Gender", "Price per Unit"],
                title="Customer Spending Pattern",
            )
            fig_scatter.update_layout(xaxis_title="Age", yaxis_title="Total Amount")
            st.plotly_chart(fig_scatter, use_container_width=True)

        st.markdown("### Key Insights for Phase 1")
        top_category = (
            revenue_by_category.iloc[0]["Product Category"]
            if not revenue_by_category.empty else "N/A"
        )
        top_month = (
            monthly_sales.sort_values("Total Amount", ascending=False).iloc[0]["Month"]
            if not monthly_sales.empty else "N/A"
        )

        st.write(
            f"- **Top revenue category:** {top_category}\n"
            f"- **Strongest month in the filtered view:** {top_month}\n"
            "- **Gender and age analysis** can support customer segmentation.\n"
            "- **Spending patterns** suggest this dataset is suitable for descriptive analytics, segmentation, and dashboarding."
        )

with tab2:
    st.subheader("Phase 2 — Data Mining Project")
    st.write(
        "This tab is reserved for the full data mining implementation, predictive or segmentation models, "
        "dashboard storytelling, and final project delivery."
    )

    st.markdown("### Planned Components")
    st.write(
        "- Data preprocessing and feature engineering\n"
        "- CRISP-DM workflow implementation\n"
        "- Customer segmentation / clustering\n"
        "- Business insight dashboard\n"
        "- Final model interpretation and presentation"
    )

    st.success("Coming soon — this section will contain the full data mining pipeline and model outputs.")