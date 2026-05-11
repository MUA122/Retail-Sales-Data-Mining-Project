

import os
import numpy as np
import pandas as pd

DATA_PATH_OPTIONS = [
    "retail_sales_dataset.csv",
    "retail_sales_dataset(1).csv",
    "/mnt/data/retail_sales_dataset.csv",
    "/mnt/data/retail_sales_dataset(1).csv",
    "retail_sales_big_dataset.csv",
    "retail_sales_dataset(2).csv",
    "/mnt/data/retail_sales_big_dataset.csv",
    "/mnt/data/retail_sales_dataset(2).csv",
]

REQUIRED_COLUMNS = [
    "Transaction ID",
    "Date",
    "Customer ID",
    "Gender",
    "Age",
    "Product Category",
    "Quantity",
    "Price per Unit",
    "Total Amount",
]


def find_data_path(custom_path=None):
    """Find the CSV file path automatically, or use a custom path if provided."""
    if custom_path and os.path.exists(custom_path):
        return custom_path

    for path in DATA_PATH_OPTIONS:
        if os.path.exists(path):
            return path

    raise FileNotFoundError(
        "Dataset not found. Put retail_sales_dataset.csv in the same folder "
        "as these Python files, or pass a valid path to load_and_preprocess_data()."
    )


def load_raw_data(custom_path=None):
    """Load the raw CSV file without cleaning."""
    data_path = find_data_path(custom_path)
    return pd.read_csv(data_path), data_path


def validate_columns(df):
    """Check that the expected project columns exist."""
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return True


def build_data_details(df):
    """Return useful information about shape, columns, data types, missing values, and duplicates."""
    details = {
        "Rows": df.shape[0],
        "Columns": df.shape[1],
        "Duplicate Rows": int(df.duplicated().sum()),
        "Column Names": list(df.columns),
    }

    quality_table = pd.DataFrame(
        {
            "Column": df.columns,
            "Data Type": df.dtypes.astype(str).values,
            "Missing Values": df.isna().sum().values,
            "Missing %": (df.isna().mean().values * 100).round(2),
            "Unique Values": df.nunique(dropna=True).values,
        }
    )

    numeric_summary = df.describe(include=[np.number]).T.reset_index().rename(columns={"index": "Column"})
    categorical_summary = df.describe(include=["object", "category"]).T.reset_index().rename(columns={"index": "Column"})

    return details, quality_table, numeric_summary, categorical_summary


def clean_data(df):
    """
    Clean the retail dataset.
    Steps:
    1. Remove duplicate rows.
    2. Convert Date to datetime.
    3. Convert numeric columns safely.
    4. Fill missing categorical values using mode.
    5. Fill missing numeric values using median.
    6. Remove impossible values such as negative age, quantity, price, or total.
    7. Recalculate Total Amount when it is missing or invalid.
    8. Add Month, Year, Age Group, and High Spender columns.
    """
    df = df.copy()
    validate_columns(df)

    before_rows = len(df)
    before_missing = int(df.isna().sum().sum())
    before_duplicates = int(df.duplicated().sum())

    df = df.drop_duplicates().reset_index(drop=True)

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    numeric_cols = ["Age", "Quantity", "Price per Unit", "Total Amount"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    categorical_cols = ["Gender", "Product Category"]
    for col in categorical_cols:
        mode_value = df[col].dropna().mode()
        fill_value = mode_value.iloc[0] if not mode_value.empty else "Unknown"
        df[col] = df[col].fillna(fill_value)

    for col in numeric_cols:
        median_value = df[col].median()
        df[col] = df[col].fillna(median_value)

    # Remove clearly invalid records.
    df = df[
        (df["Age"].between(1, 120))
        & (df["Quantity"] > 0)
        & (df["Price per Unit"] > 0)
        & (df["Total Amount"] >= 0)
    ].copy()

    # If Total Amount is zero or inconsistent because of missing source values, recalculate it.
    calculated_total = df["Quantity"] * df["Price per Unit"]
    df["Total Amount"] = np.where(df["Total Amount"] <= 0, calculated_total, df["Total Amount"])

    # Date features.
    if df["Date"].isna().any():
        df["Date"] = df["Date"].fillna(df["Date"].dropna().median())

    df["Month"] = df["Date"].dt.strftime("%b")
    df["Month Number"] = df["Date"].dt.month
    df["Year"] = df["Date"].dt.year

    df["Age Group"] = pd.cut(
        df["Age"],
        bins=[0, 17, 24, 34, 44, 54, 64, 120],
        labels=["Under 18", "18-24", "25-34", "35-44", "45-54", "55-64", "65+"],
        include_lowest=True,
    )

    df["High Spender"] = np.where(
        df["Total Amount"] >= df["Total Amount"].quantile(0.65),
        "High",
        "Normal/Low",
    )

    after_rows = len(df)
    after_missing = int(df.isna().sum().sum())

    cleaning_report = {
        "Rows Before Cleaning": before_rows,
        "Rows After Cleaning": after_rows,
        "Rows Removed": before_rows - after_rows,
        "Duplicate Rows Removed": before_duplicates,
        "Missing Values Before": before_missing,
        "Missing Values After": after_missing,
        "New Features Created": ["Month", "Month Number", "Year", "Age Group", "High Spender"],
    }

    return df.reset_index(drop=True), cleaning_report


def prepare_customer_level_data(df):
    """Aggregate transaction-level data into customer-level data for K-Means."""
    def safe_mode(series):
        modes = series.dropna().mode()
        return modes.iloc[0] if not modes.empty else "Unknown"

    customer_df = (
        df.groupby("Customer ID", as_index=False)
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


def load_and_preprocess_data(custom_path=None):
    """Main function used by all model files and the Streamlit app."""
    raw_df, data_path = load_raw_data(custom_path)
    raw_details, raw_quality, raw_numeric, raw_categorical = build_data_details(raw_df)
    clean_df, cleaning_report = clean_data(raw_df)
    clean_details, clean_quality, clean_numeric, clean_categorical = build_data_details(clean_df)

    return {
        "data_path": data_path,
        "raw_df": raw_df,
        "df": clean_df,
        "raw_details": raw_details,
        "raw_quality": raw_quality,
        "raw_numeric_summary": raw_numeric,
        "raw_categorical_summary": raw_categorical,
        "cleaning_report": cleaning_report,
        "clean_details": clean_details,
        "clean_quality": clean_quality,
        "clean_numeric_summary": clean_numeric,
        "clean_categorical_summary": clean_categorical,
    }


if __name__ == "__main__":
    output = load_and_preprocess_data()
    print("Data path:", output["data_path"])
    print("Cleaning report:")
    for key, value in output["cleaning_report"].items():
        print(f"- {key}: {value}")
    print("\nClean data preview:")
    print(output["df"].head())
