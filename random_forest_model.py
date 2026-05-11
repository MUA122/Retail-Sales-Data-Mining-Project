import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


HIGH_SPENDER_FEATURES = ["Gender", "Age", "Product Category", "Quantity"]
CATEGORY_FEATURES = ["Gender", "Age", "Quantity", "Price per Unit"]


def _clean_columns(df):
    df = df.copy()
    df.columns = df.columns.astype(str).str.strip()
    return df


def _check_columns(df, cols):
    missing = [col for col in cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}. Available columns: {list(df.columns)}")


def _make_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _make_pipeline(categorical_features, numeric_features):
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", _make_encoder(), categorical_features),
            ("num", "passthrough", numeric_features),
        ],
        remainder="drop",
    )

    classifier = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def _feature_importance(model):
    rf = model.named_steps["classifier"]
    preprocessor = model.named_steps["preprocessor"]

    feature_names = preprocessor.get_feature_names_out()

    importance_df = pd.DataFrame(
        {
            "Feature": feature_names,
            "Importance": rf.feature_importances_,
        }
    ).sort_values("Importance", ascending=False)

    return importance_df.reset_index(drop=True)


def train_random_forest_high_spender(df):
    df = _clean_columns(df)

    needed_cols = HIGH_SPENDER_FEATURES + ["Total Amount"]
    _check_columns(df, needed_cols)

    model_df = df.dropna(subset=needed_cols).copy()

    # High Spender = top 35% transactions by Total Amount
    high_threshold = model_df["Total Amount"].quantile(0.65)
    model_df["High Spender Binary"] = (
        model_df["Total Amount"] >= high_threshold
    ).astype(int)

    X = model_df[HIGH_SPENDER_FEATURES].copy()
    y = model_df["High Spender Binary"].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    model = _make_pipeline(
        categorical_features=["Gender", "Product Category"],
        numeric_features=["Age", "Quantity"],
    )

    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_test)[:, 1]

    cutoff = 0.50
    predictions = (probabilities >= cutoff).astype(int)

    metrics = {
        "Accuracy": float(accuracy_score(y_test, predictions)),
        "Balanced Accuracy": float(balanced_accuracy_score(y_test, predictions)),
        "Precision": float(precision_score(y_test, predictions, zero_division=0)),
        "Recall": float(recall_score(y_test, predictions, zero_division=0)),
        "F1 Score": float(f1_score(y_test, predictions, zero_division=0)),
        "ROC AUC": float(roc_auc_score(y_test, probabilities)),
        "High Spender Threshold": float(high_threshold),
        "Best Probability Cutoff": float(cutoff),
        "Features Used": ", ".join(HIGH_SPENDER_FEATURES),
        "Leakage Note": "Total Amount and Price per Unit are excluded from High-Spender features",
    }

    report = pd.DataFrame(
        classification_report(
            y_test,
            predictions,
            target_names=["Normal/Low", "High"],
            output_dict=True,
            zero_division=0,
        )
    ).transpose()

    result_df = X_test.copy()
    result_df["Actual High Spender"] = np.where(
        y_test.values == 1,
        "High",
        "Normal/Low",
    )
    result_df["Predicted High Spender"] = np.where(
        predictions == 1,
        "High",
        "Normal/Low",
    )
    result_df["High Spending Probability"] = probabilities

    threshold_rows = []

    for threshold in np.arange(0.30, 0.81, 0.05):
        threshold_preds = (probabilities >= threshold).astype(int)

        threshold_rows.append(
            {
                "Cutoff": round(float(threshold), 2),
                "Precision": precision_score(y_test, threshold_preds, zero_division=0),
                "Recall": recall_score(y_test, threshold_preds, zero_division=0),
                "F1 Score": f1_score(y_test, threshold_preds, zero_division=0),
            }
        )

    return {
        "model": model,
        "metrics": metrics,
        "confusion_matrix": confusion_matrix(y_test, predictions),
        "classification_report": report,
        "result_df": result_df,
        "feature_importance": _feature_importance(model),
        "threshold_table": pd.DataFrame(threshold_rows),
    }


def train_random_forest_category(df):
    df = _clean_columns(df)

    needed_cols = CATEGORY_FEATURES + ["Product Category"]
    _check_columns(df, needed_cols)

    model_df = df.dropna(subset=needed_cols).copy()

    X = model_df[CATEGORY_FEATURES].copy()
    y = model_df["Product Category"].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    model = _make_pipeline(
        categorical_features=["Gender"],
        numeric_features=["Age", "Quantity", "Price per Unit"],
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    metrics = {
        "Accuracy": float(accuracy_score(y_test, predictions)),
        "Balanced Accuracy": float(balanced_accuracy_score(y_test, predictions)),
        "F1 Weighted": float(
            f1_score(y_test, predictions, average="weighted", zero_division=0)
        ),
        "Features Used": ", ".join(CATEGORY_FEATURES),
    }

    report = pd.DataFrame(
        classification_report(
            y_test,
            predictions,
            output_dict=True,
            zero_division=0,
        )
    ).transpose()

    result_df = X_test.copy()
    result_df["Actual Category"] = y_test.values
    result_df["Predicted Category"] = predictions

    return {
        "model": model,
        "metrics": metrics,
        "classification_report": report,
        "result_df": result_df,
        "feature_importance": _feature_importance(model),
    }