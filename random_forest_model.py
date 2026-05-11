
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
from sklearn.preprocessing import OneHotEncoder, StandardScaler


HIGH_SPENDER_FEATURES = ["Gender", "Age", "Product Category", "Quantity"]
CATEGORY_FEATURES = ["Gender", "Age", "Quantity", "Price per Unit"]


def _clean_column_names(df):
    df = df.copy()
    df.columns = df.columns.astype(str).str.strip()
    return df


def _check_columns(df, columns):
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for Random Forest model: {missing}. Available columns: {list(df.columns)}")


def _make_ohe():
    # Compatible with new and old scikit-learn versions.
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def train_random_forest_high_spender(df):
    df = _clean_column_names(df)
    needed_cols = HIGH_SPENDER_FEATURES + ["Total Amount"]
    _check_columns(df, needed_cols)

    model_df = df.dropna(subset=needed_cols).copy()
    high_threshold = model_df["Total Amount"].quantile(0.65)
    model_df["High Spender Binary"] = (model_df["Total Amount"] >= high_threshold).astype(int)

    X = model_df[HIGH_SPENDER_FEATURES].copy()
    y = model_df["High Spender Binary"].copy()

    categorical_features = ["Gender", "Product Category"]
    numeric_features = ["Age", "Quantity"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", _make_ohe(), categorical_features),
            ("num", StandardScaler(), numeric_features),
        ],
        remainder="drop",
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=120,
                    max_depth=12,
                    min_samples_split=8,
                    min_samples_leaf=4,
                    random_state=42,
                    class_weight="balanced",
                    n_jobs=-1,
                ),
            ),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.35, random_state=42, stratify=y
    )

    model.fit(X_train, y_train)
    probabilities = model.predict_proba(X_test)[:, 1]

    thresholds = np.arange(0.20, 0.81, 0.01)
    threshold_scores = [f1_score(y_test, (probabilities >= t).astype(int), zero_division=0) for t in thresholds]
    best_threshold = float(thresholds[int(np.argmax(threshold_scores))])
    predictions = (probabilities >= best_threshold).astype(int)

    # Cross-validation on very large Streamlit datasets can be slow, so the app reports holdout-test metrics.
    cv_accuracy = np.nan

    metrics = {
        "Accuracy": float(accuracy_score(y_test, predictions)),
        "Balanced Accuracy": float(balanced_accuracy_score(y_test, predictions)),
        "Precision": float(precision_score(y_test, predictions, zero_division=0)),
        "Recall": float(recall_score(y_test, predictions, zero_division=0)),
        "F1 Score": float(f1_score(y_test, predictions, zero_division=0)),
        "ROC AUC": float(roc_auc_score(y_test, probabilities)) if len(np.unique(y_test)) > 1 else np.nan,
        "CV Accuracy": cv_accuracy,
        "High Spender Threshold": float(high_threshold),
        "Best Probability Cutoff": best_threshold,
        "Features Used": ", ".join(HIGH_SPENDER_FEATURES),
        "Leakage Fix": "Price per Unit removed from High-Spender features",
    }

    cm = confusion_matrix(y_test, predictions)
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
    result_df["Actual High Spender"] = np.where(y_test.values == 1, "High", "Normal/Low")
    result_df["Predicted High Spender"] = np.where(predictions == 1, "High", "Normal/Low")
    result_df["High Spending Probability"] = probabilities

    return {
        "model": model,
        "X_test": X_test,
        "y_test": y_test,
        "predictions": predictions,
        "probabilities": probabilities,
        "metrics": metrics,
        "confusion_matrix": cm,
        "classification_report": report,
        "result_df": result_df,
    }


def train_random_forest_category(df):
    df = _clean_column_names(df)
    needed_cols = CATEGORY_FEATURES + ["Product Category"]
    _check_columns(df, needed_cols)

    model_df = df.dropna(subset=needed_cols).copy()

    X = model_df[CATEGORY_FEATURES].copy()
    y = model_df["Product Category"].copy()

    categorical_features = ["Gender"]
    numeric_features = ["Age", "Quantity", "Price per Unit"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", _make_ohe(), categorical_features),
            ("num", StandardScaler(), numeric_features),
        ],
        remainder="drop",
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=120,
                    max_depth=14,
                    min_samples_split=8,
                    min_samples_leaf=3,
                    random_state=42,
                    class_weight="balanced",
                    n_jobs=-1,
                ),
            ),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.35, random_state=42, stratify=y
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    # Cross-validation on very large Streamlit datasets can be slow, so the app reports holdout-test metrics.
    cv_accuracy = np.nan

    metrics = {
        "Accuracy": float(accuracy_score(y_test, predictions)),
        "Balanced Accuracy": float(balanced_accuracy_score(y_test, predictions)),
        "Precision Macro": float(precision_score(y_test, predictions, average="macro", zero_division=0)),
        "Recall Macro": float(recall_score(y_test, predictions, average="macro", zero_division=0)),
        "F1 Macro": float(f1_score(y_test, predictions, average="macro", zero_division=0)),
        "CV Accuracy": cv_accuracy,
        "Features Used": ", ".join(CATEGORY_FEATURES),
    }

    cm = confusion_matrix(y_test, predictions, labels=model.classes_)
    report = pd.DataFrame(classification_report(y_test, predictions, output_dict=True, zero_division=0)).transpose()

    result_df = X_test.copy()
    result_df["Actual Category"] = y_test.values
    result_df["Predicted Category"] = predictions

    return {
        "model": model,
        "X_test": X_test,
        "y_test": y_test,
        "predictions": predictions,
        "metrics": metrics,
        "confusion_matrix": cm,
        "classification_report": report,
        "classes": model.classes_,
        "result_df": result_df,
    }
