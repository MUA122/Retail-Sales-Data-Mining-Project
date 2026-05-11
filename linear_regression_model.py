
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def train_linear_regression(df):
    model_df = df.dropna(
        subset=["Gender", "Product Category", "Age", "Quantity", "Price per Unit", "Total Amount"]
    ).copy()

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

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    errors = y_test.values - predictions

    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)

    metrics = {
        "R2 Score": float(r2_score(y_test, predictions)),
        "Explained Variance": float(explained_variance_score(y_test, predictions)),
        "MSE": float(mse),
        "RMSE": float(rmse),
        "MAE": float(mean_absolute_error(y_test, predictions)),
        "Mean Error": float(np.mean(errors)),
        "Median Absolute Error": float(np.median(np.abs(errors))),
        "Max Error": float(np.max(np.abs(errors))),
    }

    result_df = X_test.copy()
    result_df["Actual Total Amount"] = y_test.values
    result_df["Predicted Total Amount"] = predictions
    result_df["Error"] = errors
    result_df["Absolute Error"] = np.abs(errors)

    # Get readable coefficients after preprocessing.
    preprocessor_fitted = model.named_steps["preprocessor"]
    regressor = model.named_steps["regressor"]
    feature_names = preprocessor_fitted.get_feature_names_out()
    coefficients = pd.DataFrame(
        {
            "Feature": feature_names,
            "Coefficient": regressor.coef_,
        }
    ).sort_values("Coefficient", ascending=False)

    return {
        "model": model,
        "X_test": X_test,
        "y_test": y_test,
        "predictions": predictions,
        "metrics": metrics,
        "result_df": result_df,
        "coefficients": coefficients,
    }
