import pandas as pd
import numpy as np
import joblib
from io import BytesIO

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


def detect_problem_type(df, target_col):
    if target_col not in df.columns:
        return "unknown"

    target = df[target_col]

    if pd.api.types.is_numeric_dtype(target):
        unique_count = target.nunique(dropna=True)
        if unique_count <= 10:
            return "classification"
        return "regression"

    return "classification"


def prepare_features_and_target(df, target_col):
    model_df = df.copy()

    if target_col not in model_df.columns:
        return None, None

    model_df = model_df.dropna(subset=[target_col])
    X = model_df.drop(columns=[target_col])
    y = model_df[target_col]

    return X, y


def build_preprocessor(X):
    numeric_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "string", "category", "bool"]).columns.tolist()

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )

    return preprocessor


def get_model_candidates(problem_type):
    if problem_type == "classification":
        return {
            "LogisticRegression": LogisticRegression(max_iter=1000),
            "DecisionTreeClassifier": DecisionTreeClassifier(random_state=42),
            "RandomForestClassifier": RandomForestClassifier(random_state=42),
        }
    else:
        return {
            "LinearRegression": LinearRegression(),
            "Ridge": Ridge(),
            "DecisionTreeRegressor": DecisionTreeRegressor(random_state=42),
            "RandomForestRegressor": RandomForestRegressor(random_state=42),
        }


def evaluate_predictions(problem_type, y_test, predictions):
    if problem_type == "classification":
        return {
            "accuracy": float(accuracy_score(y_test, predictions)),
            "f1_score_weighted": float(f1_score(y_test, predictions, average="weighted")),
        }
    else:
        rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
        return {
            "mae": float(mean_absolute_error(y_test, predictions)),
            "rmse": rmse,
            "r2_score": float(r2_score(y_test, predictions)),
        }


def train_baseline_model(df, target_col, model_name="auto", test_size=0.2, random_state=42):
    X, y = prepare_features_and_target(df, target_col)

    if X is None or y is None or X.empty:
        return {"error": "Could not prepare features and target."}

    problem_type = detect_problem_type(df, target_col)

    if y.nunique(dropna=True) < 2:
        return {"error": "Target column must have at least 2 unique values."}

    preprocessor = build_preprocessor(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    if problem_type == "classification":
        if model_name == "random_forest":
            model = RandomForestClassifier(random_state=random_state)
            model_used = "RandomForestClassifier"
        else:
            model = LogisticRegression(max_iter=1000)
            model_used = "LogisticRegression"
    else:
        if model_name == "random_forest":
            model = RandomForestRegressor(random_state=random_state)
            model_used = "RandomForestRegressor"
        else:
            model = LinearRegression()
            model_used = "LinearRegression"

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    metrics = evaluate_predictions(problem_type, y_test, predictions)

    metrics["problem_type"] = problem_type
    metrics["model_used"] = model_used

    preview_df = X_test.copy().head(20).reset_index(drop=True)
    preview_df["actual_target"] = y_test.reset_index(drop=True).head(20)
    preview_df["predicted_target"] = pd.Series(predictions).head(20)

    return {
        "metrics": metrics,
        "preview": preview_df,
        "problem_type": problem_type,
        "model_used": model_used,
        "pipeline": pipeline,
        "feature_columns": X.columns.tolist(),
        "target_column": target_col,
    }


def compare_models(df, target_col, test_size=0.2, random_state=42):
    X, y = prepare_features_and_target(df, target_col)

    if X is None or y is None or X.empty:
        return {"error": "Could not prepare features and target."}

    problem_type = detect_problem_type(df, target_col)

    if y.nunique(dropna=True) < 2:
        return {"error": "Target column must have at least 2 unique values."}

    model_candidates = get_model_candidates(problem_type)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    results = []
    fitted_pipelines = {}

    for model_name, model in model_candidates.items():
        # Build a fresh preprocessor for each pipeline to avoid shared state
        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(X)),
                ("model", model),
            ]
        )

        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        metrics = evaluate_predictions(problem_type, y_test, predictions)

        row = {"model": model_name}
        row.update(metrics)
        results.append(row)
        fitted_pipelines[model_name] = pipeline

    results_df = pd.DataFrame(results)

    if problem_type == "classification":
        best_model_name = results_df.sort_values(
            by=["accuracy", "f1_score_weighted"],
            ascending=False
        ).iloc[0]["model"]
    else:
        best_model_name = results_df.sort_values(
            by=["r2_score", "rmse"],
            ascending=[False, True]
        ).iloc[0]["model"]

    best_pipeline = fitted_pipelines[best_model_name]
    best_predictions = best_pipeline.predict(X_test)

    preview_df = X_test.copy().head(20).reset_index(drop=True)
    preview_df["actual_target"] = y_test.reset_index(drop=True).head(20)
    preview_df["predicted_target"] = pd.Series(best_predictions).head(20)

    return {
        "problem_type": problem_type,
        "results_df": results_df,
        "best_model_name": best_model_name,
        "best_pipeline": best_pipeline,
        "preview": preview_df,
        "feature_columns": X.columns.tolist(),
        "target_column": target_col,
    }


def extract_feature_importance(pipeline, top_n=15):
    try:
        preprocessor = pipeline.named_steps["preprocessor"]
        model = pipeline.named_steps["model"]

        feature_names = preprocessor.get_feature_names_out()

        if hasattr(model, "feature_importances_"):
            importance_values = model.feature_importances_
        elif hasattr(model, "coef_"):
            coef = model.coef_
            if np.ndim(coef) > 1:
                importance_values = np.mean(np.abs(coef), axis=0)
            else:
                importance_values = np.abs(coef)
        else:
            return pd.DataFrame(columns=["feature", "importance"])

        importance_df = pd.DataFrame({
            "feature": feature_names,
            "importance": importance_values
        }).sort_values(by="importance", ascending=False).head(top_n)

        return importance_df.reset_index(drop=True)
    except Exception:
        return pd.DataFrame(columns=["feature", "importance"])


def save_pipeline_to_bytes(pipeline):
    buffer = BytesIO()
    joblib.dump(pipeline, buffer)
    buffer.seek(0)
    return buffer.getvalue()


def predict_with_pipeline(pipeline, new_df, required_columns):
    predict_df = new_df.copy()

    missing_cols = [col for col in required_columns if col not in predict_df.columns]
    if missing_cols:
        return None, f"Missing required columns: {', '.join(missing_cols)}"

    predict_df = predict_df[required_columns].copy()
    predictions = pipeline.predict(predict_df)

    result_df = new_df.copy()
    result_df["prediction"] = predictions

    return result_df, "Predictions generated successfully."