import pandas as pd
import numpy as np


def get_top_correlated_pairs(df, top_n=10, min_abs_corr=0.5):
    numeric_df = df.select_dtypes(include=np.number)

    if numeric_df.shape[1] < 2:
        return pd.DataFrame(columns=["column_1", "column_2", "correlation"])

    corr_matrix = numeric_df.corr().abs()
    pairs = corr_matrix.unstack().reset_index()
    pairs.columns = ["column_1", "column_2", "correlation"]

    pairs = pairs[pairs["column_1"] != pairs["column_2"]]
    pairs["pair_key"] = pairs.apply(
        lambda row: tuple(sorted([row["column_1"], row["column_2"]])),
        axis=1
    )
    pairs = pairs.drop_duplicates(subset="pair_key").drop(columns=["pair_key"])
    pairs = pairs[pairs["correlation"] >= min_abs_corr]
    pairs = pairs.sort_values(by="correlation", ascending=False).head(top_n)

    return pairs.reset_index(drop=True)


def detect_suspicious_columns(df, missing_threshold=0.5):
    results = []
    total_rows = len(df)

    for col in df.columns:
        missing_ratio = df[col].isna().mean() if total_rows > 0 else 0
        unique_count = df[col].nunique(dropna=True)

        if unique_count <= 1:
            results.append({
                "column": col,
                "issue": "Single unique value / constant column",
                "severity": "High"
            })
            continue  # no need to check further for this column

        if missing_ratio >= missing_threshold:
            results.append({
                "column": col,
                "issue": f"High missing values ({missing_ratio:.0%})",
                "severity": "High"
            })

        # Only flag binary columns that are NOT already flagged for missing values
        if unique_count == 2 and missing_ratio < missing_threshold:
            results.append({
                "column": col,
                "issue": "Binary / low-cardinality column — may be a label or flag",
                "severity": "Info"
            })

    return pd.DataFrame(results)


def recommend_chart_for_column(df, column):
    if column not in df.columns:
        return "Unknown"

    series = df[column]

    if pd.api.types.is_numeric_dtype(series):
        return "Histogram, box plot, line chart"

    if pd.api.types.is_datetime64_any_dtype(series):
        return "Line chart by time, monthly trend chart"

    unique_count = series.nunique(dropna=True)

    if unique_count <= 15:
        return "Bar chart, count plot"

    return "Top categories bar chart, frequency table"


def suggest_target_columns(df):
    suggestions = []

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            if df[col].nunique(dropna=True) > 10:
                suggestions.append({
                    "column": col,
                    "reason": "Continuous numeric column, suitable for regression or trend analysis"
                })
            elif 2 <= df[col].nunique(dropna=True) <= 10:
                suggestions.append({
                    "column": col,
                    "reason": "Low-cardinality numeric column, may be suitable for classification"
                })

        elif pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
            nunique = df[col].nunique(dropna=True)
            if 2 <= nunique <= 20:
                suggestions.append({
                    "column": col,
                    "reason": "Categorical column with manageable classes, may be suitable as a target"
                })

    return pd.DataFrame(suggestions)