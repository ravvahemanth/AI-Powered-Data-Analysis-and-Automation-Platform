import pandas as pd
import numpy as np


def detect_outliers_iqr(df, column):
    if not pd.api.types.is_numeric_dtype(df[column]):
        return pd.DataFrame()

    series = df[column].dropna()
    if series.empty:
        return pd.DataFrame()

    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1

    if iqr == 0:
        return pd.DataFrame()

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return df[(df[column] < lower) | (df[column] > upper)]


def data_quality_summary(df):
    total_rows, total_cols = df.shape
    duplicate_rows = int(df.duplicated().sum())
    missing_cells = int(df.isna().sum().sum())
    missing_percent = round((missing_cells / (total_rows * total_cols)) * 100, 2) if total_rows * total_cols > 0 else 0

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    text_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()

    return {
        "Rows": total_rows,
        "Columns": total_cols,
        "Duplicate Rows": duplicate_rows,
        "Missing Cells": missing_cells,
        "Missing %": missing_percent,
        "Numeric Columns": len(numeric_cols),
        "Text Columns": len(text_cols)
    }