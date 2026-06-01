import pandas as pd


def trim_text_columns(df):
    df = df.copy()
    text_cols = df.select_dtypes(include=["object", "string"]).columns
    for col in text_cols:
        df[col] = df[col].apply(lambda x: x.strip() if isinstance(x, str) else x)
    return df


def fill_missing_values(df):
    df = df.copy()
    for col in df.columns:
        if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
            mode_val = df[col].dropna().mode()
            if not mode_val.empty:
                df[col] = df[col].fillna(mode_val[0])
        elif pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            # Fill datetime NaT with the median (middle) timestamp
            valid = df[col].dropna()
            if not valid.empty:
                median_ts = valid.sort_values().iloc[len(valid) // 2]
                df[col] = df[col].fillna(median_ts)
    return df


def convert_date_columns(df):
    df = df.copy()
    for col in df.columns:
        if "date" in col.lower() or "time" in col.lower():
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def fill_column_missing(df, column):
    df = df.copy()
    if pd.api.types.is_object_dtype(df[column]) or pd.api.types.is_string_dtype(df[column]):
        mode_val = df[column].dropna().mode()
        if not mode_val.empty:
            df[column] = df[column].fillna(mode_val[0])
    elif pd.api.types.is_numeric_dtype(df[column]):
        df[column] = df[column].fillna(df[column].median())
    return df


def add_new_column(df, column_name, default_value=""):
    df = df.copy()
    df[column_name] = default_value
    return df


def drop_selected_column(df, column):
    df = df.copy()
    df = df.drop(columns=[column])
    return df


def rename_column(df, old_name, new_name):
    df = df.copy()
    df = df.rename(columns={old_name: new_name})
    return df