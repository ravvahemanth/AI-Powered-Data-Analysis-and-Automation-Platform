import pandas as pd


def apply_suggestion(df, suggestion_type, column=None):
    df = df.copy()

    if suggestion_type == "Duplicate Rows":
        df = df.drop_duplicates(ignore_index=True)

    elif suggestion_type == "Missing Values" and column in df.columns:
        if pd.api.types.is_object_dtype(df[column]) or pd.api.types.is_string_dtype(df[column]):
            mode_val = df[column].dropna().mode()
            if not mode_val.empty:
                df[column] = df[column].fillna(mode_val[0])
        elif pd.api.types.is_numeric_dtype(df[column]):
            df[column] = df[column].fillna(df[column].median())

    elif suggestion_type == "Text Formatting" and column in df.columns:
        df[column] = df[column].apply(lambda x: x.strip() if isinstance(x, str) else x)

    elif suggestion_type == "Inconsistent Text Case" and column in df.columns:
        df[column] = df[column].apply(lambda x: x.strip().lower() if isinstance(x, str) else x)

    elif suggestion_type == "Possible Date Column" and column in df.columns:
        df[column] = pd.to_datetime(df[column], errors="coerce")

    return df


def apply_all_safe_suggestions(df, suggestions_df):
    updated_df = df.copy()

    for _, row in suggestions_df.iterrows():
        if row["confidence"] in ["High", "Medium"]:
            updated_df = apply_suggestion(
                updated_df,
                row["type"],
                row["column"] if row["column"] != "All Rows" else None
            )

    return updated_df