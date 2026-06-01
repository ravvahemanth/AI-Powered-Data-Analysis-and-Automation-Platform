import pandas as pd


def convert_column_type(df, column, target_type):
    df = df.copy()

    if column not in df.columns:
        return df, f"Column '{column}' not found."

    try:
        if target_type == "string":
            df[column] = df[column].astype("string")

        elif target_type == "integer":
            df[column] = pd.to_numeric(df[column], errors="coerce").astype("Int64")

        elif target_type == "float":
            df[column] = pd.to_numeric(df[column], errors="coerce")

        elif target_type == "datetime":
            df[column] = pd.to_datetime(df[column], errors="coerce")

        elif target_type == "category":
            df[column] = df[column].astype("category")

        else:
            return df, "Unsupported target type."

        return df, f"Column '{column}' converted to {target_type} successfully."

    except Exception as e:
        return df, f"Conversion failed: {e}"