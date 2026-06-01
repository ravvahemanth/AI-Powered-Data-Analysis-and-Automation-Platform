import pandas as pd


def split_column(df, column, delimiter=" ", max_splits=-1, prefix=None):
    df = df.copy()

    if column not in df.columns:
        return df, f"Column '{column}' not found."

    split_df = df[column].astype(str).str.split(delimiter, n=max_splits, expand=True)

    if prefix is None or prefix.strip() == "":
        prefix = f"{column}_part"

    split_df.columns = [f"{prefix}_{i+1}" for i in range(split_df.shape[1])]

    # Drop columns that are entirely null or contain only "None" strings — these are
    # empty trailing splits (e.g. splitting "4.1/5" by "/" with no max gives part_1 and part_2,
    # but a value like "NEW" with no "/" produces part_1="NEW" and part_2=None)
    valid_cols = [
        col for col in split_df.columns
        if not split_df[col].replace("None", pd.NA).isna().all()
    ]
    split_df = split_df[valid_cols]

    for col in split_df.columns:
        df[col] = split_df[col]

    return df, f"Column '{column}' split into {len(valid_cols)} new columns."


def extract_numeric_from_column(df, column, delimiter="/", replace_original=False):
    """
    Splits a column by delimiter, keeps only the first part, converts to float.
    Handles missing/invalid values safely as NaN.
    e.g. '4.1/5' -> 4.1,  'NEW' -> NaN,  '' -> NaN
    """
    df = df.copy()

    if column not in df.columns:
        return df, f"Column '{column}' not found."

    extracted = (
        df[column]
        .astype(str)
        .str.strip()
        .str.split(delimiter)
        .str[0]
        .str.strip()
        .replace({"nan": None, "none": None, "": None, "NEW": None, "new": None})
    )
    extracted = pd.to_numeric(extracted, errors="coerce")

    if replace_original:
        df[column] = extracted
        out_col = column
    else:
        out_col = f"{column}_clean"
        df[out_col] = extracted

    return df, f"Extracted numeric values into '{out_col}' ({extracted.notna().sum()} valid, {extracted.isna().sum()} NaN)."


def merge_columns(df, col1, col2, new_column, separator=" "):
    df = df.copy()

    if col1 not in df.columns or col2 not in df.columns:
        return df, "One or both columns were not found."

    df[new_column] = df[col1].astype(str).fillna("") + separator + df[col2].astype(str).fillna("")
    df[new_column] = df[new_column].str.strip()

    return df, f"Columns '{col1}' and '{col2}' merged into '{new_column}'."


def derive_date_feature(df, column, feature_type):
    df = df.copy()

    if column not in df.columns:
        return df, f"Column '{column}' not found."

    df[column] = pd.to_datetime(df[column], errors="coerce")

    if feature_type == "year":
        df[f"{column}_year"] = df[column].dt.year
    elif feature_type == "month":
        df[f"{column}_month"] = df[column].dt.month
    elif feature_type == "day":
        df[f"{column}_day"] = df[column].dt.day
    elif feature_type == "month_year":
        df[f"{column}_month_year"] = df[column].dt.to_period("M").astype(str)
    else:
        return df, "Unsupported date feature type."

    return df, f"Derived '{feature_type}' feature from '{column}'."