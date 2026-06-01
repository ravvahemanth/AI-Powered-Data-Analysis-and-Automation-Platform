import pandas as pd


def make_unique_columns(columns):
    seen = {}
    new_columns = []
    rename_report = []

    for col in columns:
        col = str(col).strip()
        if col == "" or col.lower() in ["nan", "none"]:
            col = "unnamed_column"

        base = col
        if base in seen:
            seen[base] += 1
            new_name = f"{base}_{seen[base]}"
            rename_report.append((base, new_name))
        else:
            seen[base] = 1
            new_name = base

        new_columns.append(new_name)

    return new_columns, rename_report


def standardize_and_uniquify_columns(df):
    df = df.copy()

    cleaned_cols = (
        pd.Index(df.columns)
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace(r"[^a-zA-Z0-9_]", "", regex=True)
    )

    cleaned_cols = [col if col not in ["", "nan", "none"] else "unnamed_column" for col in cleaned_cols]
    unique_cols, rename_report = make_unique_columns(cleaned_cols)

    df.columns = unique_cols
    return df, rename_report