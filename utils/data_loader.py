import pandas as pd
from utils.helpers import standardize_and_uniquify_columns


def load_data(file):
    name = file.name.lower()

    if name.endswith(".csv"):
        # Try UTF-8 first, fall back to latin-1 for files with special characters
        try:
            df = pd.read_csv(file, encoding="utf-8")
        except UnicodeDecodeError:
            file.seek(0)
            df = pd.read_csv(file, encoding="latin-1")

    elif name.endswith(".xlsx"):
        df = pd.read_excel(file, engine="openpyxl")

    elif name.endswith(".xls"):
        df = pd.read_excel(file, engine="xlrd")

    else:
        raise ValueError(f"Unsupported file type: {file.name}")

    df, rename_report = standardize_and_uniquify_columns(df)
    return df, rename_report
