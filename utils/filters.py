import pandas as pd


def filter_by_categories(df, column, selected_values):
    if column not in df.columns or not selected_values:
        return df
    return df[df[column].astype(str).isin([str(v) for v in selected_values])]


def filter_by_numeric_range(df, column, min_val, max_val):
    if column not in df.columns:
        return df
    if not pd.api.types.is_numeric_dtype(df[column]):
        return df
    return df[df[column].between(min_val, max_val, inclusive="both")]


def filter_by_text_search(df, column, search_text):
    if column not in df.columns or not search_text:
        return df
    return df[df[column].astype(str).str.contains(search_text, case=False, na=False)]


def apply_all_filters(df, category_column=None, category_values=None,
                      numeric_column=None, min_val=None, max_val=None,
                      text_column=None, search_text=None):
    filtered_df = df.copy()

    if category_column and category_values:
        filtered_df = filter_by_categories(filtered_df, category_column, category_values)

    if numeric_column and min_val is not None and max_val is not None:
        filtered_df = filter_by_numeric_range(filtered_df, numeric_column, min_val, max_val)

    if text_column and search_text:
        filtered_df = filter_by_text_search(filtered_df, text_column, search_text)

    return filtered_df