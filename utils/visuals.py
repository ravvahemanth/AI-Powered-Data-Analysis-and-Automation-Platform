import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_missing_values(df):
    missing = df.isna().sum().sort_values(ascending=False)
    missing = missing[missing > 0]

    if missing.empty:
        return None

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(missing.index.astype(str), missing.values, color="#4f98a3")
    ax.set_title("Missing Values by Column")
    ax.set_xlabel("Columns")
    ax.set_ylabel("Missing Count")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    return fig


def plot_correlation_heatmap(df):
    numeric_df = df.select_dtypes(include="number")

    if numeric_df.shape[1] < 2:
        return None

    corr = numeric_df.corr(numeric_only=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation Heatmap")
    fig.tight_layout()
    return fig


def plot_histogram(df, column):
    if column not in df.columns:
        return None

    if not pd.api.types.is_numeric_dtype(df[column]):
        return None

    series = df[column].dropna()
    if series.empty:
        return None

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(series, bins=20, color="#01696f", edgecolor="white")
    ax.set_title(f"Histogram of {column}")
    ax.set_xlabel(column)
    ax.set_ylabel("Frequency")
    fig.tight_layout()
    return fig


def plot_boxplot(df, column):
    if column not in df.columns:
        return None

    if not pd.api.types.is_numeric_dtype(df[column]):
        return None

    series = df[column].dropna()
    if series.empty:
        return None

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.boxplot(y=series, ax=ax, color="#4f98a3")
    ax.set_title(f"Box Plot of {column}")
    ax.set_ylabel(column)
    fig.tight_layout()
    return fig


def top_categories_table(df, column, top_n=10):
    if column not in df.columns:
        return pd.DataFrame(columns=["category", "count"])

    series = df[column].dropna().astype(str)
    if series.empty:
        return pd.DataFrame(columns=["category", "count"])

    counts = series.value_counts().head(top_n)
    result = counts.rename_axis("category").reset_index(name="count")
    return result