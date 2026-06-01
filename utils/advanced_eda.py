import pandas as pd
import numpy as np
import plotly.express as px


def create_scatter_plot(df, x_col, y_col, color_col=None):
    if x_col not in df.columns or y_col not in df.columns:
        return None

    plot_df = df.copy()

    fig = px.scatter(
        plot_df,
        x=x_col,
        y=y_col,
        color=color_col if color_col in plot_df.columns else None,
        title=f"Scatter Plot: {x_col} vs {y_col}",
        opacity=0.7,
    )
    return fig


def create_grouped_summary(df, group_col, value_col, agg_func="mean"):
    if group_col not in df.columns or value_col not in df.columns:
        return pd.DataFrame()

    grouped = (
        df.groupby(group_col)[value_col]
        .agg(agg_func)
        .reset_index()
        .rename(columns={value_col: f"{agg_func}_{value_col}"})
    )
    return grouped


def create_grouped_bar_chart(summary_df, x_col, y_col):
    if summary_df.empty or x_col not in summary_df.columns or y_col not in summary_df.columns:
        return None

    fig = px.bar(
        summary_df,
        x=x_col,
        y=y_col,
        title=f"Grouped Summary: {y_col} by {x_col}",
    )
    return fig


def create_date_trend(df, date_col, value_col, freq="ME"):
    if date_col not in df.columns or value_col not in df.columns:
        return pd.DataFrame(), None

    temp_df = df.copy()
    temp_df[date_col] = pd.to_datetime(temp_df[date_col], errors="coerce")
    temp_df = temp_df.dropna(subset=[date_col, value_col])

    if temp_df.empty:
        return pd.DataFrame(), None

    # Map legacy single-letter aliases to pandas 2.x month-end anchors
    freq_map = {"M": "ME", "Q": "QE", "Y": "YE"}
    freq = freq_map.get(freq, freq)

    trend_df = (
        temp_df.set_index(date_col)
        .resample(freq)[value_col]
        .mean()
        .reset_index()
    )

    trend_df = trend_df.sort_values(by=date_col)

    fig = px.line(
        trend_df,
        x=date_col,
        y=value_col,
        title=f"Trend of {value_col} over time",
        markers=True,
    )

    return trend_df, fig