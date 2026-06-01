import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from utils.filters import apply_all_filters
from utils.data_loader import load_data
from utils.cleaners import (
    trim_text_columns,
    fill_missing_values,
    convert_date_columns,
    fill_column_missing,
    add_new_column,
    drop_selected_column,
    rename_column,
)
from utils.visuals import (
    plot_missing_values,
    plot_correlation_heatmap,
    plot_histogram,
    plot_boxplot,
    top_categories_table,
)
from utils.helpers import standardize_and_uniquify_columns
from utils.analysis import detect_outliers_iqr, data_quality_summary
from utils.ai_suggestions import generate_ai_suggestions
from utils.suggestion_actions import apply_suggestion, apply_all_safe_suggestions
from utils.type_converter import convert_column_type
from utils.feature_engineering import split_column, merge_columns, derive_date_feature, extract_numeric_from_column
from utils.exporter import (
    dataframe_to_csv_bytes,
    dataframe_to_excel_bytes,
    history_to_csv_bytes,
    create_full_html_report,
)
from utils.state_manager import (
    initialize_session_state,
    save_state,
    undo_last_change,
    reset_to_original,
    log_action,
)
from utils.ml_module import (
    train_baseline_model,
    detect_problem_type,
    save_pipeline_to_bytes,
    predict_with_pipeline,
    compare_models,
    extract_feature_importance,
)
from utils.smart_insights import (
    get_top_correlated_pairs,
    detect_suspicious_columns,
    recommend_chart_for_column,
    suggest_target_columns,
)
from utils.advanced_eda import (
    create_scatter_plot,
    create_grouped_summary,
    create_grouped_bar_chart,
    create_date_trend,
)

st.set_page_config(page_title="AI Data Analysis Model", layout="wide")

initialize_session_state()

st.title("AI Data Analysis Model")
st.write("Upload a CSV or Excel file to clean, edit, analyze, and build ML workflows.")

uploaded_file = st.file_uploader(
    "Choose a CSV or Excel file",
    type=["csv", "xlsx", "xls"],
    key="file_uploader"
)

if uploaded_file is not None:
    if st.session_state["last_uploaded_filename"] != uploaded_file.name:
        try:
            df, rename_report = load_data(uploaded_file)
            st.session_state["df"] = df.copy(deep=True)
            st.session_state["original_df"] = df.copy(deep=True)
            st.session_state["history"] = []
            st.session_state["rename_report"] = rename_report
            st.session_state["last_uploaded_filename"] = uploaded_file.name
            st.session_state["action_log"] = []
            log_action("File Upload", f"Loaded file '{uploaded_file.name}'")
        except Exception as e:
            st.error(f"Error reading file: {e}")

if st.session_state["df"] is not None:
    df = st.session_state["df"]

    top1, top2 = st.columns(2)

    with top1:
        if st.button("Undo Last Change"):
            if undo_last_change():
                log_action("Undo", "Reverted the last change")
                st.success("Last change undone successfully.")
                st.rerun()
            else:
                st.warning("No previous change to undo.")

    with top2:
        if st.button("Reset to Original File"):
            reset_to_original()
            log_action("Reset", "Reset data to original uploaded file")
            st.success("Data reset to the original uploaded file.")
            st.rerun()

    if st.session_state["rename_report"]:
        st.warning("Duplicate or invalid column names were fixed automatically.")
        rename_df = pd.DataFrame(
            st.session_state["rename_report"],
            columns=["Original Name", "New Name"]
        )
        st.dataframe(rename_df, use_container_width=True)

    st.subheader("Cleaning Actions")
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        if st.button("Remove Duplicates"):
            save_state()
            before = len(df)
            df = df.drop_duplicates(ignore_index=True)
            removed = before - len(df)
            st.session_state["df"] = df
            log_action("Remove Duplicates", f"Removed {removed} duplicate rows")
            st.success(f"Removed {removed} duplicate rows.")

    with col2:
        if st.button("Fill Missing Values"):
            save_state()
            df = fill_missing_values(df)
            st.session_state["df"] = df
            log_action("Fill Missing Values", "Filled missing values for dataset")
            st.success("Missing values filled.")

    with col3:
        if st.button("Standardize Column Names Again"):
            save_state()
            df, rename_report = standardize_and_uniquify_columns(df)
            st.session_state["df"] = df
            st.session_state["rename_report"] = rename_report
            log_action("Standardize Column Names", "Standardized and uniquified column names")
            st.success("Column names standardized and made unique.")

    with col4:
        if st.button("Trim Text Spaces"):
            save_state()
            df = trim_text_columns(df)
            st.session_state["df"] = df
            log_action("Trim Text Spaces", "Trimmed spaces in text columns")
            st.success("Text spaces cleaned.")

    with col5:
        if st.button("Convert Date Columns"):
            save_state()
            df = convert_date_columns(df)
            st.session_state["df"] = df
            log_action("Convert Date Columns", "Converted detected date/time columns")
            st.success("Date columns converted where possible.")

    df = st.session_state["df"]

    st.divider()
    st.subheader("Column Tools")

    columns = df.columns.tolist()
    selected_column = st.selectbox("Choose a column", columns)

    tool1, tool2, tool3 = st.columns(3)

    with tool1:
        if st.button("Fill Missing for Selected Column"):
            save_state()
            df = fill_column_missing(df, selected_column)
            st.session_state["df"] = df
            log_action("Fill Missing for Column", f"Filled missing values in '{selected_column}'")
            st.success(f"Missing values in '{selected_column}' filled.")

        if st.button("Drop Selected Column"):
            save_state()
            df = drop_selected_column(df, selected_column)
            st.session_state["df"] = df
            log_action("Drop Column", f"Dropped column '{selected_column}'")
            st.success(f"Column '{selected_column}' dropped. You can restore it using Undo Last Change.")
            st.rerun()

    with tool2:
        new_name = st.text_input("Rename selected column to")
        if st.button("Rename Column"):
            cleaned_new_name = new_name.strip().lower().replace(" ", "_")
            cleaned_new_name = "".join(ch for ch in cleaned_new_name if ch.isalnum() or ch == "_")

            if cleaned_new_name:
                if cleaned_new_name in df.columns and cleaned_new_name != selected_column:
                    st.warning("That column name already exists. Choose a different name.")
                else:
                    save_state()
                    df = rename_column(df, selected_column, cleaned_new_name)
                    df, rename_report = standardize_and_uniquify_columns(df)
                    st.session_state["df"] = df
                    st.session_state["rename_report"] = rename_report
                    log_action("Rename Column", f"Renamed '{selected_column}' to '{cleaned_new_name}'")
                    st.success(f"Column renamed to '{cleaned_new_name}'.")
                    st.rerun()
            else:
                st.warning("Please enter a valid new column name.")

    with tool3:
        st.write("Add New Column")
        new_col_name = st.text_input("New column name")
        default_value = st.text_input("Default value for new column")
        if st.button("Add Column"):
            cleaned_col_name = new_col_name.strip().lower().replace(" ", "_")
            cleaned_col_name = "".join(ch for ch in cleaned_col_name if ch.isalnum() or ch == "_")

            if not cleaned_col_name:
                st.warning("Please enter a valid column name.")
            elif cleaned_col_name in df.columns:
                st.warning("Column already exists.")
            else:
                save_state()
                df = add_new_column(df, cleaned_col_name, default_value)
                st.session_state["df"] = df
                log_action("Add Column", f"Added column '{cleaned_col_name}' with default '{default_value}'")
                st.success(f"Column '{cleaned_col_name}' added successfully.")
                st.rerun()

    df = st.session_state["df"]

    st.divider()
    st.subheader("Data Type Conversion")

    current_dtype = str(df[selected_column].dtype)
    st.write(f"Current data type of **{selected_column}**: `{current_dtype}`")

    target_type = st.selectbox(
        "Convert selected column to",
        ["string", "integer", "float", "datetime", "category"],
        key="dtype_conversion_select"
    )

    if st.button("Convert Column Type"):
        save_state()
        updated_df, message = convert_column_type(st.session_state["df"], selected_column, target_type)
        st.session_state["df"] = updated_df

        if "successfully" in message.lower():
            log_action("Convert Column Type", f"Converted '{selected_column}' to '{target_type}'")
            st.success(message)
            st.rerun()
        else:
            st.warning(message)

    df = st.session_state["df"]

    st.divider()
    st.subheader("Feature Engineering")

    feat_tab1, feat_tab2, feat_tab3, feat_tab4 = st.tabs(["Split Column", "Merge Columns", "Date Features", "Extract Numeric"])

    with feat_tab1:
        split_column_name = st.selectbox("Select column to split", df.columns.tolist(), key="split_col")
        split_delimiter = st.text_input("Delimiter", value=" ", key="split_delimiter")
        split_prefix = st.text_input("Prefix for new columns", value=f"{split_column_name}_part", key="split_prefix")

        if st.button("Split Column Now"):
            save_state()
            updated_df, message = split_column(
                st.session_state["df"],
                split_column_name,
                delimiter=split_delimiter,
                prefix=split_prefix
            )
            st.session_state["df"] = updated_df
            log_action("Split Column", f"Split '{split_column_name}' using delimiter '{split_delimiter}'")
            st.success(message)
            st.rerun()

    with feat_tab2:
        merge_col1 = st.selectbox("First column", df.columns.tolist(), key="merge_col1")
        merge_col2 = st.selectbox("Second column", df.columns.tolist(), key="merge_col2")
        merge_new_name = st.text_input("New merged column name", key="merge_new_name")
        merge_separator = st.text_input("Separator", value=" ", key="merge_separator")

        if st.button("Merge Columns Now"):
            if not merge_new_name.strip():
                st.warning("Please enter a new merged column name.")
            else:
                save_state()
                clean_merge_name = merge_new_name.strip().lower().replace(" ", "_")
                updated_df, message = merge_columns(
                    st.session_state["df"],
                    merge_col1,
                    merge_col2,
                    clean_merge_name,
                    merge_separator
                )
                st.session_state["df"] = updated_df
                log_action("Merge Columns", f"Merged '{merge_col1}' and '{merge_col2}' into '{clean_merge_name}'")
                st.success(message)
                st.rerun()

    with feat_tab3:
        date_column = st.selectbox("Select date column", df.columns.tolist(), key="date_feature_col")
        date_feature_type = st.selectbox(
            "Select date feature to create",
            ["year", "month", "day", "month_year"],
            key="date_feature_type"
        )

        if st.button("Create Date Feature"):
            save_state()
            updated_df, message = derive_date_feature(
                st.session_state["df"],
                date_column,
                date_feature_type
            )
            st.session_state["df"] = updated_df
            log_action("Create Date Feature", f"Created '{date_feature_type}' from '{date_column}'")
            st.success(message)
            st.rerun()

    with feat_tab4:
        st.write("Extract the numeric part from columns like `4.1/5`, `3.8/5`, etc.")
        extract_col = st.selectbox("Select column", df.columns.tolist(), key="extract_num_col")
        extract_delim = st.text_input("Delimiter (split on this, keep first part)", value="/", key="extract_delim")
        replace_original = st.checkbox("Replace original column (instead of creating a new _clean column)", key="extract_replace")

        st.caption(f"Preview of '{extract_col}': {df[extract_col].dropna().astype(str).head(5).tolist()}")

        if st.button("Extract Numeric Now"):
            save_state()
            updated_df, message = extract_numeric_from_column(
                st.session_state["df"],
                extract_col,
                delimiter=extract_delim,
                replace_original=replace_original
            )
            st.session_state["df"] = updated_df
            log_action("Extract Numeric", message)
            st.success(message)
            st.rerun()

    df = st.session_state["df"]

    st.divider()
    st.subheader("Data Quality Summary")
    summary = data_quality_summary(df)
    summary_df = pd.DataFrame(summary.items(), columns=["Metric", "Value"])
    st.dataframe(summary_df, use_container_width=True)

    st.divider()
    st.subheader("Visual Profiling Dashboard")

    viz1, viz2 = st.columns(2)

    with viz1:
        st.write("Missing Values")
        fig_missing = plot_missing_values(df)
        if fig_missing is not None:
            st.pyplot(fig_missing, clear_figure=True)
        else:
            st.info("No missing values found.")

        st.write("Correlation Heatmap")
        fig_corr = plot_correlation_heatmap(df)
        if fig_corr is not None:
            st.pyplot(fig_corr, clear_figure=True)
        else:
            st.info("Need at least 2 numeric columns for correlation heatmap.")

    with viz2:
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        if numeric_cols:
            hist_col = st.selectbox("Histogram / Box Plot column", numeric_cols, key="hist_col")
            fig_hist = plot_histogram(df, hist_col)
            if fig_hist is not None:
                st.pyplot(fig_hist, clear_figure=True)
            else:
                st.info("Histogram could not be created for this column.")

            fig_box = plot_boxplot(df, hist_col)
            if fig_box is not None:
                st.pyplot(fig_box, clear_figure=True)
            else:
                st.info("Box plot could not be created for this column.")
        else:
            st.info("No numeric columns available for histogram or box plot.")

        categorical_cols = df.select_dtypes(include=["object", "string", "category"]).columns.tolist()
        if categorical_cols:
            cat_col = st.selectbox("Top categories column", categorical_cols, key="top_cat_col")
            top_df = top_categories_table(df, cat_col)

            if not top_df.empty and {"category", "count"}.issubset(top_df.columns):
                st.dataframe(top_df, use_container_width=True)
                st.bar_chart(top_df.set_index("category")["count"])
            else:
                st.info("No category data available for this column.")
        else:
            st.info("No categorical columns available.")

    st.divider()
    st.subheader("AI Suggestions")

    @st.cache_data(show_spinner=False)
    def get_cached_suggestions(shape_key: str, col_key: str):
        # Re-read from session state inside the cached function using a stable key
        return generate_ai_suggestions(st.session_state["df"])

    suggestions_df = get_cached_suggestions(
        str(df.shape),
        ",".join(df.columns.tolist())
    )

    top_action1, top_action2 = st.columns(2)

    with top_action1:
        if not suggestions_df.empty and st.button("Apply All Safe Suggestions"):
            save_state()
            st.session_state["df"] = apply_all_safe_suggestions(st.session_state["df"], suggestions_df)
            log_action("Apply All Safe Suggestions", "Applied all safe AI-generated suggestions")
            st.success("Applied all safe AI suggestions.")
            st.rerun()

    with top_action2:
        st.caption("Bulk apply reduces repeated clicks and reruns.")

    if not suggestions_df.empty:
        for i, row in suggestions_df.iterrows():
            with st.container():
                c1, c2, c3, c4, c5, c6 = st.columns([2, 2, 4, 4, 2, 2])

                with c1:
                    st.write(f"**{row['type']}**")
                with c2:
                    st.write(str(row["column"]))
                with c3:
                    st.write(row["issue"])
                with c4:
                    st.write(row["recommendation"])
                with c5:
                    st.write(row["confidence"])
                with c6:
                    if st.button("Apply", key=f"apply_suggestion_{i}"):
                        save_state()
                        updated_df = apply_suggestion(
                            st.session_state["df"],
                            row["type"],
                            row["column"] if row["column"] != "All Rows" else None
                        )
                        st.session_state["df"] = updated_df
                        log_action("Apply AI Suggestion", f"Applied '{row['type']}' on '{row['column']}'")
                        st.success(f"Applied fix for: {row['type']} - {row['column']}")
                        st.rerun()

            st.markdown("---")
    else:
        st.success("No major data quality issues detected.")

    df = st.session_state["df"]

    st.divider()
    st.subheader("Data Preview")
    preview_rows = st.slider("Rows to preview", min_value=5, max_value=30, value=10, step=5, key="preview_rows")
    st.dataframe(df.head(preview_rows), use_container_width=True)

    info1, info2 = st.columns(2)

    with info1:
        st.subheader("Dataset Shape")
        st.write(f"Rows: {df.shape[0]}")
        st.write(f"Columns: {df.shape[1]}")
        st.subheader("Column Names")
        st.write(list(df.columns))

    with info2:
        st.subheader("Data Types")
        dtype_df = df.dtypes.astype(str).reset_index()
        dtype_df.columns = ["Column", "Data Type"]
        st.dataframe(dtype_df, use_container_width=True)

    st.subheader("Missing Values")
    missing_df = df.isnull().sum().reset_index()
    missing_df.columns = ["Column", "Missing Values"]
    st.dataframe(missing_df, use_container_width=True)

    st.divider()
    st.subheader("Row Filtering and Search")

    filter_tab1, filter_tab2, filter_tab3 = st.tabs(
        ["Category Filter", "Numeric Range Filter", "Text Search"]
    )

    filtered_df = df.copy()

    category_column = None
    category_values = []
    numeric_filter_column = None
    min_val = None
    max_val = None
    text_filter_column = None
    search_text = ""

    with filter_tab1:
        categorical_cols = df.select_dtypes(include=["object", "string", "category"]).columns.tolist()
        if categorical_cols:
            category_column = st.selectbox(
                "Select categorical column",
                categorical_cols,
                key="filter_category_column"
            )
            category_options = df[category_column].dropna().astype(str).unique().tolist()
            category_values = st.multiselect(
                "Select values to keep",
                sorted(category_options),
                key="filter_category_values"
            )
        else:
            st.info("No categorical columns available for category filtering.")

    with filter_tab2:
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        if numeric_cols:
            numeric_filter_column = st.selectbox(
                "Select numeric column",
                numeric_cols,
                key="filter_numeric_column"
            )

            numeric_series = df[numeric_filter_column].dropna()
            if not numeric_series.empty:
                min_data = float(numeric_series.min())
                max_data = float(numeric_series.max())

                min_val, max_val = st.slider(
                    "Select numeric range",
                    min_value=min_data,
                    max_value=max_data,
                    value=(min_data, max_data),
                    key="filter_numeric_range"
                )
            else:
                st.info("Selected numeric column has no valid values.")
        else:
            st.info("No numeric columns available for numeric filtering.")

    with filter_tab3:
        text_cols = df.select_dtypes(include=["object", "string", "category"]).columns.tolist()
        if text_cols:
            text_filter_column = st.selectbox(
                "Select text column",
                text_cols,
                key="filter_text_column"
            )
            search_text = st.text_input(
                "Search text",
                key="filter_search_text"
            )
        else:
            st.info("No text columns available for search.")

    filtered_df = apply_all_filters(
        df,
        category_column=category_column,
        category_values=category_values,
        numeric_column=numeric_filter_column,
        min_val=min_val,
        max_val=max_val,
        text_column=text_filter_column,
        search_text=search_text,
    )

    st.write(f"Filtered rows: {filtered_df.shape[0]} / {df.shape[0]}")
    st.dataframe(filtered_df.head(100), use_container_width=True)

    filter_action1, filter_action2 = st.columns(2)

    with filter_action1:
        if st.button("Replace Main Data with Filtered Data"):
            save_state()
            st.session_state["df"] = filtered_df.copy()
            log_action("Apply Row Filters", f"Replaced main dataset with {filtered_df.shape[0]} filtered rows")
            st.success("Main dataset replaced with filtered data.")
            st.rerun()

    with filter_action2:
        filtered_csv = filtered_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Filtered CSV",
            data=filtered_csv,
            file_name="filtered_data.csv",
            mime="text/csv"
        )

    st.divider()
    st.subheader("Column Insights")

    if selected_column in df.columns:
        unique_count = df[selected_column].nunique(dropna=True)
        st.write(f"Unique values in **{selected_column}**: {unique_count}")

        if not pd.api.types.is_numeric_dtype(df[selected_column]):
            value_counts = df[selected_column].astype(str).value_counts(dropna=False).head(10)
            chart_df = value_counts.rename_axis("value").reset_index(name="count")
            st.dataframe(chart_df, use_container_width=True)
            st.bar_chart(chart_df.set_index("value")["count"])
        else:
            st.dataframe(df[selected_column].describe().to_frame(name="Value"), use_container_width=True)
            st.line_chart(df[selected_column].dropna())

            outliers = detect_outliers_iqr(df, selected_column)
            st.subheader("Outliers")
            st.write(f"Outlier rows found in {selected_column}: {len(outliers)}")
            if not outliers.empty:
                st.dataframe(outliers, use_container_width=True)

    st.divider()
    st.subheader("Top N Lookup")
    st.caption("Find rows with the highest or lowest value in any numeric column — e.g. restaurant with most votes.")

    numeric_cols_lookup = df.select_dtypes(include=np.number).columns.tolist()
    all_cols = df.columns.tolist()

    if numeric_cols_lookup:
        lu_col1, lu_col2, lu_col3, lu_col4 = st.columns(4)

        with lu_col1:
            lookup_value_col = st.selectbox("Rank by (numeric)", numeric_cols_lookup, key="lookup_value_col")
        with lu_col2:
            lookup_label_col = st.selectbox("Show column", all_cols, key="lookup_label_col")
        with lu_col3:
            lookup_order = st.selectbox("Order", ["Highest first", "Lowest first"], key="lookup_order")
        with lu_col4:
            lookup_n = st.number_input("Top N rows", min_value=1, max_value=100, value=5, step=1, key="lookup_n")

        ascending = lookup_order == "Lowest first"
        lookup_result = (
            df[[lookup_label_col, lookup_value_col]]
            .dropna(subset=[lookup_value_col])
            .sort_values(by=lookup_value_col, ascending=ascending)
            .head(int(lookup_n))
            .reset_index(drop=True)
        )
        st.dataframe(lookup_result, use_container_width=True)
    else:
        st.info("Need at least 1 numeric column for Top N lookup.")

    st.divider()
    st.subheader("Charts")

    numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
    categorical_columns = df.select_dtypes(include=["object", "string", "category"]).columns.tolist()

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        if numeric_columns:
            num_chart_col = st.selectbox("Select numeric column for chart", numeric_columns)
            st.line_chart(df[num_chart_col].dropna())
        else:
            st.info("No numeric columns available for numeric chart.")

    with chart_col2:
        if categorical_columns:
            cat_chart_col = st.selectbox("Select categorical column for chart", categorical_columns)
            top_categories = df[cat_chart_col].astype(str).value_counts().head(10)
            top_df = top_categories.reset_index()
            top_df.columns = ["Category", "Count"]
            st.bar_chart(top_df.set_index("Category"))
        else:
            st.info("No categorical columns available for category chart.")

    st.subheader("Basic Statistics")
    st.dataframe(df.describe(include="all").fillna(""), use_container_width=True)

    st.divider()
    st.subheader("Smart Insights")

    insight_tab1, insight_tab2, insight_tab3, insight_tab4 = st.tabs(
        ["Top Correlations", "Suspicious Columns", "Chart Recommendation", "Suggested Targets"]
    )

    with insight_tab1:
        top_corr_df = get_top_correlated_pairs(df, top_n=10, min_abs_corr=0.5)
        if not top_corr_df.empty:
            st.dataframe(top_corr_df, use_container_width=True)
        else:
            st.info("No strong numeric correlations found.")

    with insight_tab2:
        suspicious_df = detect_suspicious_columns(df, missing_threshold=0.5)
        st.session_state["latest_suspicious_df"] = suspicious_df
        if not suspicious_df.empty:
            st.dataframe(suspicious_df, use_container_width=True)
        else:
            st.success("No suspicious columns detected based on current rules.")

    with insight_tab3:
        recommend_col = st.selectbox(
            "Select column for chart recommendation",
            df.columns.tolist(),
            key="recommend_chart_col"
        )
        chart_reco = recommend_chart_for_column(df, recommend_col)
        st.write(f"Recommended chart type for **{recommend_col}**: {chart_reco}")

    with insight_tab4:
        target_df = suggest_target_columns(df)
        st.session_state["latest_target_df"] = target_df
        if not target_df.empty:
            st.dataframe(target_df, use_container_width=True)
        else:
            st.info("No strong target column suggestions available.")

    st.divider()
    st.subheader("ML Starter Module")

    if len(df.columns) >= 2:
        ml_col1, ml_col2, ml_col3 = st.columns(3)

        with ml_col1:
            target_column = st.selectbox(
                "Select target column",
                df.columns.tolist(),
                key="ml_target_column"
            )

        detected_problem = detect_problem_type(df, target_column)

        with ml_col2:
            st.write(f"Detected problem type: **{detected_problem}**")

        with ml_col3:
            model_choice = st.selectbox(
                "Choose baseline model",
                ["auto", "random_forest"],
                key="ml_model_choice"
            )

        test_size = st.slider(
            "Test size",
            min_value=0.1,
            max_value=0.4,
            value=0.2,
            step=0.05,
            key="ml_test_size"
        )

        if st.button("Train Baseline Model"):
            result = train_baseline_model(
                df,
                target_col=target_column,
                model_name=model_choice,
                test_size=test_size,
                random_state=42,
            )

            if "error" in result:
                st.error(result["error"])
            else:
                st.session_state["trained_pipeline"] = result["pipeline"]
                st.session_state["trained_feature_columns"] = result["feature_columns"]
                st.session_state["trained_target_column"] = result["target_column"]
                st.session_state["latest_model_metrics"] = result["metrics"]

                log_action(
                    "Train Baseline Model",
                    f"Trained {result['model_used']} for target '{target_column}'"
                )
                st.success(f"Model trained successfully using {result['model_used']}")

                metrics_df = pd.DataFrame(
                    result["metrics"].items(),
                    columns=["Metric", "Value"]
                )
                st.dataframe(metrics_df, use_container_width=True)

                st.subheader("Prediction Preview")
                st.dataframe(result["preview"], use_container_width=True)

        if st.session_state["trained_pipeline"] is not None:
            st.divider()
            st.subheader("Model Export and New Predictions")

            model_bytes = save_pipeline_to_bytes(st.session_state["trained_pipeline"])
            st.download_button(
                label="Download Trained Model (.joblib)",
                data=model_bytes,
                file_name="trained_pipeline.joblib",
                mime="application/octet-stream"
            )

            prediction_file = st.file_uploader(
                "Upload new CSV file for prediction",
                type=["csv"],
                key="prediction_file_uploader"
            )

            if prediction_file is not None:
                try:
                    new_pred_df = pd.read_csv(prediction_file)
                    st.write("New Data Preview")
                    st.dataframe(new_pred_df.head(10), use_container_width=True)

                    if st.button("Run Predictions on New Data"):
                        prediction_result_df, message = predict_with_pipeline(
                            st.session_state["trained_pipeline"],
                            new_pred_df,
                            st.session_state["trained_feature_columns"]
                        )

                        if prediction_result_df is None:
                            st.error(message)
                        else:
                            log_action(
                                "Predict New Data",
                                f"Generated predictions for {len(prediction_result_df)} rows"
                            )
                            st.success(message)
                            st.dataframe(prediction_result_df.head(20), use_container_width=True)

                            pred_csv = prediction_result_df.to_csv(index=False).encode("utf-8")
                            st.download_button(
                                label="Download Predictions CSV",
                                data=pred_csv,
                                file_name="predictions.csv",
                                mime="text/csv"
                            )
                except Exception as e:
                    st.error(f"Error reading prediction file: {e}")
    else:
        st.info("Need at least 2 columns to train a model.")

    st.divider()
    st.subheader("Automatic Model Comparison")

    if len(df.columns) >= 2:
        compare_target_col = st.selectbox(
            "Select target column for model comparison",
            df.columns.tolist(),
            key="compare_target_col"
        )

        compare_problem = detect_problem_type(df, compare_target_col)
        st.write(f"Detected problem type: **{compare_problem}**")

        compare_test_size = st.slider(
            "Comparison test size",
            min_value=0.1,
            max_value=0.4,
            value=0.2,
            step=0.05,
            key="compare_test_size"
        )

        if st.button("Compare Models Automatically"):
            compare_result = compare_models(
                df,
                target_col=compare_target_col,
                test_size=compare_test_size,
                random_state=42
            )

            if "error" in compare_result:
                st.error(compare_result["error"])
            else:
                best_pipeline = compare_result["best_pipeline"]
                best_model_name = compare_result["best_model_name"]

                st.session_state["trained_pipeline"] = best_pipeline
                st.session_state["trained_feature_columns"] = compare_result["feature_columns"]
                st.session_state["trained_target_column"] = compare_result["target_column"]
                st.session_state["latest_model_comparison_df"] = compare_result["results_df"]

                log_action(
                    "Automatic Model Comparison",
                    f"Best model selected: {best_model_name} for target '{compare_target_col}'"
                )

                st.success(f"Best model selected: {best_model_name}")
                st.dataframe(compare_result["results_df"], use_container_width=True)

                st.subheader("Best Model Prediction Preview")
                st.dataframe(compare_result["preview"], use_container_width=True)

                importance_df = extract_feature_importance(best_pipeline, top_n=15)
                st.session_state["latest_feature_importance_df"] = importance_df

                if not importance_df.empty:
                    st.subheader("Feature Importance")
                    st.dataframe(importance_df, use_container_width=True)
                    st.bar_chart(importance_df.set_index("feature")["importance"])
                else:
                    st.info("Feature importance is not available for the selected best model.")
    else:
        st.info("Need at least 2 columns to compare models.")

    st.divider()
    st.subheader("Advanced EDA")

    eda_tab1, eda_tab2, eda_tab3 = st.tabs(
        ["Scatter Plot", "Grouped Summary", "Date Trend"]
    )

    with eda_tab1:
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        categorical_cols = df.select_dtypes(include=["object", "string", "category"]).columns.tolist()

        if len(numeric_cols) >= 2:
            x_axis = st.selectbox("X-axis", numeric_cols, key="scatter_x")
            y_axis = st.selectbox("Y-axis", numeric_cols, key="scatter_y")
            color_axis = st.selectbox(
                "Color by (optional)",
                ["None"] + categorical_cols,
                key="scatter_color"
            )

            scatter_fig = create_scatter_plot(
                df,
                x_axis,
                y_axis,
                None if color_axis == "None" else color_axis
            )

            if scatter_fig is not None:
                st.plotly_chart(scatter_fig, use_container_width=True)
        else:
            st.info("Need at least 2 numeric columns for scatter plot.")

    with eda_tab2:
        categorical_cols = df.select_dtypes(include=["object", "string", "category"]).columns.tolist()
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

        if categorical_cols and numeric_cols:
            grp_c1, grp_c2, grp_c3, grp_c4, grp_c5 = st.columns(5)

            with grp_c1:
                group_col = st.selectbox("Group by column", categorical_cols, key="group_col")
            with grp_c2:
                value_col = st.selectbox("Value column", numeric_cols, key="group_value_col")
            with grp_c3:
                agg_func = st.selectbox("Aggregation", ["mean", "sum", "min", "max", "median"], key="group_agg")
            with grp_c4:
                sort_order = st.selectbox("Sort by", ["Highest first", "Lowest first", "None"], key="group_sort")
            with grp_c5:
                top_n_group = st.number_input("Show top N", min_value=1, max_value=500, value=20, step=5, key="group_top_n")

            grouped_df = create_grouped_summary(df, group_col, value_col, agg_func)

            if not grouped_df.empty:
                y_name = grouped_df.columns[1]

                if sort_order == "Highest first":
                    grouped_df = grouped_df.sort_values(by=y_name, ascending=False)
                elif sort_order == "Lowest first":
                    grouped_df = grouped_df.sort_values(by=y_name, ascending=True)

                grouped_df = grouped_df.head(int(top_n_group)).reset_index(drop=True)

                st.dataframe(grouped_df, use_container_width=True)

                grouped_fig = create_grouped_bar_chart(grouped_df, group_col, y_name)
                if grouped_fig is not None:
                    st.plotly_chart(grouped_fig, use_container_width=True)
        else:
            st.info("Need at least 1 categorical and 1 numeric column for grouped summary.")

    with eda_tab3:
        date_cols = df.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"]).columns.tolist()
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

        possible_date_cols = date_cols if date_cols else df.columns.tolist()

        if numeric_cols:
            trend_date_col = st.selectbox("Date column", possible_date_cols, key="trend_date_col")
            trend_value_col = st.selectbox("Value column", numeric_cols, key="trend_value_col")
            trend_freq = st.selectbox(
                "Frequency",
                ["D", "W", "ME", "QE", "YE"],
                format_func=lambda x: {"D": "Daily", "W": "Weekly", "ME": "Monthly", "QE": "Quarterly", "YE": "Yearly"}.get(x, x),
                key="trend_freq"
            )

            trend_df, trend_fig = create_date_trend(df, trend_date_col, trend_value_col, trend_freq)

            if not trend_df.empty and trend_fig is not None:
                st.dataframe(trend_df, use_container_width=True)
                st.plotly_chart(trend_fig, use_container_width=True)
            else:
                st.info("Could not generate trend chart. Check date/value columns.")
        else:
            st.info("Need at least 1 numeric column for trend analysis.")

    st.divider()
    st.subheader("Export")

    csv_data = dataframe_to_csv_bytes(df)
    excel_data = dataframe_to_excel_bytes(df)
    history_data = history_to_csv_bytes(st.session_state["action_log"])

    full_report_data = create_full_html_report(
        df=df,
        summary_dict=summary,
        action_log=st.session_state["action_log"],
        suggestions_df=suggestions_df,
        suspicious_df=st.session_state.get("latest_suspicious_df"),
        target_df=st.session_state.get("latest_target_df"),
        model_metrics=st.session_state.get("latest_model_metrics"),
        model_comparison_df=st.session_state.get("latest_model_comparison_df"),
        feature_importance_df=st.session_state.get("latest_feature_importance_df"),
        report_title="AI Data Analysis Full Report"
    )

    exp1, exp2, exp3, exp4 = st.columns(4)

    with exp1:
        st.download_button(
            label="Download CSV",
            data=csv_data,
            file_name="cleaned_data.csv",
            mime="text/csv"
        )

    with exp2:
        st.download_button(
            label="Download Excel",
            data=excel_data,
            file_name="cleaned_data.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    with exp3:
        st.download_button(
            label="Download Full HTML Report",
            data=full_report_data,
            file_name="full_data_report.html",
            mime="text/html"
        )

    with exp4:
        st.download_button(
            label="Download Action History",
            data=history_data,
            file_name="action_history.csv",
            mime="text/csv"
        )

else:
    st.info("Please upload a CSV or Excel file to continue.")