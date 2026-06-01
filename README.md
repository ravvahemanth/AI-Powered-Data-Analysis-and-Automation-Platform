# AI-Powered Data Analysis and Automation Platform

A full-featured, browser-based data analysis platform built with **Streamlit**, **Pandas**, **Scikit-learn**, and **Plotly**. Upload a CSV or Excel file and get instant data cleaning, visual profiling, AI-driven suggestions, machine learning workflows, and exportable reports — all without writing a single line of code.

---

## Table of Contents

- [Features](#features)
- [Tech Stack](#tech-stack)
- [Getting Started](#getting-started)
- [Project Structure](#project-structure)
- [Detailed Operations](#detailed-operations)
  - [1. File Upload and Loading](#1-file-upload-and-loading)
  - [2. Data Cleaning](#2-data-cleaning)
  - [3. Column Tools](#3-column-tools)
  - [4. Data Type Conversion](#4-data-type-conversion)
  - [5. Feature Engineering](#5-feature-engineering)
  - [6. Data Quality Summary](#6-data-quality-summary)
  - [7. Visual Profiling Dashboard](#7-visual-profiling-dashboard)
  - [8. AI Suggestions](#8-ai-suggestions)
  - [9. Data Preview and Statistics](#9-data-preview-and-statistics)
  - [10. Row Filtering and Search](#10-row-filtering-and-search)
  - [11. Column Insights and Outlier Detection](#11-column-insights-and-outlier-detection)
  - [12. Top N Lookup](#12-top-n-lookup)
  - [13. Charts](#13-charts)
  - [14. Smart Insights](#14-smart-insights)
  - [15. ML Starter Module](#15-ml-starter-module)
  - [16. Automatic Model Comparison](#16-automatic-model-comparison)
  - [17. Advanced EDA](#17-advanced-eda)
  - [18. Export](#18-export)
  - [19. Undo / Reset](#19-undo--reset)
- [Screenshots](#screenshots)
- [License](#license)

---

## Features

- Upload CSV, XLSX, or XLS files with automatic encoding detection
- One-click data cleaning: remove duplicates, fill missing values, trim whitespace, convert dates
- Per-column tools: rename, drop, add, fill missing, convert type
- Feature engineering: split columns, merge columns, extract date parts, extract numeric values
- AI-generated data quality suggestions with one-click apply
- Visual profiling: missing value bar chart, correlation heatmap, histograms, box plots
- Smart insights: top correlated pairs, suspicious columns, chart recommendations, target column suggestions
- Row filtering by category, numeric range, or text search
- Top N lookup: find rows with highest or lowest values in any numeric column
- ML Starter: auto-detect problem type, train a baseline model, preview predictions
- Automatic model comparison across multiple algorithms with feature importance
- Advanced EDA: scatter plots, grouped summaries, date trend charts
- Full HTML report export, CSV/Excel export, action history export, trained model download
- Undo last change and reset to original file at any point

---

## Tech Stack

| Library | Purpose |
|---|---|
| Streamlit | Web UI framework |
| Pandas | Data manipulation |
| NumPy | Numerical operations |
| Scikit-learn | Machine learning pipelines |
| Plotly | Interactive charts |
| Matplotlib / Seaborn | Static charts |
| openpyxl / xlrd | Excel file reading |
| XlsxWriter | Excel file writing |
| Joblib | Model serialization |

---

## Getting Started

### Prerequisites

- Python 3.9 or higher
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/ravvahemanth/AI-Powered-Data-Analysis-and-Automation-Platform.git
cd AI-Powered-Data-Analysis-and-Automation-Platform

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

The app will open at `http://localhost:8501` in your browser.

---

## Project Structure

```
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── utils/
│   ├── advanced_eda.py         # Scatter plots, grouped summaries, date trends
│   ├── ai_suggestions.py       # Rule-based AI data quality suggestions
│   ├── analysis.py             # Outlier detection (IQR), data quality summary
│   ├── cleaners.py             # Cleaning functions (trim, fill, convert dates, etc.)
│   ├── data_loader.py          # File loading with encoding fallback
│   ├── exporter.py             # CSV, Excel, HTML report, action history export
│   ├── feature_engineering.py  # Split, merge, date features, extract numeric
│   ├── filters.py              # Category, numeric range, text search filters
│   ├── helpers.py              # Column standardization and uniquification
│   ├── ml_module.py            # ML training, comparison, feature importance, prediction
│   ├── smart_insights.py       # Correlations, suspicious columns, chart recommendations
│   ├── state_manager.py        # Session state, undo/reset, action logging
│   ├── suggestion_actions.py   # Apply individual or bulk AI suggestions
│   ├── type_converter.py       # Column type conversion
│   └── visuals.py              # Missing value chart, heatmap, histogram, box plot
```

---

## Detailed Operations

### 1. File Upload and Loading

- Accepts **CSV**, **XLSX**, and **XLS** files via a drag-and-drop uploader.
- CSV files are read with UTF-8 encoding, falling back to Latin-1 for files with special characters.
- Excel files use `openpyxl` (XLSX) or `xlrd` (XLS) engines.
- Column names are automatically **standardized** on load: lowercased, spaces replaced with underscores, special characters removed.
- **Duplicate or invalid column names** are made unique automatically (e.g., `name` and `name` become `name` and `name_2`), and a rename report is shown to the user.
- Uploading a new file resets all state, history, and action logs.

---

### 2. Data Cleaning

Five one-click cleaning actions available at the top of the page:

| Action | What it does |
|---|---|
| Remove Duplicates | Drops all duplicate rows, resets the index |
| Fill Missing Values | Fills numeric columns with the median, text columns with the mode, datetime columns with the median timestamp |
| Standardize Column Names Again | Re-runs column name standardization and uniquification |
| Trim Text Spaces | Strips leading and trailing whitespace from all string/object columns |
| Convert Date Columns | Detects columns whose names contain "date" or "time" and converts them to `datetime` |

All cleaning actions save the previous state to the undo history before applying.

---

### 3. Column Tools

A column selector lets you pick any column and apply targeted operations:

- **Fill Missing for Selected Column** — fills missing values in just the chosen column (median for numeric, mode for text).
- **Drop Selected Column** — removes the column from the dataset. Reversible via Undo.
- **Rename Column** — renames the selected column to a user-supplied name. The new name is cleaned (lowercased, underscores, alphanumeric only) and checked for conflicts.
- **Add New Column** — creates a new column with a user-defined name and a default fill value.

---

### 4. Data Type Conversion

Displays the current data type of the selected column and allows conversion to:

| Target Type | Behavior |
|---|---|
| `string` | Casts to pandas `StringDtype` |
| `integer` | Converts via `pd.to_numeric`, stores as nullable `Int64` |
| `float` | Converts via `pd.to_numeric`, stores as float64 |
| `datetime` | Parses with `pd.to_datetime`, coerces unparseable values to `NaT` |
| `category` | Casts to pandas `CategoricalDtype` |

Conversion errors are reported without crashing the app.

---

### 5. Feature Engineering

Four tabs for creating new columns from existing data:

#### Split Column
Splits a text column into multiple new columns using a configurable delimiter. Empty trailing splits are automatically dropped. A custom prefix can be set for the new column names.

#### Merge Columns
Concatenates two columns into a new column with a configurable separator. Both columns are cast to string before merging.

#### Date Features
Extracts a date component from a datetime column and stores it as a new column:
- `year` — integer year
- `month` — integer month (1–12)
- `day` — integer day of month
- `month_year` — period string like `2024-03`

#### Extract Numeric
Splits a column by a delimiter (default `/`) and keeps only the first part, converting it to float. Useful for columns like `4.1/5` or `3.8/5`. Can replace the original column or create a new `_clean` column.

---

### 6. Data Quality Summary

A summary table showing:

| Metric | Description |
|---|---|
| Rows | Total row count |
| Columns | Total column count |
| Duplicate Rows | Number of fully duplicate rows |
| Missing Cells | Total count of null/NaN cells |
| Missing % | Percentage of all cells that are missing |
| Numeric Columns | Count of numeric columns |
| Text Columns | Count of object/string columns |

---

### 7. Visual Profiling Dashboard

Rendered in a two-column layout:

**Left column:**
- **Missing Values Bar Chart** — horizontal bar chart showing missing count per column. Hidden if no missing values exist.
- **Correlation Heatmap** — annotated heatmap of Pearson correlations between all numeric columns. Requires at least 2 numeric columns.

**Right column:**
- **Histogram** — frequency distribution for a selected numeric column (20 bins).
- **Box Plot** — quartile and outlier visualization for a selected numeric column.
- **Top Categories Table + Bar Chart** — value counts for a selected categorical column, top 10 values.

---

### 8. AI Suggestions

A rule-based suggestion engine scans the dataset and flags:

| Suggestion Type | Trigger |
|---|---|
| Duplicate Rows | Any duplicate rows found |
| Missing Values | Any column with missing values (High confidence if >20%) |
| Text Formatting | Values with leading/trailing whitespace |
| Inconsistent Text Case | Columns where lowercasing reduces unique value count |
| Possible Date Column | Columns where >70% of values parse as dates |

Each suggestion shows the type, column, issue description, recommendation, and confidence level. Individual suggestions can be applied with a single click, or all High/Medium confidence suggestions can be applied in bulk with **Apply All Safe Suggestions**.

---

### 9. Data Preview and Statistics

- **Data Preview** — configurable slider (5–30 rows) to preview the current state of the dataset.
- **Dataset Shape** — row and column counts.
- **Column Names** — full list of current column names.
- **Data Types** — table of column names and their current dtypes.
- **Missing Values** — per-column missing value counts.
- **Basic Statistics** — `df.describe(include="all")` covering count, mean, std, min, max, and frequency for all columns.

---

### 10. Row Filtering and Search

Three filter tabs that can be combined:

| Filter Type | How it works |
|---|---|
| Category Filter | Multi-select from unique values in a categorical column |
| Numeric Range Filter | Slider to set min/max bounds on a numeric column |
| Text Search | Case-insensitive substring search on a text column |

All active filters are applied simultaneously. The filtered row count is shown, and up to 100 rows are previewed. Two actions are available:
- **Replace Main Data with Filtered Data** — permanently replaces the working dataset with the filtered subset (undoable).
- **Download Filtered CSV** — downloads the filtered data directly without modifying the main dataset.

---

### 11. Column Insights and Outlier Detection

For the currently selected column:

- **Categorical columns** — shows unique value count, top 10 value counts as a table, and a bar chart.
- **Numeric columns** — shows descriptive statistics, a line chart of values, and an **outlier detection** table using the IQR method (values below Q1 − 1.5×IQR or above Q3 + 1.5×IQR).

---

### 12. Top N Lookup

Quickly find the rows with the highest or lowest value in any numeric column:

- Choose a numeric column to rank by.
- Choose any column to display alongside the rank.
- Select ascending or descending order.
- Set N (1–100 rows).

Useful for questions like "which restaurant has the most votes?" or "which product has the lowest price?".

---

### 13. Charts

Two side-by-side charts:

- **Numeric Line Chart** — line chart of a selected numeric column's values.
- **Categorical Bar Chart** — bar chart of the top 10 value counts for a selected categorical column.

---

### 14. Smart Insights

Four insight tabs:

#### Top Correlations
Lists the top 10 numeric column pairs with absolute Pearson correlation ≥ 0.5, sorted by strength.

#### Suspicious Columns
Flags columns that may need attention:
- **Constant columns** — only one unique value.
- **High missing values** — more than 50% of values are null.
- **Binary/low-cardinality columns** — exactly 2 unique values (may be a label or flag).

#### Chart Recommendation
For any selected column, recommends the most appropriate chart type based on its data type and cardinality.

#### Suggested Targets
Recommends columns that are good candidates for use as a machine learning target:
- Continuous numeric columns → regression
- Low-cardinality numeric columns → classification
- Categorical columns with 2–20 unique values → classification

---

### 15. ML Starter Module

A guided machine learning workflow:

1. **Select a target column** — the column you want to predict.
2. **Auto-detect problem type** — classification (≤10 unique numeric values or categorical) or regression (continuous numeric).
3. **Choose a baseline model** — `auto` (Logistic Regression / Linear Regression) or `random_forest`.
4. **Set test size** — slider from 10% to 40% (default 20%).
5. **Train** — builds a full Scikit-learn pipeline with:
   - Median imputation + StandardScaler for numeric features
   - Most-frequent imputation + OneHotEncoder for categorical features
   - The selected model as the final estimator
6. **View metrics** — accuracy + weighted F1 for classification; MAE, RMSE, R² for regression.
7. **Prediction preview** — shows actual vs. predicted values for the test set.
8. **Download trained model** — exports the full pipeline as a `.joblib` file.
9. **Predict on new data** — upload a new CSV and run the trained pipeline to generate predictions, which can be downloaded as CSV.

---

### 16. Automatic Model Comparison

Trains and evaluates multiple models on the same train/test split and selects the best one:

**Classification candidates:**
- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier

**Regression candidates:**
- Linear Regression
- Ridge Regression
- Decision Tree Regressor
- Random Forest Regressor

Results are shown in a comparison table. The best model is selected by accuracy + F1 (classification) or R² + RMSE (regression). The best pipeline is saved to session state for export and prediction. **Feature importance** is extracted and displayed as a table and bar chart for tree-based models, or from coefficients for linear models.

---

### 17. Advanced EDA

Three interactive Plotly-powered tabs:

#### Scatter Plot
Plot any two numeric columns against each other. Optionally color points by a categorical column to reveal group-level patterns.

#### Grouped Summary
Aggregate a numeric column by a categorical grouping column using mean, sum, min, max, or median. Results are sortable and limited to a configurable top N. Displayed as a table and an interactive bar chart.

#### Date Trend
Resample a numeric column over time at Daily, Weekly, Monthly, Quarterly, or Yearly frequency. Displays a trend table and an interactive line chart with markers.

---

### 18. Export

Four download buttons at the bottom of the page:

| Export | Format | Contents |
|---|---|---|
| Download CSV | `.csv` | Current cleaned dataset |
| Download Excel | `.xlsx` | Current cleaned dataset in a formatted sheet |
| Download Full HTML Report | `.html` | Dataset preview, quality summary, action history, AI suggestions, suspicious columns, target suggestions, model metrics, model comparison, feature importance |
| Download Action History | `.csv` | Log of every action taken in the session with details |

---

### 19. Undo / Reset

- **Undo Last Change** — reverts the dataset to its state before the most recent action. Supports multiple levels of undo (one per action taken).
- **Reset to Original File** — restores the dataset to exactly what was loaded from the uploaded file, clearing all history and action logs.

---

## Screenshots

> Coming soon — run the app locally and upload any CSV to explore all features.

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
