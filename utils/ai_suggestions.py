import pandas as pd


def generate_ai_suggestions(df):
    suggestions = []
    total_rows = len(df)

    duplicate_rows = int(df.duplicated().sum())
    if duplicate_rows > 0:
        suggestions.append({
            "type": "Duplicate Rows",
            "column": "All Rows",
            "issue": f"{duplicate_rows} duplicate rows found.",
            "recommendation": "Remove duplicate rows.",
            "confidence": "High"
        })

    for col in df.columns:
        missing_count = int(df[col].isna().sum())
        missing_percent = (missing_count / total_rows * 100) if total_rows > 0 else 0

        if missing_percent > 0:
            confidence = "High" if missing_percent > 20 else "Medium"
            suggestions.append({
                "type": "Missing Values",
                "column": col,
                "issue": f"{missing_count} missing values ({missing_percent:.2f}%).",
                "recommendation": "Fill missing values or review this column.",
                "confidence": confidence
            })

        if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
            non_null = df[col].dropna().astype(str)

            if len(non_null) == 0:
                continue

            stripped_changed = (non_null != non_null.str.strip()).sum()
            if stripped_changed > 0:
                suggestions.append({
                    "type": "Text Formatting",
                    "column": col,
                    "issue": f"{stripped_changed} values contain extra spaces.",
                    "recommendation": "Trim text spaces.",
                    "confidence": "High"
                })

            lower_unique = non_null.str.lower().nunique()
            original_unique = non_null.nunique()
            if lower_unique < original_unique:
                suggestions.append({
                    "type": "Inconsistent Text Case",
                    "column": col,
                    "issue": "Values may differ only by uppercase/lowercase style.",
                    "recommendation": "Standardize text case.",
                    "confidence": "Medium"
                })

            parsed_dates = pd.to_datetime(non_null, errors="coerce")
            parse_ratio = parsed_dates.notna().mean()
            if parse_ratio > 0.7:
                suggestions.append({
                    "type": "Possible Date Column",
                    "column": col,
                    "issue": f"{parse_ratio:.0%} of values look like dates.",
                    "recommendation": "Convert this column to datetime.",
                    "confidence": "Medium"
                })

    return pd.DataFrame(suggestions)