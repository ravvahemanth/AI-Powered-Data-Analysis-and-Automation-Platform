import pandas as pd
from io import BytesIO
from datetime import datetime


def dataframe_to_csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8")


def dataframe_to_excel_bytes(df):
    output = BytesIO()

    try:
        with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
            df.to_excel(writer, sheet_name="Cleaned_Data", index=False)
    except ModuleNotFoundError:
        output = BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="Cleaned_Data", index=False)

    output.seek(0)
    return output.getvalue()


def history_to_csv_bytes(history_list):
    if not history_list:
        history_df = pd.DataFrame([{"action": "None", "details": "No transformation history available"}])
    else:
        history_df = pd.DataFrame(history_list)

    return history_df.to_csv(index=False).encode("utf-8")


def _df_to_html(df, max_rows=20):
    if df is None or len(df) == 0:
        return "<p>No data available.</p>"
    return df.head(max_rows).to_html(index=False, border=0)


def create_full_html_report(
    df,
    summary_dict,
    action_log=None,
    suggestions_df=None,
    suspicious_df=None,
    target_df=None,
    model_metrics=None,
    model_comparison_df=None,
    feature_importance_df=None,
    report_title="AI Data Analysis Report",
):
    rows, cols = df.shape if df is not None else (0, 0)
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    summary_html = ""
    for key, value in summary_dict.items():
        summary_html += f"<tr><td>{key}</td><td>{value}</td></tr>"

    action_log_df = pd.DataFrame(action_log) if action_log else pd.DataFrame(columns=["action", "details"])

    if model_metrics:
        model_metrics_df = pd.DataFrame(model_metrics.items(), columns=["Metric", "Value"])
    else:
        model_metrics_df = pd.DataFrame(columns=["Metric", "Value"])

    html = f"""
    <html>
    <head>
        <title>{report_title}</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 30px;
                background: #f8fafc;
                color: #1e293b;
            }}
            h1, h2, h3 {{
                color: #0f172a;
            }}
            .card {{
                background: white;
                padding: 20px;
                margin-bottom: 24px;
                border-radius: 12px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.08);
            }}
            table {{
                border-collapse: collapse;
                width: 100%;
                margin-top: 10px;
            }}
            th, td {{
                border: 1px solid #cbd5e1;
                padding: 8px;
                text-align: left;
                font-size: 14px;
            }}
            th {{
                background: #e2e8f0;
            }}
            .kpi {{
                display: inline-block;
                background: #dbeafe;
                padding: 10px 16px;
                margin-right: 10px;
                margin-bottom: 10px;
                border-radius: 10px;
                font-weight: bold;
            }}
        </style>
    </head>
    <body>
        <h1>{report_title}</h1>
        <p style="color:#64748b;font-size:13px;">Generated: {generated_at}</p>

        <div class="card">
            <h2>Dataset Overview</h2>
            <div class="kpi">Rows: {rows}</div>
            <div class="kpi">Columns: {cols}</div>
            <h3>Preview</h3>
            {_df_to_html(df, max_rows=20)}
        </div>

        <div class="card">
            <h2>Data Quality Summary</h2>
            <table>
                <tr><th>Metric</th><th>Value</th></tr>
                {summary_html}
            </table>
        </div>

        <div class="card">
            <h2>Action History</h2>
            {_df_to_html(action_log_df, max_rows=50)}
        </div>

        <div class="card">
            <h2>AI Suggestions</h2>
            {_df_to_html(suggestions_df, max_rows=50)}
        </div>

        <div class="card">
            <h2>Suspicious Columns</h2>
            {_df_to_html(suspicious_df, max_rows=50)}
        </div>

        <div class="card">
            <h2>Suggested Target Columns</h2>
            {_df_to_html(target_df, max_rows=50)}
        </div>

        <div class="card">
            <h2>Model Metrics</h2>
            {_df_to_html(model_metrics_df, max_rows=50)}
        </div>

        <div class="card">
            <h2>Model Comparison</h2>
            {_df_to_html(model_comparison_df, max_rows=50)}
        </div>

        <div class="card">
            <h2>Feature Importance</h2>
            {_df_to_html(feature_importance_df, max_rows=50)}
        </div>
    </body>
    </html>
    """

    return html.encode("utf-8")