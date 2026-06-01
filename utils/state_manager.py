import streamlit as st


def initialize_session_state():
    if "df" not in st.session_state:
        st.session_state["df"] = None

    if "original_df" not in st.session_state:
        st.session_state["original_df"] = None

    if "history" not in st.session_state:
        st.session_state["history"] = []

    if "rename_report" not in st.session_state:
        st.session_state["rename_report"] = []

    if "last_uploaded_filename" not in st.session_state:
        st.session_state["last_uploaded_filename"] = None

    if "action_log" not in st.session_state:
        st.session_state["action_log"] = []

    if "trained_pipeline" not in st.session_state:
        st.session_state["trained_pipeline"] = None

    if "trained_feature_columns" not in st.session_state:
        st.session_state["trained_feature_columns"] = []

    if "trained_target_column" not in st.session_state:
        st.session_state["trained_target_column"] = None
    if "latest_model_metrics" not in st.session_state:
        st.session_state["latest_model_metrics"] = None

    if "latest_model_comparison_df" not in st.session_state:
        st.session_state["latest_model_comparison_df"] = None

    if "latest_feature_importance_df" not in st.session_state:
        st.session_state["latest_feature_importance_df"] = None

    if "latest_suspicious_df" not in st.session_state:
        st.session_state["latest_suspicious_df"] = None

    if "latest_target_df" not in st.session_state:
        st.session_state["latest_target_df"] = None

def save_state():
    if st.session_state["df"] is not None:
        st.session_state["history"].append(st.session_state["df"].copy(deep=True))


def log_action(action_name, details=""):
    st.session_state["action_log"].append({
        "action": action_name,
        "details": details
    })


def undo_last_change():
    if st.session_state["history"]:
        st.session_state["df"] = st.session_state["history"].pop()
        return True
    return False


def reset_to_original():
    if st.session_state["original_df"] is not None:
        st.session_state["df"] = st.session_state["original_df"].copy(deep=True)
        st.session_state["history"] = []
        st.session_state["action_log"] = []