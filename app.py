# app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go

# ---------------------------------------
# Page config: must be first Streamlit call
# ---------------------------------------
st.set_page_config(page_title="Bankruptcy Prediction App", page_icon="🏦", layout="wide")

# ---------------------------------------
# Minimal CSS for nicer cards & spacing
# (keeps overall Streamlit look but tidies UI)
# ---------------------------------------
st.markdown(
    """
    <style>
    .card {
        background: #ffffff;
        border: 1px solid rgba(0,0,0,0.06);
        border-radius: 10px;
        padding: 18px;
        box-shadow: 0 6px 20px rgba(16,24,40,0.04);
    }
    .result-card {
        border-radius: 10px;
        padding: 14px;
    }
    .suggestion {
        padding: 10px 12px;
        border-radius: 8px;
        font-weight: 600;
        margin-top: 8px;
    }
    .good { background:#e9f9ef; border-left:6px solid #00b37a; color:#0a3d2e; }
    .bad  { background:#fff3f3; border-left:6px solid #e63946; color:#641414; }
    .small-muted { color: #6b7280; font-size:13px; }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------------------------
# Constants and model loading
# ---------------------------------------
MODEL_PATH = "final_logreg_model.pkl"
LABEL_CANDIDATES = ["class", "Class", "target", "Target", "y", "label", "Label"]

@st.cache_resource
def load_model(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)

try:
    model = load_model(MODEL_PATH)
except FileNotFoundError:
    st.error(f"Model file '{MODEL_PATH}' not found in the app folder. Upload it and restart.")
    st.stop()

# ---------------------------------------
# Helpers
# ---------------------------------------
def prepare_features_from_df(df: pd.DataFrame, model) -> pd.DataFrame:
    X = df.copy()
    # remove common label column (if present)
    for c in LABEL_CANDIDATES:
        if c in X.columns:
            X = X.drop(columns=[c])
            break
    if hasattr(model, "feature_names_in_"):
        expected = list(model.feature_names_in_)
        for col in expected:
            if col not in X.columns:
                X[col] = 0.0
        X = X.reindex(columns=expected)
    else:
        X = X.select_dtypes(include=[np.number])
    return X.fillna(0)

def get_class_indices(model):
    classes = list(getattr(model, "classes_", []))
    idx_bank = None
    try:
        idx_bank = classes.index(0)
    except Exception:
        for i, c in enumerate(classes):
            s = str(c).lower()
            if "bank" in s:
                idx_bank = i
                break
    if idx_bank is None:
        idx_bank = 0
    idx_non = 1 if idx_bank == 0 and len(classes) > 1 else 0
    return idx_bank, idx_non

def donut_plot(prob_bank, prob_non, title="Predicted probabilities"):
    fig = go.Figure(
        data=[
            go.Pie(
                labels=["Bankruptcy", "Non-Bankruptcy"],
                values=[prob_bank, prob_non],
                hole=0.54,
                marker=dict(colors=["#ef553b", "#00cc96"]),
                sort=False,
                textinfo="none",
                hoverinfo="label+percent"
            )
        ]
    )
    # central label
    fig.update_layout(
        annotations=[dict(text=f"{prob_bank*100:.1f}%<br>Bankrupt", x=0.5, y=0.5, font_size=18, showarrow=False)],
        showlegend=True,
        margin=dict(l=0, r=0, t=30, b=0),
        title={"text": title, "x":0.5}
    )
    return fig

# ---------------------------------------
# Sidebar (info + upload)
# ---------------------------------------
with st.sidebar:
    st.title("About")
    st.write("Bankruptcy prediction using a trained Logistic Regression model.")
    st.markdown("---")
    uploaded = st.file_uploader("Upload CSV or Excel for batch prediction (optional)", type=["csv", "xlsx"])
    st.caption("If your file has `class` labels (text or numeric), the app will show counts in the pie chart.")
    st.markdown("---")
    st.write("Model info:")
    st.write(f"- Type: `{model.__class__.__name__}`")
    st.write(f"- Probabilities: {'Yes' if hasattr(model,'predict_proba') else 'No'}")
    if hasattr(model, "classes_"):
        st.write(f"- Classes: {list(model.classes_)}")
    st.markdown("---")
    st.write("Developer:Pranav K")

# ---------------------------------------
# Main header
# ---------------------------------------
st.title("🏦 Bankruptcy Prediction App")
st.write("Use manual inputs for a single prediction (donut chart + metrics) or upload a dataset for batch predictions.")

# ---------------------------------------
# Single prediction UI (improved)
# ---------------------------------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("### 📊 Single Company Prediction", unsafe_allow_html=True)
st.markdown("<div class='small-muted'>Choose values (0, 0.5, 1) for each indicator and click Predict.</div>", unsafe_allow_html=True)

opts = [0.0, 0.5, 1.0]
col1, col2 = st.columns(2)
with col1:
    industrial_risk = st.selectbox("Industrial Risk", opts, index=1)
    management_risk = st.selectbox("Management Risk", opts, index=1)
    financial_flexibility = st.selectbox("Financial Flexibility", opts, index=1)
with col2:
    credibility = st.selectbox("Credibility", opts, index=1)
    competitiveness = st.selectbox("Competitiveness", opts, index=1)
    operating_risk = st.selectbox("Operating Risk", opts, index=1)

st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

single_df = pd.DataFrame([{
    "industrial_risk": industrial_risk,
    "management_risk": management_risk,
    "financial_flexibility": financial_flexibility,
    "credibility": credibility,
    "competitiveness": competitiveness,
    "operating_risk": operating_risk
}])

# Predict button and display
if st.button("🔍 Predict (Single)"):
    try:
        X_single = prepare_features_from_df(single_df, model)
        pred = model.predict(X_single)[0]
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_single)[0]
            idx_bank, idx_non = get_class_indices(model)
            prob_bank = float(probs[idx_bank]) if idx_bank is not None else 0.0
            prob_non = float(probs[idx_non]) if idx_non is not None else (1.0 - prob_bank)
        else:
            prob_bank = 1.0 if pred == 0 else 0.0
            prob_non = 1.0 - prob_bank

        # Top result card (two-column)
        c1, c2 = st.columns([1, 1])
        with c1:
            if pred == 0:
                st.markdown("<div class='result-card bad'>", unsafe_allow_html=True)
                st.error("⚠️ Prediction: RISK OF BANKRUPTCY")
                st.markdown("</div>", unsafe_allow_html=True)
                st.markdown("<div class='suggestion bad'>💡 Suggestion: Improve liquidity and reduce operating/management risk.</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='result-card'>", unsafe_allow_html=True)
                st.success("✅ Prediction: FINANCIALLY HEALTHY (Non-Bankrupt)")
                st.markdown("</div>", unsafe_allow_html=True)
                st.markdown("<div class='suggestion good'>💡 Suggestion: Maintain competitiveness and strong financial flexibility.</div>", unsafe_allow_html=True)

            st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
            # probability metrics
            st.metric("Bankruptcy Probability", f"{prob_bank*100:.1f}%")
            st.metric("Non-Bankruptcy Probability", f"{prob_non*100:.1f}%")

        # Donut chart on the right
        with c2:
            fig = donut_plot(prob_bank, prob_non, title="Predicted probability (donut)")
            st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Prediction failed: {e}")

st.markdown("</div>", unsafe_allow_html=True)  # close card

# ---------------------------------------
# Batch upload & simple pie summary
# ---------------------------------------
if uploaded is not None:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.header("📁 Batch Dataset Prediction (Uploaded)")
    try:
        if uploaded.name.endswith(".csv"):
            df = pd.read_csv(uploaded)
        else:
            df = pd.read_excel(uploaded)
    except Exception as e:
        st.error(f"Could not read file: {e}")
        df = None

    if df is not None:
        st.subheader("Uploaded data preview")
        st.dataframe(df.head())

        try:
            X = prepare_features_from_df(df, model)
            preds = model.predict(X)
            df["Prediction"] = np.where(preds == 0, "Bankruptcy", "Non-Bankruptcy")
            st.success("Batch predictions complete.")
            st.dataframe(df.head())

            bankrupt_count = (df["Prediction"] == "Bankruptcy").sum()
            non_bankrupt_count = (df["Prediction"] == "Non-Bankruptcy").sum()

            st.subheader("📈 Dataset Summary")
            colA, colB = st.columns(2)
            colA.metric("Bankruptcy cases", int(bankrupt_count))
            colB.metric("Non-Bankruptcy cases", int(non_bankrupt_count))

            # Pie chart for dataset
            plot_pie = go.Figure(data=[go.Pie(
                labels=["Bankruptcy", "Non-Bankruptcy"],
                values=[bankrupt_count, non_bankrupt_count],
                hole=0.45,
                marker=dict(colors=["#ef553b", "#00cc96"]),
                textinfo="label+percent"
            )])
            plot_pie.update_layout(title_text="Bankruptcy distribution in uploaded dataset", title_x=0.5, margin=dict(t=30, b=0))
            st.plotly_chart(plot_pie, use_container_width=True)

            # download
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download Predictions CSV", csv, "predictions.csv", "text/csv")

        except Exception as e:
            st.error(f"Batch prediction failed: {e}")

    st.markdown("</div>", unsafe_allow_html=True)


