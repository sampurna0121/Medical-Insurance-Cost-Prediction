# medical.py
import os
import io
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# ---------------------------
# Page setup
# ---------------------------
st.set_page_config(
    page_title="Medical Insurance Charges",
    page_icon="💡",
    layout="wide",
)

# ---------------------------
# Utilities
# ---------------------------
FEATURE_ORDER = ["age", "sex", "bmi", "children", "smoker", "region", "obese"]

REGION_MAP = {
    "northeast": 0,
    "northwest": 1,
    "southeast": 2,
    "southwest": 3
}

def bmi_category(bmi: float) -> str:
    if bmi < 18.5: return "Underweight"
    if bmi < 25:   return "Normal"
    if bmi < 30:   return "Overweight"
    return "Obese"

@st.cache_resource
def load_model(path: str = "gradient_boosting_model.pkl"):
    return joblib.load(path)

@st.cache_data
def load_dataset(csv_path: str = "medical_insurance.csv"):
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        return df
    return None

def make_feature_frame(age, sex, bmi, children, smoker, region):
    """Return a single-row DataFrame with the EXACT feature order your model expects."""
    sex_enc = 1 if sex == "male" else 0
    smoker_enc = 1 if smoker == "yes" else 0
    region_enc = REGION_MAP[region]
    obese = int(bmi > 30)

    row = {
        "age": age,
        "sex": sex_enc,
        "bmi": bmi,
        "children": children,
        "smoker": smoker_enc,
        "region": region_enc,
        "obese": obese,
    }
    return pd.DataFrame([row], columns=FEATURE_ORDER), obese

def format_money(x):
    try:
        return f"${x:,.2f}"
    except Exception:
        return str(x)

# ---------------------------
# Sidebar inputs
# ---------------------------
st.sidebar.header("Enter Patient Details")

age = st.sidebar.number_input("Age", min_value=18, max_value=100, value=30, step=1)
sex = st.sidebar.selectbox("Sex", ["male", "female"], index=0)
bmi = st.sidebar.number_input("BMI (Body Mass Index)", min_value=10.0, max_value=60.0, value=25.0, step=0.1)
children = st.sidebar.number_input("Number of Children", min_value=0, max_value=10, value=0, step=1)
smoker = st.sidebar.selectbox("Smoker", ["yes", "no"], index=1)
region = st.sidebar.selectbox("Region", list(REGION_MAP.keys()), index=0)

# ---------------------------
# Main header
# ---------------------------
st.markdown("## 💡 Medical Insurance Charges Prediction")
st.caption("Predict medical insurance charges based on health & lifestyle factors.")

# Build features for prediction
X_pred, obese_flag = make_feature_frame(age, sex, bmi, children, smoker, region)
bmi_cat = bmi_category(bmi)

# Load model
try:
    model = load_model("gradient_boosting_model.pkl")
except Exception as e:
    st.error("Could not load the model file `gradient_boosting_model.pkl`.")
    st.exception(e)
    st.stop()

# Predict
pred_btn = st.sidebar.button("Predict Charges", use_container_width=True)
if pred_btn:
    try:
        pred = float(model.predict(X_pred)[0])
        st.success(f"🪙 Estimated Insurance Charges: {format_money(pred)}")
    except Exception as e:
        st.error("Prediction failed. Check feature encoding/order and model compatibility.")
        st.exception(e)

# ---------------------------
# Patient Summary (nice layout)
# ---------------------------
st.markdown("### 🧾 Patient Summary")

c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
c1.metric("Age", age)
c2.metric("Sex", sex.capitalize())
c3.metric("BMI", f"{bmi:.1f}")
c4.metric("Children", children)
c5.metric("Smoker", smoker.upper())
c6.metric("Region", region.capitalize())
c7.metric("Obese", "Yes" if obese_flag else "No")

# Table view (encoded features)
with st.expander("See encoded features sent to the model"):
    st.dataframe(X_pred)

# ---------------------------
# Optional EDA & Visuals (if CSV is available)
# ---------------------------
st.markdown("### 📊 Insights")

df = load_dataset("medical_insurance.csv")
if df is None:
    st.info("`medical_insurance.csv` not found next to the app. Place it there to see visual insights.")
else:
    # Basic clean
    df = df.drop_duplicates().copy()
    df["obese"] = df["bmi"] > 30

    # Layout for charts
    g1, g2 = st.columns(2)

    # 1) Smoker vs Charges
    with g1:
        st.caption("Average Charges by Smoking Status")
        fig, ax = plt.subplots(figsize=(5, 3))
        sns.barplot(data=df, x="smoker", y="charges", ax=ax, estimator=np.mean, errorbar=None)
        ax.set_xlabel("Smoker")
        ax.set_ylabel("Avg Charges")
        st.pyplot(fig)

    # 2) Region-wise avg charges
    with g2:
        st.caption("Average Charges by Region")
        fig, ax = plt.subplots(figsize=(5, 3))
        sns.barplot(data=df, x="region", y="charges", ax=ax, estimator=np.mean, errorbar=None)
        ax.set_xlabel("Region")
        ax.set_ylabel("Avg Charges")
        st.pyplot(fig)

    # 3) Age vs Charges (colored by smoker)
    st.caption("Charges vs Age (colored by smoker)")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.scatterplot(data=df, x="age", y="charges", hue="smoker", alpha=0.6, ax=ax)
    ax.set_xlabel("Age")
    ax.set_ylabel("Charges")
    st.pyplot(fig)

# ---------------------------
# Download mini report
# ---------------------------
st.markdown("### ⬇️ Download Prediction Report")

report = {
    "age": age,
    "sex": sex,
    "bmi": round(bmi, 1),
    "children": int(children),
    "smoker": smoker,
    "region": region,
    "obese": "yes" if obese_flag else "no",
}

if pred_btn:
    report["predicted_charges"] = pred
else:
    report["predicted_charges"] = None

rep_df = pd.DataFrame([report])

csv_buf = io.StringIO()
rep_df.to_csv(csv_buf, index=False)
st.download_button(
    label="Download CSV",
    data=csv_buf.getvalue(),
    file_name="insurance_prediction_report.csv",
    mime="text/csv",
)

# --- Save prediction history ---
history_row = {
    "age": age,
    "sex": sex,
    "bmi": bmi,
    "children": children,
    "smoker": smoker,
    "region": region,
    "obese": "yes" if obese_flag else "no",
    "predicted_charges": pred if pred_btn else None
}

history_df = pd.DataFrame([history_row])

# Append to CSV (create if not exists)
try:
    history_df.to_csv(
        "predictions_history.csv",
        mode="a",
        header=not os.path.exists("predictions_history.csv"),
        index=False
    )
    st.info("📂 Prediction saved to 'predictions_history.csv'")
except Exception as e:
    st.error(f"⚠️ Could not save prediction history: {e}")

# ---------------------------
# Model info
# ---------------------------
with st.expander("ℹ️ Model Info"):
    st.write("""
    **Algorithm:** Gradient Boosting Regressor  
    **Inputs expected (in order):** `age, sex, bmi, children, smoker, region, obese`  
    - `sex`: male=1, female=0  
    - `smoker`: yes=1, no=0  
    - `region`: northeast=0, northwest=1, southeast=2, southwest=3  
    - `obese`: 1 if BMI>30 else 0
    """)
