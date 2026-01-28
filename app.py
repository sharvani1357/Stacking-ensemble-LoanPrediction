import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.metrics import accuracy_score

# --------------------------------------------------
# Page Config
# --------------------------------------------------
st.set_page_config(
    page_title="Smart Loan Approval System",
    layout="centered"
)

# --------------------------------------------------
# App Title & Description
# --------------------------------------------------
st.title("🎯 Smart Loan Approval System – Stacking Model")
st.write(
    "This system uses a **Stacking Ensemble Machine Learning model** to predict "
    "whether a loan will be approved by combining multiple ML models for better decision making."
)

st.markdown("---")

# --------------------------------------------------
# Load Dataset
# --------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("data/raw/train_u6lujuX_CVtuZ9i.csv")

df = load_data()

# --------------------------------------------------
# Data Preprocessing
# --------------------------------------------------
# Handle missing values
for col in df.select_dtypes(include=["int64", "float64"]):
    df[col].fillna(df[col].mean(), inplace=True)

for col in df.select_dtypes(include=["object"]):
    df[col].fillna(df[col].mode()[0], inplace=True)

# Feature & Target split
X = df.drop(["Loan_ID", "Loan_Status"], axis=1)
y = df["Loan_Status"].map({"Y": 1, "N": 0})

# Encoding categorical features
X = pd.get_dummies(X, drop_first=True)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# --------------------------------------------------
# Sidebar – User Input Section
# --------------------------------------------------
st.sidebar.header("📋 Applicant Details")

app_income = st.sidebar.number_input("Applicant Income", value=5000)
coapp_income = st.sidebar.number_input("Co-Applicant Income", value=2000)
loan_amount = st.sidebar.number_input("Loan Amount", value=150)
loan_term = st.sidebar.number_input("Loan Amount Term", value=360)

credit_history = st.sidebar.radio("Credit History", ["Yes", "No"])
credit_history_val = 1 if credit_history == "Yes" else 0

employment = st.sidebar.selectbox(
    "Employment Status",
    ["Salaried", "Self-Employed"]
)

property_area = st.sidebar.selectbox(
    "Property Area",
    ["Urban", "Semiurban", "Rural"]
)

# --------------------------------------------------
# Prepare User Input
# --------------------------------------------------
input_data = pd.DataFrame({
    "ApplicantIncome": [app_income],
    "CoapplicantIncome": [coapp_income],
    "LoanAmount": [loan_amount],
    "Loan_Amount_Term": [loan_term],
    "Credit_History": [credit_history_val],
    "Self_Employed_Yes": [1 if employment == "Self-Employed" else 0],
    "Property_Area_Semiurban": [1 if property_area == "Semiurban" else 0],
    "Property_Area_Urban": [1 if property_area == "Urban" else 0],
})

# Align columns
input_data = input_data.reindex(columns=X.columns, fill_value=0)
input_scaled = scaler.transform(input_data)

# --------------------------------------------------
# Model Architecture Display
# --------------------------------------------------
st.markdown("## 🧠 Stacking Model Architecture")
st.write("""
**Base Models Used:**
- Logistic Regression  
- Decision Tree  
- Random Forest  

**Meta Model Used:**
- Logistic Regression  
""")

st.markdown("---")

# --------------------------------------------------
# Build Base Models
# --------------------------------------------------
base_models = [
    ("lr", LogisticRegression(max_iter=1000)),
    ("dt", DecisionTreeClassifier(max_depth=3, random_state=42)),
    ("rf", RandomForestClassifier(n_estimators=100, random_state=42))
]

# --------------------------------------------------
# Build Stacking Model
# --------------------------------------------------
stack_model = StackingClassifier(
    estimators=base_models,
    final_estimator=LogisticRegression(),
    cv=5
)

stack_model.fit(X_train, y_train)

# --------------------------------------------------
# Prediction Button
# --------------------------------------------------
if st.button("🔘 Check Loan Eligibility (Stacking Model)"):

    # Base model predictions
    lr_pred = base_models[0][1].fit(X_train, y_train).predict(input_scaled)[0]
    dt_pred = base_models[1][1].fit(X_train, y_train).predict(input_scaled)[0]
    rf_pred = base_models[2][1].fit(X_train, y_train).predict(input_scaled)[0]

    # Stacking prediction
    final_pred = stack_model.predict(input_scaled)[0]
    confidence = np.max(stack_model.predict_proba(input_scaled)) * 100

    # --------------------------------------------------
    # Output Section
    # --------------------------------------------------
    st.markdown("## 📊 Prediction Result")

    if final_pred == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")

    st.markdown("### 📊 Base Model Predictions")
    st.write(f"**Logistic Regression:** {'Approved' if lr_pred == 1 else 'Rejected'}")
    st.write(f"**Decision Tree:** {'Approved' if dt_pred == 1 else 'Rejected'}")
    st.write(f"**Random Forest:** {'Approved' if rf_pred == 1 else 'Rejected'}")

    st.markdown("### 🧠 Final Stacking Decision")
    st.write(f"**Confidence Score:** {confidence:.2f}%")

    # --------------------------------------------------
    # Business Explanation
    # --------------------------------------------------
    st.markdown("### 💡 Business Explanation")
    st.info(
        "Based on the applicant’s income, credit history, loan amount, and "
        "combined predictions from multiple machine learning models, the system "
        "evaluates repayment capability. Therefore, the stacking model predicts "
        f"that the loan will be **{'approved' if final_pred == 1 else 'rejected'}**."
    )
