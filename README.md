# 🎯 Smart Loan Approval System – Stacking Model

A complete **Streamlit web application** that predicts whether a loan will be **Approved** or **Rejected** using a **Stacking Ensemble Machine Learning model**.

The system combines predictions from multiple base models to make a more accurate and reliable decision.

🔗 **Deployed App:** https://stackingclass.streamlit.app/

---

## 📌 Project Overview

Loan approval decisions depend on multiple factors such as income, credit history, employment status, and property area.  
A single ML model may fail to capture all important patterns.

This project uses **Stacking Ensemble Learning**, where:
- Multiple base models learn different patterns
- A meta-model combines their predictions
- Final decision is more accurate and explainable

---

## 🧠 Model Architecture

### 🔹 Base Models
- Logistic Regression  
- Decision Tree  
- Random Forest  

### 🔹 Meta Model
- Logistic Regression  

📌 The meta-model learns from base model predictions to make the final decision.

---

## 📋 User Inputs

The application collects the following applicant details:

- Applicant Income  
- Co-Applicant Income  
- Loan Amount  
- Loan Amount Term  
- Credit History (Yes / No)  
- Employment Status (Salaried / Self-Employed)  
- Property Area (Urban / Semi-Urban / Rural)  

All inputs are user-friendly and clearly labeled.

---

## 🔘 Prediction Flow

1. User enters applicant details  
2. Base models generate individual predictions  
3. Meta-model combines base model outputs  
4. Final loan approval decision is displayed  

---

## 📊 Output Display

The app displays:

- ✅ **Loan Approved** (Green highlight)  
- ❌ **Loan Rejected** (Red highlight)  

### Additional Information:
- Base model predictions  
- Final stacking decision  
- Confidence score  
- Business explanation  

---

## 💡 Business Explanation (Mandatory Section)

The system explains decisions in simple business terms:

> “Based on income, credit history, and combined predictions from multiple models, the applicant is likely / unlikely to repay the loan. Therefore, the stacking model predicts loan approval / rejection.”

---

## 📁 Project Structure

Smart-Loan-Approval-System/
- ├── app.py
- ├── requirements.txt
- ├── README.md
- └── data/
- └── raw/
- └── train_u6lujuX_CVtuZ9i.csv

## ⚙️ How to Run the App Locally

### 1️⃣ Install Dependencies

 - pip install -r requirements.txt
### 2️⃣ Run Streamlit App
 - python -m streamlit run app.py
### 🏁 Conclusion
 - This project demonstrates an end-to-end loan approval system using stacking ensemble learning with a clean Streamlit interface.
 - It improves decision accuracy, provides explainable predictions, and aligns with real-world banking requirements.

