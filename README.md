# 💡 Medical Insurance Cost Prediction

## 📌 Project Overview
This project predicts **individual medical insurance charges** based on factors such as age, gender, BMI, smoking status, number of dependents, and region.  
It includes **data preprocessing, EDA, model training (with MLflow), and deployment using Streamlit**.

## 🛠 Skills Takeaway
- Python  
- Data Cleaning & Preprocessing  
- EDA & Feature Engineering  
- Machine Learning (Regression Models)  
- MLflow Experiment Tracking  
- Streamlit Deployment  

## 🏥 Domain
Healthcare & Insurance  

## 📝 Problem Statement
Build an end-to-end regression model to predict **medical insurance charges**.  
Deploy the best-performing model in a **Streamlit app** for user interaction.  

## 💼 Business Use Cases
- Helping insurance companies calculate personalized premiums.  
- Allowing individuals to compare insurance policies.  
- Supporting consultants to estimate medical costs.  
- Improving financial awareness among policyholders.  

## 📊 Approach
1. **Data Preprocessing**
   - Loaded dataset, handled duplicates/missing values  
   - Encoded categorical variables (sex, smoker, region)  
   - Feature engineering: BMI classification, Obesity flag  

2. **Model Training & Evaluation**
   - Trained multiple regression models: Linear Regression, Random Forest, Gradient Boosting, XGBoost, etc.  
   - Evaluated using **RMSE, MAE, R²**  
   - Logged experiments with **MLflow**  
   - Registered the best model (`GradientBoostingRegressor`)  

3. **Streamlit App**
   - Sidebar input form for patient details  
   - Predicts insurance charges instantly  
   - Shows patient summary and EDA visualizations  
   - Allows download of prediction report  
   - Saves prediction history (`predictions_history.csv`)  

## 📂 Dataset
- **medical_insurance.csv**  
- Features: `age, sex, bmi, children, smoker, region, charges`  

## 🔎 EDA Highlights
- Smokers pay **3x higher charges** than non-smokers  
- Charges increase with **age**  
- **BMI > 30 (obese)** individuals pay significantly more  
- **Region** has a minor effect compared to smoker & BMI  

## 📈 Results
- Best Model: **Gradient Boosting Regressor**  
- RMSE: ~4100  
- R² Score: ~0.82  
- Streamlit app deployed for interactive predictions  

## 🚀 Streamlit App Features
- Input: Age, Sex, BMI, Children, Smoker, Region  
- Output: Predicted insurance charges  
- Visual Insights (EDA charts)  
- Downloadable Prediction Report  

## 📦 Deliverables
- Jupyter Notebook (`medical_insurance.ipynb`)  
- Clean dataset (`medical_insurance.csv`)  
- Trained model (`gradient_boosting_model.pkl`)  
- Streamlit app (`medical.py`)  
- Prediction logs (`predictions_history.csv`)  
- Documentation (`README.md`)  

## 📊 Tools & Libraries
- Python, Pandas, NumPy, Seaborn, Matplotlib  
- scikit-learn, XGBoost  
- Streamlit  
- MLflow  

## 🏁 How to Run
```bash
pip install -r requirements.txt
streamlit run medical.py
