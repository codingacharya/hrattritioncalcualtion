import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Set Streamlit title
st.title("🔍 HR Predictive Analytics - Attrition Prediction")

# Upload CSV File
uploaded_file = st.file_uploader("Upload Employee Data (CSV)", type=["csv"])

if uploaded_file:
    # Load dataset
    df = pd.read_csv(uploaded_file)
    
    # Display dataset preview
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # Show dataset info
    st.subheader("Dataset Summary")
    st.write(df.describe())

    # Data preprocessing
    st.subheader("Data Preprocessing")
    
    if "Attrition" not in df.columns:
        st.error("Dataset must contain an 'Attrition' column with values Yes/No.")
    else:
        # Encode categorical variables
        le = LabelEncoder()
        df["Attrition"] = le.fit_transform(df["Attrition"])  # Yes -> 1, No -> 0

        # Select features (you can modify this based on your dataset)
        features = ["Age", "MonthlyIncome", "JobSatisfaction", "YearsAtCompany", "WorkLifeBalance"]
        df = df.dropna()  # Drop missing values if any

        # Splitting data
        X = df[features]
        y = df["Attrition"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train Logistic Regression Model
        model = LogisticRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Display accuracy
        accuracy = accuracy_score(y_test, y_pred)
        st.write(f"📊 **Model Accuracy:** {accuracy:.2%}")

        # Classification report
        st.text("Classification Report:")
        st.text(classification_report(y_test, y_pred))

        # Feature importance (coefficients)
        st.subheader("Feature Importance")
        feature_importance = pd.DataFrame({"Feature": features, "Importance": np.abs(model.coef_[0])})
        feature_importance = feature_importance.sort_values(by="Importance", ascending=False)
        st.dataframe(feature_importance)

        # Visualization - Attrition Distribution
        st.subheader("Attrition Distribution")
        fig, ax = plt.subplots()
        sns.countplot(x="Attrition", data=df, ax=ax, palette="Set2")
        st.pyplot(fig)

        # Visualization - Salary vs Attrition
        st.subheader("Monthly Income vs Attrition")
        fig, ax = plt.subplots()
        sns.boxplot(x="Attrition", y="MonthlyIncome", data=df, ax=ax, palette="coolwarm")
        st.pyplot(fig)

        # Prediction on user input
        st.subheader("🔮 Predict Attrition for an Employee")
        age = st.number_input("Age", min_value=18, max_value=65, value=30)
        monthly_income = st.number_input("Monthly Income", min_value=1000, max_value=50000, value=5000)
        job_satisfaction = st.slider("Job Satisfaction (1-4)", min_value=1, max_value=4, value=3)
        years_at_company = st.slider("Years at Company", min_value=0, max_value=40, value=5)
        work_life_balance = st.slider("Work-Life Balance (1-4)", min_value=1, max_value=4, value=3)

        if st.button("Predict"):
            user_data = np.array([[age, monthly_income, job_satisfaction, years_at_company, work_life_balance]])
            prediction = model.predict(user_data)
            attrition_risk = "Yes" if prediction[0] == 1 else "No"
            st.write(f"**Attrition Prediction:** {attrition_risk} 🚀")

else:
    st.info("Please upload a CSV file to analyze attrition risk.")

