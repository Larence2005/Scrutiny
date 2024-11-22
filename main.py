import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score
import mysql.connector

def load_data(file):
    if file.name.endswith('.csv'):
        return pd.read_csv(file)
    elif file.name.endswith('.xlsx'):
        return pd.read_excel(file)
    else:
        st.error("Unsupported file format. Please upload a CSV or Excel file.")
        return None

def visualize_data(data):
    st.subheader("Data Visualization")
    st.write("Automatically generated plots:")
    numeric_cols = data.select_dtypes(include=np.number).columns.tolist()

    for col in numeric_cols:
        st.write(f"Distribution of {col}:")
        fig, ax = plt.subplots()
        sns.histplot(data[col], kde=True, ax=ax)
        st.pyplot(fig)

    if len(numeric_cols) > 1:
        st.write("Correlation heatmap:")
        fig, ax = plt.subplots()
        sns.heatmap(data[numeric_cols].corr(), annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)

def apply_ml(data):
    st.subheader("Machine Learning Insights")
    target_column = st.selectbox("Select the target column for prediction:", data.columns)

    if target_column:
        X = data.drop(columns=[target_column])
        y = data[target_column]

        if X.select_dtypes(include=np.number).shape[1] == 0:
            st.error("All feature columns are non-numeric, unable to perform ML.")
            return

        X = pd.get_dummies(X, drop_first=True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        if y.dtype in ['int64', 'float64'] and y.nunique() > 2:
            # Regression
            st.write("Applying Linear Regression...")
            lr = LinearRegression()
            lr.fit(X_train, y_train)
            y_pred = lr.predict(X_test)
            st.write("Mean Squared Error:", mean_squared_error(y_test, y_pred))

            st.write("Applying Random Forest Regressor...")
            rf = RandomForestRegressor(random_state=42)
            rf.fit(X_train, y_train)
            y_pred = rf.predict(X_test)
            st.write("Mean Squared Error:", mean_squared_error(y_test, y_pred))
        else:
            # Classification
            st.write("Applying Random Forest Classifier...")
            rf = RandomForestClassifier(random_state=42)
            rf.fit(X_train, y_train)
            y_pred = rf.predict(X_test)
            st.write("Accuracy Score:", accuracy_score(y_test, y_pred))

def main():
    st.title("AI-Powered Dataset Analyzer")
    st.write("Upload your dataset, and this tool will analyze and provide insights automatically.")

    uploaded_file = st.file_uploader("Upload your dataset (CSV or Excel):", type=['csv', 'xlsx'])
    if uploaded_file:
        data = load_data(uploaded_file)
        if data is not None:
            st.subheader("Data Preview")
            st.write(data.head())

            st.subheader("Data Summary")
            st.write(data.describe())

            if st.checkbox("Visualize Data"):
                visualize_data(data)

            if st.checkbox("Apply Machine Learning"):
                apply_ml(data)

if __name__ == "__main__":
    main()
