import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score

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

    # Get numeric columns for visualization
    numeric_cols = data.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()

    # For Numeric Columns - Histograms and Boxplots
    for col in numeric_cols:
        st.write(f"Distribution of {col} (Histogram):")
        fig, ax = plt.subplots()
        sns.histplot(data[col], kde=True, ax=ax)
        st.pyplot(fig)

        st.write(f"{col} (Box Plot to detect outliers):")
        fig, ax = plt.subplots()
        sns.boxplot(x=data[col], ax=ax)
        st.pyplot(fig)

    # For Numeric Columns - Scatter Plots
    if len(numeric_cols) > 1:
        st.write("Scatter Plots between numeric columns:")
        for i in range(len(numeric_cols)):
            for j in range(i + 1, len(numeric_cols)):
                fig, ax = plt.subplots()
                sns.scatterplot(x=data[numeric_cols[i]], y=data[numeric_cols[j]], ax=ax)
                st.pyplot(fig)

    # For Categorical Columns - Pie Charts
    for col in categorical_cols:
        st.write(f"Distribution of {col} (Pie Chart):")
        fig, ax = plt.subplots()
        data[col].value_counts().plot.pie(autopct='%1.1f%%', startangle=90, ax=ax, colors=sns.color_palette("Set3", len(data[col].unique())))
        ax.set_ylabel('')
        st.pyplot(fig)

    # For Categorical Columns - Bar Charts
    for col in categorical_cols:
        st.write(f"Distribution of {col} (Bar Chart):")
        fig, ax = plt.subplots()
        sns.countplot(x=data[col], ax=ax, palette="Set2")
        st.pyplot(fig)

    # Correlation heatmap for numerical columns
    if len(numeric_cols) > 1:
        st.write("Correlation heatmap:")
        fig, ax = plt.subplots(figsize=(10, 8))
        corr_matrix = data[numeric_cols].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax, fmt='.2f', linewidths=0.5, cbar_kws={'shrink': 0.8})
        st.pyplot(fig)

def apply_ml(data):
    st.subheader("Machine Learning Insights")
    target_column = st.selectbox("Select the target column for prediction:", data.columns)

    if target_column:
        X = data.drop(columns=[target_column])
        y = data[target_column]

        # Handling categorical and numerical data
        if X.select_dtypes(include=np.number).shape[1] == 0:
            st.error("All feature columns are non-numeric, unable to perform ML.")
            return

        # One-hot encoding categorical features
        X = pd.get_dummies(X, drop_first=True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Apply Linear Regression if the target is numeric
        if y.dtype in ['int64', 'float64'] and y.nunique() > 2:
            st.write("Applying Linear Regression...")
            lr = LinearRegression()
            lr.fit(X_train, y_train)
            y_pred = lr.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            st.write(f"Mean Squared Error (Linear Regression): {mse:.2f}")

            # Suggestion based on MSE
            if mse > 1000:  # Adjust threshold as per your dataset scale
                st.warning("The model has a high error. Consider feature engineering or tuning the model.")

            # Apply Random Forest Regression
            st.write("Applying Random Forest Regressor...")
            rf = RandomForestRegressor(random_state=42)
            rf.fit(X_train, y_train)
            y_pred = rf.predict(X_test)
            mse_rf = mean_squared_error(y_test, y_pred)
            st.write(f"Mean Squared Error (Random Forest): {mse_rf:.2f}")
            
            if mse_rf < mse:
                st.success("Random Forest Regressor performed better. Consider using it for prediction.")
            else:
                st.warning("Random Forest Regressor did not perform better. Try tuning the hyperparameters.")

        # Apply Random Forest Classifier if the target is categorical
        else:
            st.write("Applying Random Forest Classifier...")
            rf = RandomForestClassifier(random_state=42)
            rf.fit(X_train, y_train)
            y_pred = rf.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            st.write(f"Accuracy (Random Forest): {acc:.2f}")
            
            # Suggestion based on accuracy
            if acc < 0.7:
                st.warning("The model has low accuracy. Try tuning the model or adding more features.")

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
