import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

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

    # Get numeric and categorical columns for visualization
    numeric_cols = data.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()

    # For Numeric Columns - Histograms, Boxplots, Line Charts, Scatter Plots, etc.
    for col in numeric_cols:
        st.write(f"Visualize {col}:")
        graph_type = st.selectbox(f"Select graph type for {col}:", ["Histogram", "Box Plot", "Scatter Plot", "Line Chart", "Area Chart", "None"], key=f"num_{col}")

        if graph_type == "Histogram":
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.histplot(data[col], kde=True, ax=ax)
            st.pyplot(fig)

        elif graph_type == "Box Plot":
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.boxplot(x=data[col], ax=ax)
            st.pyplot(fig)

        elif graph_type == "Scatter Plot":
            if len(numeric_cols) > 1:
                other_col = st.selectbox(f"Select another column to plot with {col}:", numeric_cols, key=f"scatter_{col}")
                if other_col != col:
                    fig, ax = plt.subplots(figsize=(8, 5))
                    sns.scatterplot(x=data[col], y=data[other_col], ax=ax)
                    st.pyplot(fig)

        elif graph_type == "Line Chart":
            fig, ax = plt.subplots(figsize=(8, 5))
            data[col].plot(kind='line', ax=ax)
            st.pyplot(fig)

        elif graph_type == "Area Chart":
            fig, ax = plt.subplots(figsize=(8, 5))
            data[col].plot(kind='area', ax=ax, alpha=0.3)
            st.pyplot(fig)

    # For Categorical Columns - Pie Charts, Bar Charts, Count Plots
    for col in categorical_cols:
        st.write(f"Visualize {col}:")
        graph_type = st.selectbox(f"Select graph type for {col}:", ["Pie Chart", "Bar Chart", "Count Plot", "None"], key=f"cat_{col}")

        if graph_type == "Pie Chart":
            fig, ax = plt.subplots(figsize=(8, 5))
            data[col].value_counts().plot.pie(autopct='%1.1f%%', startangle=90, ax=ax, colors=sns.color_palette("Set3", len(data[col].unique())))
            ax.set_ylabel('')
            st.pyplot(fig)

        elif graph_type == "Bar Chart":
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.countplot(x=data[col], ax=ax, palette="Set2")
            st.pyplot(fig)

        elif graph_type == "Count Plot":
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.countplot(x=data[col], ax=ax)
            st.pyplot(fig)

    # Correlation heatmap for numerical columns
    if len(numeric_cols) > 1:
        st.write("Correlation heatmap for numeric variables:")
        fig, ax = plt.subplots(figsize=(12, 8))
        corr_matrix = data[numeric_cols].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax, fmt='.2f', linewidths=0.5, cbar_kws={'shrink': 0.8})
        st.pyplot(fig)

    # Pairplot for all numeric variables
    if len(numeric_cols) > 1:
        st.write("Pairplot for numeric variables:")
        fig = sns.pairplot(data[numeric_cols])
        st.pyplot(fig)

    # Distribution of each numeric column using box plot
    if len(numeric_cols) > 0:
        st.write("Distribution of all numeric variables:")
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.boxplot(data=data[numeric_cols], ax=ax)
        st.pyplot(fig)

    # Heatmap of categorical column values
    if len(categorical_cols) > 0:
        st.write("Heatmap for categorical columns:")
        fig, ax = plt.subplots(figsize=(12, 8))
        heatmap_data = pd.crosstab(index=data[categorical_cols[0]], columns=data[categorical_cols[1]] if len(categorical_cols) > 1 else data[categorical_cols[0]])
        sns.heatmap(heatmap_data, annot=True, fmt='d', cmap='Blues', ax=ax)
        st.pyplot(fig)

    # Violin Plot for numeric columns
    if len(numeric_cols) > 1:
        st.write("Violin plot for numeric variables:")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.violinplot(data=data[numeric_cols], ax=ax)
        st.pyplot(fig)

    # KDE Plot for numeric columns
    if len(numeric_cols) > 1:
        st.write("KDE plot for numeric variables:")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.kdeplot(data=data[numeric_cols[0]], shade=True, ax=ax)
        st.pyplot(fig)

def main():
    st.title("Dynamic Dataset Analyzer")
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

if __name__ == "__main__":
    main()
