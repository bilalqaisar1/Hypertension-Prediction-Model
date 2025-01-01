#file_path = 'C:\\Users\\dell\\Desktop\\IDS-Project\\Hypertension_data.csv'
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

file_path = 'C:\\Users\\dell\\Desktop\\IDS-Project\\Hypertension_data.csv'
data = pd.read_csv(file_path)

st.set_page_config(page_title="Hypertension Dataset Analysis", layout="wide")

st.markdown("""
    <style>
    [data-testid="stSidebar"] {
        background: linear-gradient(to right, #1e88e5, #42a5f5);
        color: white;
        font-family: 'Arial', sans-serif;
        width: 220px;
    }

    [data-testid="stSidebar"] .st-radio {
        color: white;
    }
    [data-testid="stSidebar"] .st-radio > div:hover {
        color: #bbdefb;
        background-color: rgba(255, 255, 255, 0.2);
        border-radius: 5px;
        padding: 5px;
        transition: all 0.3s ease-in-out;
    }

    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #e3f2fd, #bbdefb);
        color: #0d47a1;
        font-family: 'Helvetica', sans-serif;
    }

    header[data-testid="stHeader"] {
        background: linear-gradient(135deg, #283593, #1e88e5);
        color: white;
        padding: 10px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
    }

    .stApp > header::before {
        content: 'IDS-PROJECT';
        display: block;
        color: white;
        font-size: 2rem;
        text-align: center;
        padding: 10px 0;
        margin-top: -15px;
    }

    h1, h2, h3, h4 {
        color: #283593;
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.1);
    }

    .dataframe {
        border: 1px solid #283593;
        border-radius: 8px;
        box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.2);
        background-color: #ffffff;
    }

    table th {
        background-color: #1e88e5;
        color: white;
        text-transform: uppercase;
    }

    button {
        background-color: transparent;
        color: #283593;
        border: 1px solid #283593;
        border-radius: 5px;
        padding: 10px 15px;
        box-shadow: 0px 4px 15px rgba(40, 53, 147, 0.3);
        transition: all 0.3s ease-in-out;
    }
    button:hover {
        background-color: rgba(40, 53, 147, 0.1);
        transform: scale(1.05);
    }

    .stActionButton {
        margin-top: -10px;
    }
    </style>
""", unsafe_allow_html=True)

def overview_page():
    st.title("Overview")
    st.write("### Dataset Overview")
    st.dataframe(data.head(10))
    st.write("### Key Details")
    st.write(f"Number of Rows: {data.shape[0]}")
    st.write(f"Number of Columns: {data.shape[1]}")
    st.write("### Summary Statistics")
    st.dataframe(data.describe())
    st.write("#### Missing Values")
    st.dataframe(data.isnull().sum())

def page_one():
    st.title("Exploratory Data Analysis")
    data = pd.read_csv('Hypertension_data.csv')
    st.write("#### Feature Distributions")
    for col in data.select_dtypes(include=np.number).columns:
        fig, ax = plt.subplots()
        sns.histplot(data[col], kde=True, ax=ax)
        st.pyplot(fig)
    st.write("#### Box Plots for Outliers")
    for col in data.select_dtypes(include=np.number).columns:
        fig, ax = plt.subplots()
        sns.boxplot(x=data[col], ax=ax)
        st.pyplot(fig)

def page_two():
    st.write("### Pairplot of Features")
    pairplot_fig = sns.pairplot(data)
    st.pyplot(pairplot_fig)
    st.write("### Histograms of Features")
    data.hist(bins=20, figsize=(12, 10), color='blue', edgecolor='black')
    st.pyplot(plt)
    st.write("### Scatter Plot: Age vs BMI")
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=data['age'], y=data['BMI'], hue=data['prevalentHyp'], palette='coolwarm')
    plt.title('Scatter Plot: Age vs BMI')
    st.pyplot(plt)
    st.write("### Count Plot: Education Levels")
    if 'education' in data.columns:
        plt.figure(figsize=(8, 6))
        sns.countplot(x=data['education'])
        st.pyplot(plt)

def page_three():
    st.write("### Correlation Heatmap")
    plt.figure(figsize=(10, 6))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    st.pyplot(plt)

def ml_model_page():
    st.write("### Machine Learning Model")
    st.write("#### Model: Random Forest Classifier")
    X = data.drop('prevalentHyp', axis=1)
    y = data['prevalentHyp']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    st.write("#### Classification Report")
    st.dataframe(pd.DataFrame(report).transpose())
    st.write("#### Make a Prediction")
    cols = st.columns(len(X.columns))
    user_input = []
    for idx, col in enumerate(X.columns):
        with cols[idx % len(cols)]:
            user_input.append(st.number_input(f"{col}", value=0))

    center = st.container()
    with center:
        if st.button("Predict", key="predict_button"):
            prediction = model.predict([user_input])
            st.write(f"### Prediction: {'Hypertension' if prediction[0] == 1 else 'No Hypertension'}")

def main():
    st.sidebar.title("Navigation")
    pages = {
        "Overview": overview_page,
        "EDA I": page_one,
        "EDA II": page_two,
        "CoRelation": page_three,
        "ML Model": ml_model_page,
    }
    choice = st.sidebar.radio("Go to", list(pages.keys()))
    pages[choice]()

if __name__ == "__main__":
    main()
