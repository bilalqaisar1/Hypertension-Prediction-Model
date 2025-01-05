
# Hypertension Dataset Analysis Dashboard

This project is an **interactive web-based dashboard** for exploring and analyzing a hypertension dataset using Python. It includes advanced data visualization, exploratory data analysis (EDA), correlation analysis, and machine learning model predictions.

---

## Features

### 🔍 **Exploratory Data Analysis (EDA)**
- **Distribution Analysis**: Histograms and box plots to analyze feature distributions and outliers.
- **Multi-variable Analysis**: Pair plots, scatter plots, and relationship matrices for deeper insights into the data.

### 📈 **Correlation Analysis**
- Generate a detailed correlation matrix heatmap.
- Supports both 2D and 3D visualization for feature relationships.

### 🤖 **Machine Learning Model**
- **Random Forest Classifier**: Predicts the risk of hypertension based on health metrics.
- **Feature Importance**: Visualizes the most influential features in predicting hypertension.
- **Interactive Prediction**: Users can input their own data to get hypertension risk predictions.

### 🖥️ **Interactive Dashboard**
- Built using **Streamlit**, with an intuitive multi-page navigation system.
- Custom styling and animations enhance the user experience.

---

## Dataset Information

The dataset contains various health and demographic metrics that are key in understanding hypertension risk, including:

| **Feature**              | **Description**                                     |
|--------------------------|-----------------------------------------------------|
| `age`                    | Age of the patient (in years)                       |
| `BMI`                    | Body Mass Index                                     |
| `prevalentHyp`           | Hypertension status (target variable)               |
| `education`              | Education level                                     |
| Other metrics            | Additional health and lifestyle factors             |

---

## Pages Overview

1. **Overview**: Project summary, dataset preview, and key insights.
2. **EDA I**: Feature distributions and outlier detection.
3. **EDA II**: Multi-variable analysis with pair plots and scatter plots.
4. **Correlation**: Correlation matrix and feature relationship analysis.
5. **ML Model**: Machine learning-based hypertension prediction.

---

## Technologies Used

- **Python Libraries**:  
  - `pandas` for data manipulation  
  - `matplotlib` & `seaborn` for static plots  
  - `plotly` for interactive visualizations  
  - `scikit-learn` for machine learning  
  - `streamlit` for building the web app
- **Front-end Enhancements**: Custom CSS styling for a polished look.

---

## How to Run Locally

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/your-repository.git
   cd your-repository
   ```

2. **Install the required packages**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit app**:
   ```bash
   streamlit run Project.py
   ```

---


---

## Developer

- **Bilal Qaisar**  
  Data Analyst and Dashboard Developer  
  [LinkedIn](#) • [GitHub](#) • [YouTube](#)

---

## License

This project is licensed under the MIT License.
