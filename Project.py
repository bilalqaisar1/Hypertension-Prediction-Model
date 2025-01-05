#file_path = 'C:\\Users\\dell\\Desktop\\IDS-Project\\Hypertension_data.csv'
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import time
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

file_path = 'C:\\Users\\dell\\Desktop\\IDS-Project\\Hypertension_data.csv'
data = pd.read_csv(file_path)

st.set_page_config(page_title="Hypertension Dataset Analysis", layout="wide")

st.markdown("""
    <style>
    [data-testid="stSidebar"] {
        background: #36454F !important;
        color: #4E0707;
        font-family: 'Arial', sans-serif;
        width: 220px;
    }

    [data-testid="stSidebar"] .st-radio {
        color: #4E0707;
    }
    [data-testid="stSidebar"] .st-radio > div:hover {
        color: #722F37;
        background-color: rgba(78, 7, 7, 0.1);
        border-radius: 5px;
        padding: 5px;
        transition: all 0.3s ease-in-out;
    }

    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #F9E076, #F4D03F);
        color: #4E0707;
        font-family: 'Helvetica', sans-serif;
    }

    header[data-testid="stHeader"] {
        background: #36454F !important;
        color: #4E0707;
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
        color: #722F37;
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.1);
    }

    .dataframe {
        border: 1px solid #D6B85A;
        border-radius: 8px;
        box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);
        background-color: #FFFFFF;
    }

    table th {
        background-color: #D6B85A;
        color: #4E0707;
        text-transform: uppercase;
    }

    button {
        background-color: transparent;
        color: #4E0707;
        border: 1px solid #722F37;
        border-radius: 5px;
        padding: 10px 15px;
        box-shadow: 0px 4px 15px rgba(214, 184, 90, 0.3);
        transition: all 0.3s ease-in-out;
    }
    button:hover {
        background-color: rgba(78, 7, 7, 0.1);
        transform: scale(1.05);
    }

    .stActionButton {
        margin-top: -10px;
    }

    .overview-card {
        background: rgba(255, 255, 255, 0.9);
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 10px 0;
        border-left: 5px solid #C8A951;
    }
    
    .section-header {
        color: #722F37;
        font-size: 24px;
        font-weight: bold;
        margin-bottom: 15px;
        border-bottom: 2px solid #D4B85E;
        padding-bottom: 5px;
    }
    
    .description-text {
        color: #4E0707;
        font-size: 16px;
        line-height: 1.6;
        margin-bottom: 15px;
    }
    
    .highlight-box {
        background: rgba(200, 169, 81, 0.1);
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }

    /* Styling for hamburger menu button */
    button[data-testid="baseButton-headerNoPadding"] {
        background: white !important;
        border-radius: 5px !important;
        color: #4E0707 !important;
        transition: all 0.3s ease-in-out !important;
    }

    /* Hover effect for hamburger menu button */
    button[data-testid="baseButton-headerNoPadding"]:hover {
        background: #F9E076 !important;
        transform: scale(1.1) !important;
        box-shadow: 0 0 10px rgba(244, 208, 63, 0.5) !important;
    }

    /* Styling for deploy and three dots buttons */
    button[kind="secondary"] {
        background: white !important;
        color: #4E0707 !important;
    }

    button[kind="secondary"]:hover {
        background: #F9E076 !important;
        color: #4E0707 !important;
    }

    /* Additional styling for the three dots menu */
    .stDeployButton {
        background: white !important;
        color: #4E0707 !important;
    }

    .stDeployButton:hover {
        background: #F9E076 !important;
        color: #4E0707 !important;
    }

    /* Advanced Card Styling */
    .glass-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 20px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
    }

    /* Progress Bar Animation */
    .stProgress > div > div > div > div {
        background-image: linear-gradient(to right, #F4D03F, #F9E076);
    }

    /* Metric Animation */
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 15px;
        transition: transform 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
    }

    /* Loading Animation */
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    .loading {
        animation: pulse 2s infinite;
    }

    /* Tooltip Styling */
    .tooltip {
        position: relative;
        display: inline-block;
    }
    .tooltip .tooltiptext {
        visibility: hidden;
        background-color: #555;
        color: #fff;
        text-align: center;
        border-radius: 6px;
        padding: 5px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -60px;
        opacity: 0;
        transition: opacity 0.3s;
    }
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }

    .social-links {
        background: rgba(255, 255, 255, 0.1);
        padding: 15px;
        border-radius: 10px;
        margin-top: 30px;
        text-align: center;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    .social-button {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        margin: 0 10px;
        padding: 8px 15px;
        border-radius: 20px;
        text-decoration: none;
        color: white;
        font-weight: 500;
        transition: all 0.3s ease;
        border: none;
    }
    .linkedin {
        background: #0077B5;
    }
    .instagram {
        background: linear-gradient(45deg, #405DE6, #5851DB, #833AB4, #C13584, #E1306C, #FD1D1D);
    }
    .facebook {
        background: #4267B2;
    }
    .twitter {
        background: #1DA1F2;
    }
    .whatsapp {
        background: #25D366;
    }
    .social-button:hover {
        transform: translateY(-3px);
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.3);
    }
    .copyright {
        text-align: center;
        padding: 10px;
        color: #4E0707;
        font-size: 12px;
        margin-top: 10px;
    }
    </style>
""", unsafe_allow_html=True)

def overview_page():
    # Add a loading animation
    with st.spinner('Loading dashboard...'):
        time.sleep(1)
    
    # Get current time
    current_time = datetime.now()
    formatted_date = current_time.strftime("%B %d, %Y")
    formatted_time = current_time.strftime("%I:%M %p")
    
    # Using separate divs for better control
    st.markdown("""
        <style>
        .header-container {
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: rgba(255, 255, 255, 0.1);
            padding: 15px;
            border-radius: 10px;
            margin: 10px 0;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.2);
        }
        .dev-info, .time-info {
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .icon-circle {
            width: 35px;
            height: 35px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .dev-icon {
            background: #722F37;
            color: white;
        }
        .time-icon {
            background: #F9E076;
        }
        .info-text {
            text-align: left;
        }
        .info-title {
            font-size: 14px;
            font-weight: 600;
            color: #722F37;
            margin-bottom: 2px;
        }
        .info-content {
            font-size: 12px;
            display: flex;
            gap: 8px;
        }
        .dev-badge {
            background: #722F37;
            color: white;
            padding: 2px 12px;
            border-radius: 12px;
        }
        .time-badge {
            background: #F4D03F;
            color: #4E0707;
            padding: 2px 8px;
            border-radius: 12px;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown(f"""
        <div class="header-container">
            <div class="dev-info">
                <div class="icon-circle dev-icon">👨‍💻</div>
                <div class="info-text">
                    <div class="info-title">Developed By</div>
                    <div class="info-content">
                        <span class="dev-badge">Bilal Qaisar</span>
                    </div>
                </div>
            </div>
            <div class="time-info">
                <div class="icon-circle time-icon">⏰</div>
                <div class="info-text">
                    <div class="info-title">Last Updated</div>
                    <div class="info-content">
                        <span class="time-badge">📅 {formatted_date}</span>
                        <span class="time-badge">🕒 {formatted_time}</span>
                    </div>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.title("🏥 Hypertension Analysis Dashboard")
    
    # Add metrics with animations
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
            <div class='metric-card'>
                <h3>Total Records</h3>
                <h2>{}</h2>
            </div>
        """.format(len(data)), unsafe_allow_html=True)
    
    with col2:
        hypertension_rate = (data['prevalentHyp'].mean() * 100).round(2)
        st.markdown(f"""
            <div class='metric-card'>
                <h3>Hypertension Rate</h3>
                <h2>{hypertension_rate}%</h2>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        avg_age = data['age'].mean().round(2)
        st.markdown(f"""
            <div class='metric-card'>
                <h3>Average Age</h3>
                <h2>{avg_age}</h2>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class='overview-card'>
        <div class='section-header'>📊 Project Overview</div>
        <p class='description-text'>
        Welcome to our comprehensive Hypertension Analysis Dashboard! This project focuses on analyzing and predicting hypertension using various health indicators. Our interactive platform combines data visualization, statistical analysis, and machine learning to provide valuable insights into hypertension risk factors.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class='overview-card'>
        <div class='section-header'>🔍 Dataset Information</div>
        <p class='description-text'>
        Our dataset contains valuable health metrics and demographic information including:
        <ul>
            <li><strong>Age:</strong> Patient's age in years</li>
            <li><strong>BMI:</strong> Body Mass Index</li>
            <li><strong>Education:</strong> Educational background levels</li>
            <li><strong>Prevalent Hypertension:</strong> Target variable indicating hypertension status</li>
            <!-- Add other relevant features -->
        </ul>
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.write("### 📈 Sample Data Preview")
    st.dataframe(data.head(10))
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class='highlight-box'>
            <h4>Dataset Dimensions</h4>
            <p>Rows: {data.shape[0]}<br>Columns: {data.shape[1]}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='highlight-box'>
            <h4>Key Features</h4>
            <p>• Demographic Data<br>• Health Metrics<br>• Lifestyle Factors</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class='overview-card'>
        <div class='section-header'>📑 Navigation Guide</div>
        <p class='description-text'>
        <strong>EDA I:</strong> Explore feature distributions and identify outliers through interactive visualizations.<br>
        <strong>EDA II:</strong> Analyze relationships between variables using pairplots and scatter plots.<br>
        <strong>Correlation:</strong> Understand feature relationships through correlation analysis.<br>
        <strong>ML Model:</strong> Interactive prediction system using Random Forest Classifier.
        </p>
    </div>
    """, unsafe_allow_html=True)

def page_one():
    st.title("📊 Exploratory Data Analysis - Part I")
    data = pd.read_csv('Hypertension_data.csv')
    
    st.markdown("""
    <div class='overview-card'>
        <div class='section-header'>Distribution Analysis</div>
        <p class='description-text'>
        Explore the distribution patterns of various health metrics. These visualizations help identify the spread, 
        central tendency, and potential anomalies in our data.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.write("#### Feature Distributions")
    for col in data.select_dtypes(include=np.number).columns:
        st.markdown(f"""
        <div class='highlight-box'>
            <h4>{col.title()} Distribution</h4>
            <p>This histogram shows the frequency distribution of {col}, helping us understand the typical values 
            and identify any unusual patterns.</p>
        </div>
        """, unsafe_allow_html=True)
        fig, ax = plt.subplots()
        sns.histplot(data[col], kde=True, ax=ax)
        plt.title(f'Distribution of {col.title()}')
        st.pyplot(fig)
        
    st.markdown("""
    <div class='overview-card'>
        <div class='section-header'>Outlier Analysis</div>
        <p class='description-text'>
        Box plots help visualize the presence of outliers in our dataset. These outliers might represent important 
        cases or potential data quality issues.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    for col in data.select_dtypes(include=np.number).columns:
        st.markdown(f"""
        <div class='highlight-box'>
            <h4>{col.title()} Outlier Analysis</h4>
            <p>This box plot reveals the median, quartiles, and potential outliers in the {col} variable.</p>
        </div>
        """, unsafe_allow_html=True)
        fig, ax = plt.subplots()
        sns.boxplot(x=data[col], ax=ax)
        plt.title(f'Box Plot of {col.title()}')
        st.pyplot(fig)

def page_two():
    st.title("🔍 Exploratory Data Analysis - Part II")
    
    st.markdown("""
    <div class='overview-card'>
        <div class='section-header'>Multi-variable Analysis</div>
        <p class='description-text'>
        Discover relationships between different variables through advanced visualization techniques.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class='highlight-box'>
        <h4>Feature Relationships - Pairplot</h4>
        <p>This comprehensive plot shows relationships between all numeric variables, helping identify potential 
        correlations and patterns.</p>
    </div>
    """, unsafe_allow_html=True)
    pairplot_fig = sns.pairplot(data)
    st.pyplot(pairplot_fig)
    
    st.markdown("""
    <div class='highlight-box'>
        <h4>Feature Distributions - Histogram Matrix</h4>
        <p>Multiple histograms showing the distribution of each feature, helping us understand the overall data landscape.</p>
    </div>
    """, unsafe_allow_html=True)
    data.hist(bins=20, figsize=(12, 10), color='blue', edgecolor='black')
    st.pyplot(plt)
    
    st.markdown("""
    <div class='highlight-box'>
        <h4>Age vs BMI Analysis</h4>
        <p>This scatter plot reveals the relationship between age and BMI, colored by hypertension status, 
        helping identify potential risk patterns.</p>
    </div>
    """, unsafe_allow_html=True)
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=data['age'], y=data['BMI'], hue=data['prevalentHyp'], palette='coolwarm')
    plt.title('Age vs BMI by Hypertension Status')
    st.pyplot(plt)

def page_three():
    st.title("📊 Correlation Analysis")
    
    st.markdown("""
    <div class='overview-card'>
        <div class='section-header'>Understanding Feature Relationships</div>
        <p class='description-text'>
        Explore the relationships between different variables through correlation analysis. 
        The correlation coefficient ranges from -1 to 1, where:
        <ul>
            <li>1 indicates a perfect positive correlation</li>
            <li>-1 indicates a perfect negative correlation</li>
            <li>0 indicates no linear correlation</li>
        </ul>
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Parameter selection section
    st.markdown("""
    <div class='highlight-box'>
        <h4>Parameter Selection</h4>
        <p>Select multiple parameters to analyze their relationships. Choose at least 2 parameters for correlation analysis.</p>
    </div>
    """, unsafe_allow_html=True)

    # Get numeric columns
    numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    
    # Multi-select for parameters
    selected_params = st.multiselect(
        "Select Parameters for Analysis",
        options=numeric_columns,
        default=numeric_columns[:2] if len(numeric_columns) >= 2 else numeric_columns,
        help="Select two or more parameters to analyze their relationships"
    )

    # Analysis based on number of parameters selected
    if len(selected_params) < 2:
        st.warning("Please select at least 2 parameters for correlation analysis.")
        return

    # Create columns for visualization
    col1, col2 = st.columns([2, 1])

    with col1:
        if len(selected_params) == 2:
            # 2D Correlation Analysis
            correlation = data[selected_params].corr().iloc[0, 1]
            
            st.markdown("""
            <div class='highlight-box'>
                <h4>2D Correlation Analysis</h4>
            </div>
            """, unsafe_allow_html=True)
            
            fig = px.scatter(
                data,
                x=selected_params[0],
                y=selected_params[1],
                trendline="ols",
                title=f"Correlation: {correlation:.3f}",
                template="plotly_white"
            )
            fig.update_layout(title_x=0.5, title_font_size=20)
            st.plotly_chart(fig, use_container_width=True)

        elif len(selected_params) == 3:
            # 3D Correlation Analysis
            st.markdown("""
            <div class='highlight-box'>
                <h4>3D Correlation Analysis</h4>
            </div>
            """, unsafe_allow_html=True)
            
            fig = px.scatter_3d(
                data,
                x=selected_params[0],
                y=selected_params[1],
                z=selected_params[2],
                color='prevalentHyp',
                title="3D Feature Relationship"
            )
            fig.update_layout(template="plotly_white", title_x=0.5, title_font_size=20)
            st.plotly_chart(fig, use_container_width=True)

        else:
            # Fixed Correlation Matrix for more than 3 parameters
            st.markdown("""
            <div class='highlight-box'>
                <h4>Correlation Matrix</h4>
            </div>
            """, unsafe_allow_html=True)
            
            # Calculate correlation matrix
            corr_matrix = data[selected_params].corr()
            
            # Create heatmap using plotly
            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                zmin=-1,
                zmax=1,
                text=np.round(corr_matrix, 2),
                texttemplate='%{text}',
                textfont={"size": 10},
                hoverongaps=False,
                colorscale='RdBu',
                colorbar=dict(title='Correlation')
            ))
            
            fig.update_layout(
                title="Correlation Matrix Heatmap",
                title_x=0.5,
                title_font_size=20,
                width=600,
                height=600,
                xaxis_title="Features",
                yaxis_title="Features"
            )
            
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Statistical Summary
        st.markdown("""
        <div class='highlight-box'>
            <h4>Statistical Summary</h4>
        </div>
        """, unsafe_allow_html=True)
        
        stats_df = data[selected_params].describe()
        st.dataframe(
            stats_df.style.background_gradient(cmap='YlOrRd'),
            use_container_width=True
        )

        # Correlation Details
        if len(selected_params) == 2:
            from scipy import stats
            correlation_coefficient, p_value = stats.pearsonr(
                data[selected_params[0]],
                data[selected_params[1]]
            )
            
            st.markdown("""
            <div class='highlight-box'>
                <h4>Correlation Details</h4>
            </div>
            """, unsafe_allow_html=True)
            
            st.metric("Correlation Coefficient", f"{correlation_coefficient:.3f}")
            st.metric("P-Value", f"{p_value:.3e}")
            
            # Interpretation
            correlation_strength = (
                'Strong' if abs(correlation_coefficient) > 0.7 
                else 'Moderate' if abs(correlation_coefficient) > 0.3 
                else 'Weak'
            )
            
            st.markdown(f"""
            <div class='highlight-box'>
                <h4>Interpretation</h4>
                <p>• Correlation is <strong>{correlation_strength}</strong></p>
                <p>• The correlation is {'statistically significant' if p_value < 0.05 else 'not statistically significant'} (α = 0.05)</p>
                <p>• Direction: {'Positive' if correlation_coefficient > 0 else 'Negative'} correlation</p>
            </div>
            """, unsafe_allow_html=True)

    # Distribution Analysis
    st.markdown("""
    <div class='overview-card'>
        <div class='section-header'>Distribution Analysis</div>
    </div>
    """, unsafe_allow_html=True)

    # Create a row of small distribution plots
    cols = st.columns(min(3, len(selected_params)))
    for idx, param in enumerate(selected_params):
        with cols[idx % 3]:
            fig = px.histogram(
                data,
                x=param,
                marginal="box",
                title=f"Distribution of {param}",
                template="plotly_white"
            )
            fig.update_layout(showlegend=False, height=300)
            st.plotly_chart(fig, use_container_width=True)

def ml_model_page():
    # Add a progress bar for model loading
    progress_bar = st.progress(0)
    for i in range(100):
        time.sleep(0.01)
        progress_bar.progress(i + 1)
    progress_bar.empty()

    st.markdown("""
    <div class='glass-card'>
        <div class='section-header'>🤖 Machine Learning Model</div>
        <p class='description-text'>
        Our Random Forest Classifier model analyzes various health metrics to predict hypertension risk. 
        Enter your health parameters below to get a prediction.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Add feature importance plot
    X = data.drop('prevalentHyp', axis=1)
    y = data['prevalentHyp']
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=True)

    fig = go.Figure(go.Bar(
        x=feature_importance['importance'],
        y=feature_importance['feature'],
        orientation='h'
    ))
    fig.update_layout(
        title='Feature Importance',
        xaxis_title='Importance Score',
        yaxis_title='Features'
    )
    st.plotly_chart(fig, use_container_width=True)

    # Add interactive prediction interface
    st.markdown("""
    <div class='glass-card'>
        <h3>Make a Prediction</h3>
    </div>
    """, unsafe_allow_html=True)

    # Create input fields with tooltips
    features = list(X.columns)
    user_input = []
    
    col1, col2 = st.columns(2)
    with col1:
        for i in range(0, len(features), 2):
            user_input.append(
                st.number_input(
                    f"{features[i]}",
                    help=f"Enter value for {features[i]}",
                    value=float(data[features[i]].mean())
                )
            )
    
    with col2:
        for i in range(1, len(features), 2):
            if i < len(features):
                user_input.append(
                    st.number_input(
                        f"{features[i]}",
                        help=f"Enter value for {features[i]}",
                        value=float(data[features[i]].mean())
                    )
                )

    # Prediction button with loading animation
    if st.button("Predict", key="predict_button"):
        with st.spinner('Calculating prediction...'):
            time.sleep(1)
            prediction = model.predict([user_input])
            prediction_proba = model.predict_proba([user_input])
            
            # Display result with gauge chart
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = prediction_proba[0][1] * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Hypertension Risk"},
                gauge = {
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#722F37"},
                    'steps': [
                        {'range': [0, 30], 'color': "lightgreen"},
                        {'range': [30, 70], 'color': "yellow"},
                        {'range': [70, 100], 'color': "red"}
                    ]
                }
            ))
            st.plotly_chart(fig)

            result = "Hypertension" if prediction[0] == 1 else "No Hypertension"
            st.success(f"Prediction: {result}")

def main():
    st.sidebar.markdown("""
    <div style='text-align: center; margin-bottom: 20px;'>
        <h2 style='color: white; font-size: 24px; margin-bottom: 20px;'>Navigation</h2>
    </div>
    """, unsafe_allow_html=True)

    # Custom CSS for navigation buttons
    st.markdown("""
        <style>
        .nav-button {
            width: 100%;
            padding: 15px;
            margin: 8px 0;
            border: none;
            border-radius: 10px;
            background: linear-gradient(135deg, #F4D03F, #F9E076);
            color: #4E0707;
            font-weight: bold;
            font-size: 16px;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            text-align: center;
            display: block;
            text-decoration: none;
        }
        
        .nav-button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 8px rgba(0, 0, 0, 0.2);
            background: linear-gradient(135deg, #F9E076, #F4D03F);
        }
        
        .nav-button.active {
            background: linear-gradient(135deg, #722F37, #4E0707);
            color: white;
        }
        </style>
    """, unsafe_allow_html=True)

    pages = {
        "Overview": overview_page,
        "EDA I": page_one,
        "EDA II": page_two,
        "CoRelation": page_three,
        "ML Model": ml_model_page,
    }

    icons = {
        "Overview": "🏠",
        "EDA I": "📊",
        "EDA II": "🔍",
        "CoRelation": "📈",
        "ML Model": "🤖"
    }

    # Initialize session state for current page if not exists
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "Overview"

    # Create navigation buttons
    for page in pages.keys():
        is_active = " active" if st.session_state.current_page == page else ""
        if st.sidebar.button(
            f"{icons[page]} {page}", 
            key=f"btn_{page}",
            use_container_width=True,
            type="primary" if st.session_state.current_page == page else "secondary"
        ):
            st.session_state.current_page = page
            st.rerun()

    # Only copyright section
    st.sidebar.markdown("""
        <style>
        .copyright-container {
            position: fixed;
            bottom: 20px;
            left: 0;
            width: 220px;
            padding: 15px 0;
            background: linear-gradient(
                135deg, 
                rgba(255, 255, 255, 0.1), 
                rgba(255, 255, 255, 0.05)
            );
            backdrop-filter: blur(10px);
            border-top: 1px solid rgba(255, 255, 255, 0.1);
            text-align: center;
            transition: all 0.3s ease;
        }

        .copyright-text {
            font-family: 'Helvetica Neue', sans-serif;
            font-size: 11px;
            font-weight: 500;
            color: rgba(255, 255, 255, 0.9);
            letter-spacing: 0.5px;
            text-transform: uppercase;
            margin: 0;
            padding: 0;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 4px;
        }

        .copyright-icon {
            font-size: 12px;
            animation: pulse 2s infinite;
        }

        .copyright-year {
            color: #F4D03F;
            font-weight: 600;
        }

        @keyframes pulse {
            0% {
                transform: scale(1);
                opacity: 0.8;
            }
            50% {
                transform: scale(1.1);
                opacity: 1;
            }
            100% {
                transform: scale(1);
                opacity: 0.8;
            }
        }

        .copyright-container:hover {
            background: linear-gradient(
                135deg, 
                rgba(255, 255, 255, 0.15), 
                rgba(255, 255, 255, 0.08)
            );
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        }

        .copyright-container:hover .copyright-text {
            color: rgba(255, 255, 255, 1);
        }

        @media (max-width: 768px) {
            .copyright-container {
                width: 100%;
                padding: 10px 0;
            }
            .copyright-text {
                font-size: 10px;
            }
        }
        </style>

        <div class="copyright-container">
            <p class="copyright-text">
                <span class="copyright-icon">©</span>
                <span class="copyright-year">2024</span>
                All Rights Reserved
            </p>
        </div>
    """, unsafe_allow_html=True)

    # Display current page
    pages[st.session_state.current_page]()

if __name__ == "__main__":
    main()
