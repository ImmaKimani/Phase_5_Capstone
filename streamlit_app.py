import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, time
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .fraud-alert {
        background-color: #ffebee;
        border: 2px solid #f44336;
        padding: 1rem;
        border-radius: 0.5rem;
        color: #c62828;
    }
    .safe-alert {
        background-color: #e8f5e8;
        border: 2px solid #4caf50;
        padding: 1rem;
        border-radius: 0.5rem;
        color: #2e7d32;
    }
</style>
""", unsafe_allow_html=True)

# Load trained models and preprocessors
@st.cache_resource
def load_model_artifacts():
    """Load the trained model and preprocessing artifacts"""
    try:
        # Load your simplified model files
        model = joblib.load('xgboost_simple_fraud_model.pkl')
        feature_names = joblib.load('simple_feature_names.pkl')
        # scaler = joblib.load('simple_scaler.pkl')  # Optional
        return model, feature_names
    except FileNotFoundError as e:
        st.error(f"Model files not found: {e}")
        st.info("Please ensure these files are in your app directory: xgboost_simple_fraud_model.pkl, simple_feature_names.pkl")
        return None, None

# Feature engineering functions
def calculate_time_features(transaction_datetime):
    """Extract time-based features"""
    hour = transaction_datetime.hour
    day_of_week = transaction_datetime.weekday()
    return hour, day_of_week

def calculate_amount_features(amount):
    """Calculate amount-based features"""
    log_amount = np.log(amount + 1)
    # Use the same outlier bounds from your notebook
    amt_outlier = 1 if amount < -100.73 or amount > 193.67 else 0
    return log_amount, amt_outlier

def create_feature_vector(user_inputs, feature_names):
    """Create feature vector matching your model's expected 11 features"""
    # Your model expects these 11 features in this order:
    # ['amt', 'lat', 'long', 'city_pop', 'merch_lat', 'merch_long', 'gender', 'hour', 'day_of_week', 'amt_outlier', 'log_amt']
    
    features = []
    
    for feature_name in feature_names:
        if feature_name == 'amt':
            features.append(user_inputs['amount'])
        elif feature_name == 'lat':
            features.append(user_inputs['cardholder_lat'])
        elif feature_name == 'long':
            features.append(user_inputs['cardholder_long'])
        elif feature_name == 'city_pop':
            features.append(user_inputs['city_pop'])
        elif feature_name == 'merch_lat':
            features.append(user_inputs['merchant_lat'])
        elif feature_name == 'merch_long':
            features.append(user_inputs['merchant_long'])
        elif feature_name == 'gender':
            features.append(1 if user_inputs['gender'] == 'M' else 0)
        elif feature_name == 'hour':
            features.append(user_inputs['hour'])
        elif feature_name == 'day_of_week':
            features.append(user_inputs['day_of_week'])
        elif feature_name == 'amt_outlier':
            features.append(user_inputs['amt_outlier'])
        elif feature_name == 'log_amt':
            features.append(user_inputs['log_amt'])
        else:
            features.append(0)  # Default value for unknown features
    
    return np.array(features).reshape(1, -1)

# Main application
def main():
    st.markdown('<h1 class="main-header">💳 Credit Card Fraud Detection System</h1>', unsafe_allow_html=True)
    
    # Load model
    model, feature_names = load_model_artifacts()
    
    if model is None or feature_names is None:
        st.stop()
    
    # Display model info
    st.info(f"✅ Model loaded successfully! Using {len(feature_names)} features: {', '.join(feature_names)}")
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", ["🔍 Fraud Detection", "📊 Model Analytics", "ℹ️ About"])
    
    if page == "🔍 Fraud Detection":
        fraud_detection_page(model, feature_names)
    elif page == "📊 Model Analytics":
        model_analytics_page()
    else:
        about_page()

def fraud_detection_page(model, feature_names):
    """Main fraud detection interface"""
    st.header("🔍 Real-Time Fraud Detection")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Transaction Details")
        
        # Transaction information
        with st.form("transaction_form"):
            # Basic transaction details
            amount = st.number_input("Transaction Amount ($)", min_value=0.01, value=75.50, step=0.01)
            
            # Date and time
            transaction_date = st.date_input("Transaction Date", value=datetime.now().date())
            transaction_time = st.time_input("Transaction Time", value=datetime.now().time())
            
            # Location information
            st.subheader("Location Details")
            col_loc1, col_loc2 = st.columns(2)
            
            with col_loc1:
                cardholder_lat = st.number_input("Cardholder Latitude", value=40.7128, format="%.4f")
                cardholder_long = st.number_input("Cardholder Longitude", value=-74.0060, format="%.4f")
                city_pop = st.number_input("City Population", min_value=1, value=50000)
            
            with col_loc2:
                merchant_lat = st.number_input("Merchant Latitude", value=40.7589, format="%.4f")
                merchant_long = st.number_input("Merchant Longitude", value=-73.9851, format="%.4f")
            
            # Demographic info
            st.subheader("Additional Information")
            gender = st.selectbox("Gender", ["F", "M"])
            
            # Submit button
            submitted = st.form_submit_button("🔍 Check for Fraud", use_container_width=True)
        
        if submitted:
            # Combine date and time
            transaction_datetime = datetime.combine(transaction_date, transaction_time)
            
            # Feature engineering
            hour, day_of_week = calculate_time_features(transaction_datetime)
            log_amount, amt_outlier = calculate_amount_features(amount)
            
            # Prepare user inputs
            user_inputs = {
                'amount': amount,
                'cardholder_lat': cardholder_lat,
                'cardholder_long': cardholder_long,
                'city_pop': city_pop,
                'merchant_lat': merchant_lat,
                'merchant_long': merchant_long,
                'gender': gender,
                'hour': hour,
                'day_of_week': day_of_week,
                'amt_outlier': amt_outlier,
                'log_amt': log_amount
            }
            
            # Create feature vector
            try:
                features = create_feature_vector(user_inputs, feature_names)
                
                # Make prediction
                prediction_proba = model.predict_proba(features)[0]
                fraud_probability = prediction_proba[1]  # Probability of fraud
                is_fraud = fraud_probability > 0.5
                
                # Display results in the right column
                with col2:
                    display_prediction_results(fraud_probability, is_fraud, amount, user_inputs)
                    
            except Exception as e:
                st.error(f"Prediction error: {str(e)}")
                st.info("Please check that all inputs are valid.")

def display_prediction_results(fraud_probability, is_fraud, amount, user_inputs):
    """Display prediction results with visualizations"""
    st.subheader("🎯 Prediction Results")
    
    # Risk level indicator
    risk_level = "HIGH" if fraud_probability > 0.7 else "MEDIUM" if fraud_probability > 0.3 else "LOW"
    risk_color = "#ff4444" if risk_level == "HIGH" else "#ffaa00" if risk_level == "MEDIUM" else "#44ff44"
    
    # Main prediction display
    if is_fraud:
        st.markdown(f"""
        <div class="fraud-alert">
            <h3>⚠️ FRAUD DETECTED</h3>
            <p><strong>Risk Probability:</strong> {fraud_probability:.1%}</p>
            <p><strong>Risk Level:</strong> {risk_level}</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="safe-alert">
            <h3>✅ TRANSACTION SAFE</h3>
            <p><strong>Risk Probability:</strong> {fraud_probability:.1%}</p>
            <p><strong>Risk Level:</strong> {risk_level}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Risk gauge chart
    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = fraud_probability * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Fraud Risk %"},
        delta = {'reference': 50},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': risk_color},
            'steps': [
                {'range': [0, 30], 'color': "lightgreen"},
                {'range': [30, 70], 'color': "yellow"},
                {'range': [70, 100], 'color': "lightcoral"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    fig_gauge.update_layout(height=300)
    st.plotly_chart(fig_gauge, use_container_width=True)
    
    # Additional metrics
    st.subheader("📋 Transaction Summary")
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Amount", f"${amount:,.2f}")
        st.metric("Hour", f"{user_inputs['hour']:02d}:00")
    
    with col2:
        st.metric("Risk Score", f"{fraud_probability:.1%}")
        st.metric("Status", "🚨 BLOCK" if is_fraud else "✅ APPROVE")

def model_analytics_page():
    """Display model performance analytics"""
    st.header("📊 Model Performance Analytics")
    
    # Model performance metrics (from your actual results)
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Precision", "19%", help="Accuracy of fraud predictions")
    with col2:
        st.metric("Recall", "86%", help="Percentage of fraud cases caught")
    with col3:
        st.metric("AUC-ROC", "0.9831", help="Overall model discrimination ability")
    with col4:
        st.metric("Accuracy", "98%", help="Overall classification accuracy")
    
    # Feature importance (from your actual results)
    st.subheader("🎯 Key Fraud Indicators")
    
    # Actual feature importance from your model
    feature_importance = {
        'Transaction Amount': 0.488,
        'Hour of Day': 0.127,
        'Amount Outlier Flag': 0.116,
        'Log Amount': 0.111,
        'Gender': 0.064,
        'Day of Week': 0.019,
        'City Population': 0.019,
        'Cardholder Latitude': 0.018,
        'Cardholder Longitude': 0.017,
        'Merchant Longitude': 0.011,
        'Merchant Latitude': 0.008
    }
    
    fig_importance = px.bar(
        x=list(feature_importance.values()),
        y=list(feature_importance.keys()),
        orientation='h',
        title="Feature Importance in Fraud Detection (XGBoost Model)"
    )
    fig_importance.update_layout(xaxis_title="Importance Score", yaxis_title="Features")
    st.plotly_chart(fig_importance, use_container_width=True)
    
    # Model insights
    st.subheader("🔍 Model Insights")
    
    insights = [
        f"💰 **Transaction Amount** is the most important feature (48.8% importance)",
        f"🕐 **Hour of Day** is the second most important (12.7% importance)",
        f"📊 **Amount Outlier Detection** contributes significantly (11.6% importance)",
        f"🎯 Model achieves **98.31% AUC-ROC** with only 11 features",
        f"⚡ **86% Recall** means the model catches 86% of all fraud cases",
        f"🎯 Model focuses on transaction patterns rather than demographic factors"
    ]
    
    for insight in insights:
        st.write(f"• {insight}")

def about_page():
    """About page with project information"""
    st.header("ℹ️ About This Project")
    
    st.markdown("""
    ### 🎯 Project Overview
    This Credit Card Fraud Detection System was developed using advanced machine learning techniques 
    to identify potentially fraudulent transactions in real-time.
    
    ### 📊 Dataset & Methodology
    - **Dataset Size**: 1,296,675 transactions with 11 key features
    - **Time Period**: January 2019 - July 2019
    - **Class Imbalance**: Only 0.58% fraudulent transactions
    - **Solution**: SMOTE (Synthetic Minority Oversampling Technique)
    
    ### 🤖 Machine Learning Model
    - **Algorithm**: XGBoost Classifier (Simplified)
    - **Performance**: 86% Recall, 19% Precision, 0.9831 AUC-ROC
    - **Key Strength**: Excellent fraud detection with minimal features
    - **Features Used**: Only 11 most important features for easy deployment
    
    ### 📈 Model Performance
    - **Training Data**: 2.06M transactions after SMOTE balancing
    - **Test Accuracy**: 98% overall accuracy
    - **AUC-ROC**: 0.9831 (exceptional discrimination ability)
    - **Deployment**: Optimized for real-time prediction
    
    ### 👥 Development Team
    **Group 3 Members:**
    - Immaculate Kimani
    - Joan Gatharia  
    - Bertha Karuku
    - James Mwaura
    - Evelyne Mwangi
    - John Mugambi
    
    ### 🔧 Technical Stack
    - **Data Processing**: Python, Pandas, NumPy
    - **Machine Learning**: Scikit-learn, XGBoost, SMOTE
    - **Visualization**: Matplotlib, Seaborn, Plotly
    - **Deployment**: Streamlit
    
    ### ⚠️ Important Notes
    - This is a demonstration system for educational purposes
    - Model optimized for high recall (catching fraud) over precision
    - Real-world deployment requires additional security measures
    - Model performance validated on 259,335 test transactions
    """)

if __name__ == "__main__":
    main()