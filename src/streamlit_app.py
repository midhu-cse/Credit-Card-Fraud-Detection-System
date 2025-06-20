import streamlit as st
import os
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
import json

# Page config
st.set_page_config(
    page_title="🔒 Fraud Detection System",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        color: black;
    }
    .fraud-alert {
        background: #ffe6e6;
        border: 2px solid #ff4444;
        padding: 1rem;
        border-radius: 8px;
        color: #cc0000;
    }
    .safe-alert {
        background: #e6ffe6;
        border: 2px solid #44ff44;
        padding: 1rem;
        border-radius: 8px;
        color: #008800;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🔒 Credit Card Fraud Detection System</h1>
        <p>AI-Powered Transaction Security Monitor</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("🎛️ Control Panel")
    
    # API Status Check
    try:
        response = requests.get("http://127.0.0.1:8000/health", timeout=5)
        if response.status_code == 200:
            st.sidebar.success("✅ API Connected")
            api_available = True
        else:
            st.sidebar.error("❌ API Error")
            api_available = False
    except:
        st.sidebar.error("❌ API Not Running")
        st.sidebar.info("Start API: `uvicorn main:app --reload`")
        api_available = False
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Single Prediction", "📊 Batch Analysis", "📈 Model Info", "🧪 Test Cases"])
    
    with tab1:
        single_prediction_tab(api_available)
    
    with tab2:
        batch_analysis_tab(api_available)
    
    with tab3:
        model_info_tab(api_available)
    
    with tab4:
        test_cases_tab(api_available)

def single_prediction_tab(api_available):
    st.header("🔍 Single Transaction Analysis")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Transaction Details")
        
        # Amount input
        amount = st.number_input("💰 Transaction Amount ($)", min_value=0.01, value=100.0, step=0.01)
        
        # Option to use sample data or manual input
        input_method = st.radio("Input Method:", ["🎲 Random Sample", "✏️ Manual Input"])
        
        if input_method == "🎲 Random Sample":
            if st.button("🎲 Generate Random Transaction"):
                # Generate random V1-V28 values (typical range: -3 to 3)
                st.session_state.v_values = np.random.normal(0, 1, 28).tolist()
        
        # Initialize if not exists
        if 'v_values' not in st.session_state:
            st.session_state.v_values = [0.0] * 28
        
        if input_method == "✏️ Manual Input":
            st.write("**PCA Components (V1-V28):**")
            cols = st.columns(4)
            for i in range(28):
                with cols[i % 4]:
                    st.session_state.v_values[i] = st.number_input(
                        f"V{i+1}", 
                        value=st.session_state.v_values[i], 
                        step=0.1,
                        key=f"v_{i+1}"
                    )
        
        # Prediction button
        if st.button("🔍 Analyze Transaction", type="primary") and api_available:
            make_prediction(amount, st.session_state.v_values)
    
    with col2:
        st.subheader("📊 Quick Stats")
        
        # Show some sample statistics
        st.markdown("""
                    
        <div class="metric-card" >
            <h4>🎯 Model Accuracy</h4>
            <p>~99.9% AUC Score</p>
        </div>
        <br>
        <div class="metric-card">
            <h4>⚡ Response Time</h4>
            <p>< 100ms average</p>
        </div>
        <br>
        <div class="metric-card">
            <h4>📈 Features Used</h4>
            <p>29 transaction features</p>
        </div>
        """, unsafe_allow_html=True)

def make_prediction(amount, v_values):
    """Make fraud prediction"""
    
    # Prepare data
    transaction_data = {
        "Amount": amount
    }
    
    # Add V1-V28 values
    for i, v in enumerate(v_values):
        transaction_data[f"V{i+1}"] = v
    
    try:
        # Make API request
        response = requests.post(
            "http://127.0.0.1:8000/predict",
            json=transaction_data,
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            
            # Display results
            col1, col2, col3 = st.columns(3)
            
            with col1:
                fraud_status = "🚨 FRAUDULENT" if result['is_fraud'] else "✅ LEGITIMATE"
                st.metric("Status", fraud_status)
            
            with col2:
                st.metric("Fraud Probability", f"{result['fraud_probability']:.1%}")
            
            with col3:
                st.metric("Risk Level", result['risk_level'])
            
            # Alert box
            if result['is_fraud']:
                st.markdown(f"""
                <div class="fraud-alert">
                    <h3>🚨 FRAUD DETECTED!</h3>
                    <p><strong>Probability:</strong> {result['fraud_probability']:.1%}</p>
                    <p><strong>Risk Level:</strong> {result['risk_level']}</p>
                    <p><strong>Recommendation:</strong> Block transaction and investigate</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="safe-alert">
                    <h3>✅ TRANSACTION APPROVED</h3>
                    <p><strong>Probability:</strong> {result['fraud_probability']:.1%}</p>
                    <p><strong>Risk Level:</strong> {result['risk_level']}</p>
                    <p><strong>Recommendation:</strong> Proceed with transaction</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Probability gauge
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = result['fraud_probability'] * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Fraud Probability (%)"},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 30], 'color': "lightgreen"},
                        {'range': [30, 70], 'color': "yellow"},
                        {'range': [70, 100], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
            
        else:
            st.error(f"Prediction failed: {response.text}")
            
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")

def batch_analysis_tab(api_available):
    st.header("📊 Batch Transaction Analysis")
    
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write(f"📁 Loaded {len(df)} transactions")
        st.dataframe(df.head())
        
        if st.button("🔍 Analyze All Transactions") and api_available:
            progress_bar = st.progress(0)
            results = []
            
            for i, row in df.iterrows():
                # Prepare transaction data
                transaction_data = row.to_dict()
                
                try:
                    response = requests.post(
                        "http://127.0.0.1:8000/predict",
                        json=transaction_data,
                        timeout=5
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        results.append({
                            'Transaction_ID': i,
                            'Amount': row.get('Amount', 0),
                            'Is_Fraud': result['is_fraud'],
                            'Fraud_Probability': result['fraud_probability'],
                            'Risk_Level': result['risk_level']
                        })
                    
                    progress_bar.progress((i + 1) / len(df))
                    
                except Exception as e:
                    st.error(f"Error processing transaction {i}: {str(e)}")
                    break
            
            # Display results
            if results:
                results_df = pd.DataFrame(results)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    fraud_count = results_df['Is_Fraud'].sum()
                    st.metric("🚨 Fraudulent", fraud_count)
                
                with col2:
                    legitimate_count = len(results_df) - fraud_count
                    st.metric("✅ Legitimate", legitimate_count)
                
                with col3:
                    fraud_rate = fraud_count / len(results_df) * 100
                    st.metric("📊 Fraud Rate", f"{fraud_rate:.1f}%")
                
                # Charts
                fig = px.histogram(results_df, x='Risk_Level', color='Is_Fraud',
                                title="Risk Level Distribution")
                st.plotly_chart(fig, use_container_width=True)
                
                st.dataframe(results_df)
                
                # Download results
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results",
                    data=csv,
                    file_name="fraud_analysis_results.csv",
                    mime="text/csv"
                )

def model_info_tab(api_available):
    st.header("📈 Model Information")
    
    if api_available:
        try:
            response = requests.get("http://127.0.0.1:8000/model/info", timeout=5)
            if response.status_code == 200:
                info = response.json()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("🧠 Model Details")
                    st.write(f"**Type:** {info['model_type']}")
                    st.write(f"**Estimators:** {info['n_estimators']}")
                    st.write(f"**Max Depth:** {info['max_depth']}")
                    st.write(f"**Features:** {info['features_count']}")
                
                with col2:
                    st.subheader("🎯 Performance Metrics")
                    st.write("**AUC Score:** ~99.9%")
                    st.write("**Precision:** ~85%")
                    st.write("**Recall:** ~92%")
                    st.write("**F1-Score:** ~88%")
                
                # Feature importance (if available)
                if os.path.exists("reports/feature_importance.png"):
                    st.subheader("📊 Feature Importance")
                    image = Image.open("reports/feature_importance.png")
                    st.image(image, caption="Top Features for Fraud Detection")
                
                # Confusion matrix (if available)
                if os.path.exists("reports/confusion_matrix.png"):
                    st.subheader("🎯 Confusion Matrix")
                    image = Image.open("reports/confusion_matrix.png")
                    st.image(image, caption="Model Performance Matrix")
            
        except Exception as e:
            st.error(f"Error fetching model info: {str(e)}")
    else:
        st.warning("⚠️ API not available. Start the FastAPI server to view model information.")

def test_cases_tab(api_available):
    st.header("🧪 Pre-built Test Cases")
    
    # Sample test cases
    test_cases = {
        "💳 Normal Transaction": {
            "Amount": 50.0,
            "description": "Typical small purchase",
            "V1": 0.1, "V2": -0.5, "V3": 0.3, "V4": -0.1, "V5": 0.2,
            "V6": 0.1, "V7": -0.2, "V8": 0.0, "V9": 0.1, "V10": -0.1,
            "V11": 0.0, "V12": 0.1, "V13": -0.1, "V14": 0.2, "V15": 0.0,
            "V16": -0.1, "V17": 0.1, "V18": 0.0, "V19": 0.1, "V20": -0.1,
            "V21": 0.0, "V22": 0.1, "V23": -0.1, "V24": 0.0, "V25": 0.1,
            "V26": -0.1, "V27": 0.0, "V28": 0.1
        },
        "🚨 Suspicious Transaction": {
            "Amount": 5000.0,
            "description": "Large amount with unusual patterns",
            "V1": -2.5, "V2": 3.1, "V3": -1.8, "V4": 2.2, "V5": -3.0,
            "V6": 1.9, "V7": -2.1, "V8": 2.8, "V9": -1.5, "V10": 2.0,
            "V11": -2.3, "V12": 1.7, "V13": -2.9, "V14": 2.4, "V15": -1.6,
            "V16": 2.1, "V17": -2.7, "V18": 1.8, "V19": -2.0, "V20": 2.6,
            "V21": -1.9, "V22": 2.3, "V23": -2.8, "V24": 1.5, "V25": -2.2,
            "V26": 2.9, "V27": -1.7, "V28": 2.5
        },
        "💰 High-Value Normal": {
            "Amount": 2000.0,
            "description": "High-value but legitimate pattern",
            "V1": 0.5, "V2": -0.3, "V3": 0.7, "V4": -0.2, "V5": 0.4,
            "V6": 0.3, "V7": -0.6, "V8": 0.1, "V9": 0.8, "V10": -0.4,
            "V11": 0.2, "V12": 0.6, "V13": -0.3, "V14": 0.5, "V15": 0.1,
            "V16": -0.7, "V17": 0.4, "V18": 0.2, "V19": 0.9, "V20": -0.5,
            "V21": 0.3, "V22": 0.7, "V23": -0.2, "V24": 0.4, "V25": 0.6,
            "V26": -0.8, "V27": 0.1, "V28": 0.5
        }
    }
    
    for case_name, case_data in test_cases.items():
        with st.expander(case_name):
            st.write(f"**Description:** {case_data['description']}")
            st.write(f"**Amount:** ${case_data['Amount']}")
            
            if st.button(f"Test {case_name}", key=case_name) and api_available:
                # Remove description for API call
                api_data = {k: v for k, v in case_data.items() if k != 'description'}
                
                try:
                    response = requests.post(
                        "http://127.0.0.1:8000/predict",
                        json=api_data,
                        timeout=5
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Result", "🚨 FRAUD" if result['is_fraud'] else "✅ SAFE")
                        with col2:
                            st.metric("Probability", f"{result['fraud_probability']:.1%}")
                        with col3:
                            st.metric("Risk", result['risk_level'])
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()