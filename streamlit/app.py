"""
MLOps Monitoring Dashboard
Streamlit UI for monitoring ML predictions and system health
"""
import os
import streamlit as st
import requests
from datetime import datetime
import pandas as pd

# Configuration
API_URL = os.getenv('API_URL', 'http://api:8000')
MLFLOW_URL = os.getenv('MLFLOW_URL', 'http://mlflow:5000')

# Page config
st.set_page_config(
    page_title="MLOps Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title
st.title("🤖 MLOps Monitoring Dashboard")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    st.text(f"API: {API_URL}")
    st.text(f"MLflow: {MLFLOW_URL}")
    st.markdown("---")
    
    st.header("📋 Navigation")
    page = st.radio(
        "Select View",
        ["Model Info", "Predictions", "System Health"]
    )

# Model Info Page
if page == "Model Info":
    st.header("📊 Current Model Information")
    
    try:
        response = requests.get(f"{API_URL}/model-info", timeout=5)
        if response.status_code == 200:
            model_info = response.json()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Model Loaded", 
                         "✅ Yes" if model_info.get('model_loaded') else "❌ No")
                st.metric("Model Version", model_info.get('model_version', 'N/A'))
                
            with col2:
                st.metric("MLflow URI", model_info.get('mlflow_uri', 'N/A'))
                st.metric("Model Name", model_info.get('model_name', 'N/A'))
            
            st.json(model_info)
        else:
            st.error(f"Failed to fetch model info: {response.status_code}")
    except Exception as e:
        st.error(f"Error connecting to API: {str(e)}")

# Predictions Page
elif page == "Predictions":
    st.header("🎯 Test Predictions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        amount = st.number_input("Transaction Amount ($)", min_value=0.0, value=150.0, step=10.0)
        quantity = st.number_input("Quantity", min_value=1, value=2, step=1)
        category = st.selectbox("Product Category", 
                               ['electronics', 'clothing', 'food', 'books', 'sports'])
    
    with col2:
        customer_type = st.selectbox("Customer Type", 
                                    ['regular', 'premium', 'vip'])
        location = st.selectbox("Location", ['US', 'EU', 'ASIA', 'AU'])
    
    if st.button("🚀 Make Prediction"):
        try:
            payload = {
                'amount': amount,
                'quantity': quantity,
                'product_category': category,
                'customer_type': customer_type,
                'location': location
            }
            
            response = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                
                st.success("✅ Prediction successful!")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Prediction", 
                             "High Value ✨" if result.get('prediction') == 1 else "Regular 💰")
                with col2:
                    st.metric("Confidence", f"{result.get('probability', 0) * 100:.1f}%")
                with col3:
                    st.metric("Model Version", result.get('model_version', 'N/A'))
                
                st.json(result)
            else:
                st.error(f"Prediction failed: {response.status_code}")
                st.text(response.text)
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")

# System Health Page
elif page == "System Health":
    st.header("🏥 System Health Check")
    
    services = {
        'API': f"{API_URL}/health",
        'Model Info': f"{API_URL}/model-info",
    }
    
    for service_name, endpoint in services.items():
        try:
            response = requests.get(endpoint, timeout=5)
            if response.status_code == 200:
                st.success(f"✅ {service_name}: Healthy")
                with st.expander(f"View {service_name} Details"):
                    st.json(response.json())
            else:
                st.error(f"❌ {service_name}: Unhealthy ({response.status_code})")
        except Exception as e:
            st.error(f"❌ {service_name}: Unreachable - {str(e)}")

# Footer
st.markdown("---")
st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")




