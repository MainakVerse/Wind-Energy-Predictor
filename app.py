import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_predict, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
import tensorflow as tf
import random
import folium
from streamlit_folium import folium_static

# Set random seeds for reproducibility
seed = 42
np.random.seed(seed)
random.seed(seed)
tf.random.set_seed(seed)

# App Configuration
st.set_page_config(
    page_title="WindPro: AI Wind Power Predictor", 
    page_icon="🌬️", 
    layout="wide"
)

# Custom CSS for Techy Design
st.markdown("""
<style>
    body {
        background-color: black;
        color: #00f0ff;
        font-family: 'Orbitron', sans-serif;
    }
    .stApp {
        background-color: black;
    }
    .stButton>button {
        color: #00f0ff;
        border: 2px solid #00f0ff;
        background-color: rgba(0, 240, 255, 0.1);
    }
    .stTextInput>div>div>input {
        color: #00f0ff;
        background-color: rgba(0, 240, 255, 0.1);
    }
    .stSlider>div>div>div>div {
        background-color: #00f0ff !important;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar with Product Description
st.sidebar.image("https://api.dicebear.com/7.x/bottts/svg?seed=WindPro", width=150)
st.sidebar.title("WindPro")
st.sidebar.markdown("""
WindPro is an advanced AI-powered wind power prediction platform 
that leverages machine learning to forecast wind energy generation 
with unprecedented accuracy and reliability.
""")

# Main App Tabs
tab1, tab2 = st.tabs(["Price Prediction", "Wind Map"])

with tab1:
    # Existing Price Prediction Logic (Condensed)
    @st.cache_data
    def load_data():
        # Your existing data loading logic here
        # Simplified for brevity
        pass

    data = load_data()

    # Feature input section
    col1, col2 = st.columns(2)
    with col1:
        hour = st.slider("Hour", 0, 23, 12, key="hour_slider")
        wind_speed = st.number_input("Wind Speed (m/s)", min_value=0.0, key="wind_speed_input")
    
    with col2:
        month = st.slider("Month", 1, 12, 1, key="month_slider")
        wind_direction = st.number_input("Wind Direction (°)", min_value=0.0, key="wind_direction_input")

    model_option = st.selectbox("Select Prediction Model", 
        ["Linear Regression", "XGBoost", "LSTM"], 
        key="model_selector"
    )

    # Prediction logic would be similar to your existing implementation
    # (Condensed for brevity)

with tab2:
    # Wind Map Section
    st.title("Real-Time Wind Map of India")
    
    # Hardcoded City Coordinates and Wind Speeds
    wind_data = {
        'Mumbai': {'coords': [19.0760, 72.8777], 'speed': 12.5},
        'Delhi': {'coords': [28.6139, 77.2090], 'speed': 8.3},
        'Bangalore': {'coords': [12.9716, 77.5946], 'speed': 10.2},
        'Chennai': {'coords': [13.0827, 80.2707], 'speed': 9.7},
        'Hyderabad': {'coords': [17.3850, 78.4867], 'speed': 11.1},
        'Kolkata': {'coords': [22.5726, 88.3639], 'speed': 7.6}
    }
    
    # Create a Folium map centered on India
    m = folium.Map(location=[20.5937, 78.9629], zoom_start=5, 
                   tiles='CartoDB dark_matter')
    
    # Add markers for wind speeds
    for city, data in wind_data.items():
        folium.CircleMarker(
            location=data['coords'],
            radius=data['speed'],
            popup=f"{city}: {data['speed']} m/s",
            color='#00f0ff',
            fill=True,
            fill_color='#00f0ff'
        ).add_to(m)
    
    # Display the map
    folium_static(m)
    
    # Wind Speed Gauge
    st.subheader("Wind Speed Highlights")
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Highest Wind Speed", f"{max(d['speed'] for d in wind_data.values())} m/s")
    
    with col2:
        st.metric("Lowest Wind Speed", f"{min(d['speed'] for d in wind_data.values())} m/s")

# Footer
st.markdown("""
---
© 2025 WindPro | AI-Powered Wind Energy Prediction
""")
