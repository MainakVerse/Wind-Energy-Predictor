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
    page_title="WindPro: AI Wind Energy Predictor", 
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
WindPro is an advanced AI-powered wind energy prediction platform 
that leverages machine learning to forecast wind energy generation 
with unprecedented accuracy and reliability.
""")

# Load and preprocess data
@st.cache_data
def load_data():
    wind = pd.read_csv("WIND.csv")
    wind['Date/Time'] = pd.to_datetime(wind['Date/Time'], format="%d %m %Y %H:%M")
    wind['Date'] = wind['Date/Time'].dt.normalize()
    wind['Hour'] = wind['Date/Time'].dt.hour
    wind.drop(columns=['Date/Time'], axis=1, inplace=True)
    
    hourly_avg_power = wind.groupby(['Date', 'Hour'])[['LV ActivePower (kW)', 'Theoretical_Power_Curve (KWh)', 'Wind Speed (m/s)', 'Wind Direction (°)']].mean().reset_index()
    hourly_avg_power['Month'] = hourly_avg_power['Date'].dt.month
    hourly_avg_power['Lag_1'] = hourly_avg_power['LV ActivePower (kW)'].shift(1)
    hourly_avg_power['Lag_2'] = hourly_avg_power['LV ActivePower (kW)'].shift(2)
    hourly_avg_power = hourly_avg_power.dropna()
    hourly_avg_power.set_index('Date', inplace=True)
    
    return hourly_avg_power

# Main App Tabs
tab1, tab2 = st.tabs(["Wind Energy Prediction", "Wind Map"])

with tab1:
    # Load data
    data = load_data()

    # Prepare the data
    X = data.drop(['LV ActivePower (kW)'], axis=1)
    y = data['LV ActivePower (kW)']

    # Train-Test Split
    split_ratio = 0.8
    split_index = int(len(data) * split_ratio)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    # Initialize scalers
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    # Convert Pandas Series to NumPy arrays and reshape
    X_train_np = X_train.to_numpy()
    X_test_np = X_test.to_numpy()
    y_train_np = y_train.to_numpy().reshape(-1, 1)
    y_test_np = y_test.to_numpy().reshape(-1, 1)

    # Fit and transform the training features and target
    X_train_scaled = scaler_X.fit_transform(X_train_np)
    y_train_scaled = scaler_y.fit_transform(y_train_np).ravel()

    # Transform the testing features and target
    X_test_scaled = scaler_X.transform(X_test_np)
    y_test_scaled = scaler_y.transform(y_test_np).ravel()

    # Feature inputs
    st.title("Wind Energy Generation Prediction")
    col1, col2 = st.columns(2)
    
    with col1:
        hour = st.slider("Hour of the Day", 0, 23, 12)
        theoretical_power_curve = st.number_input("Theoretical Power Curve (KWh)", min_value=0.0)
        wind_speed = st.number_input("Wind Speed (m/s)", min_value=0.0)
    
    with col2:
        wind_direction = st.number_input("Wind Direction (°)", min_value=0.0)
        month = st.slider("Month", 1, 12, 1)
        lag_1 = st.number_input("Previous Hour Power Output (kW)", min_value=0.0)
        lag_2 = st.number_input("Two Hours Ago Power Output (kW)", min_value=0.0)

    # Prepare the input for prediction
    input_data = pd.DataFrame({
        'Hour': [hour],
        'Theoretical_Power_Curve (KWh)': [theoretical_power_curve],
        'Wind Speed (m/s)': [wind_speed],
        'Wind Direction (°)': [wind_direction],
        'Month': [month],
        'Lag_1': [lag_1],
        'Lag_2': [lag_2]
    })

    # Model selection
    model_option = st.selectbox("Select Prediction Model", 
        ["Linear Regression", "XGBoost", "LSTM"], 
        key="model_selector"
    )

    # Prediction button
    if st.button("Predict Wind Energy Generation"):
        if model_option == "Linear Regression":
            # Linear Regression Model
            model = LinearRegression()
            model.fit(X_train_scaled, y_train_scaled)
            prediction_scaled = model.predict(scaler_X.transform(input_data))
            prediction = scaler_y.inverse_transform(prediction_scaled.reshape(-1, 1))[0][0]
        
        elif model_option == "XGBoost":
            # XGBoost Model with Pipeline
            pipeline = Pipeline([
                ('scaler_X', scaler_X),
                ('model', XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, objective='reg:squarederror'))
            ])
            pipeline.fit(X_train_scaled, y_train_scaled)
            prediction_scaled = pipeline.predict(scaler_X.transform(input_data))
            prediction = scaler_y.inverse_transform(prediction_scaled.reshape(-1, 1))[0][0]
        
        elif model_option == "LSTM":
            # Load pre-trained LSTM model
            model = tf.keras.models.load_model('best_lstm_model_0.92.h5')
            input_scaled = scaler_X.transform(input_data).reshape(1, 1, -1)
            prediction_scaled = model.predict(input_scaled)
            prediction = scaler_y.inverse_transform(prediction_scaled)[0][0]
        
        # Display prediction
        st.metric("Predicted Wind Energy Generation", f"{prediction:.2f} kW")
        
        # Optional: Add some context or explanation
        st.markdown("""
        ### What does this mean?
        - This prediction represents the expected electrical power output from a wind turbine.
        - The value is measured in kilowatts (kW), indicating the instantaneous power generation.
        - Factors like wind speed, direction, and time of day significantly influence the output.
        """)

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
© 2024 WindPro | AI-Powered Wind Energy Prediction
""")
