import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_predict,cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import LSTM, Dense
import tensorflow as tf
import random

# Set random seeds for reproducibility
seed = 42
np.random.seed(seed)
random.seed(seed)
tf.random.set_seed(seed)

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
y_train_np = y_train.to_numpy().reshape(-1, 1)  # Reshape to 2D
y_test_np = y_test.to_numpy().reshape(-1, 1)    # Reshape to 2D

# Fit and transform the training features and target
X_train_scaled = scaler_X.fit_transform(X_train_np)
y_train_scaled = scaler_y.fit_transform(y_train_np).ravel()  # Flatten back to 1D

# Transform the testing features and target
X_test_scaled = scaler_X.transform(X_test_np)
y_test_scaled = scaler_y.transform(y_test_np).ravel()



# Streamlit app
st.title("Wind Power Prediction")

# Feature inputs
st.sidebar.header("Input Features")
hour = st.sidebar.slider("Hour", 0, 23, 12)
theoretical_power_curve = st.sidebar.number_input("Theoretical Power Curve (KWh)", min_value=0.0)
wind_speed = st.sidebar.number_input("Wind Speed (m/s)", min_value=0.0)
wind_direction = st.sidebar.number_input("Wind Direction (°)", min_value=0.0)
month = st.sidebar.slider("Month", 1, 12, 1)
lag_1 = st.sidebar.number_input("Lag_1", min_value=0.0)
lag_2 = st.sidebar.number_input("Lag_2", min_value=0.0)

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
model_option = st.sidebar.selectbox("Select Model", ["Linear Regression", "XGBoost with pipeline","LSTM"])

if model_option == "Linear Regression":
    model = LinearRegression()
    model.fit(X_train_scaled, y_train_scaled)
    y_test_pred_scaled = model.predict(X_test_scaled)

    y_test_pred = scaler_y.inverse_transform(y_test_pred_scaled.reshape(-1, 1)).ravel()

    # Calculate metrics
    mae_lr = round(mean_absolute_error(y_test_scaled, y_test_pred_scaled), 2)
    mse_lr = round(mean_squared_error(y_test_scaled, y_test_pred_scaled), 2)
    rmse_lr = round(np.sqrt(mse_lr), 2)
    r2_lr = round(r2_score(y_test_scaled, y_test_pred_scaled), 2)
    
    # Display the prediction
    prediction = model.predict(input_data)
    st.subheader("Prediction")
    st.write(f"Predicted Wind Power (kW): {prediction[0]:.2f}")

    st.subheader("Metrics")
    st.write(f"Mean Absolute Error (MAE): {mae_lr}")
    st.write(f"Mean Squared Error (MSE): {mse_lr}")
    st.write(f"Root Mean Squared Error (RMSE): {rmse_lr}")
    st.write(f"R² Score: {r2_lr}")
    
    # Plot Actual vs Predicted
    st.subheader("Plots")

    # Create DataFrame for plotting
    df = pd.DataFrame({
        'Actual': y_test,
        'Predicted': y_test_pred
    }).reset_index()  # Reset index to avoid plotting issues

    # Calculate R^2 value
    r2 = r2_score(y_test, y_test_pred)

    # Create the plots
    plt.figure(figsize=(14, 7))

    # Plot 1: Regression Plot
    plt.subplot(1, 2, 1)
    sns.regplot(x='Actual', y='Predicted', data=df, ci=None, 
                scatter_kws={"color": "blue", "s": 10, "marker": "*"},  # Star markers
                line_kws={"color": "red", "alpha": 0.7})
    plt.xlabel('Actual Values', color='blue')
    plt.ylabel('Predicted Values', color='blue')
    plt.title(f'Regression Plot: Actual vs Predicted\n$R^2 = {r2:.4f}$', color='blue')

    # Plot 2: Line Plot of Actual vs Predicted
    y_test_limited = y_test.head(100).reset_index(drop=True)
    y_test_pred_limited = pd.Series(y_test_pred).head(100).reset_index(drop=True)
    sample_numbers = np.arange(len(y_test_limited))

    plt.subplot(1, 2, 2)
    plt.plot(sample_numbers, y_test_limited, color='blue', label='Actual Values', linestyle='-', marker='o')
    plt.plot(sample_numbers, y_test_pred_limited, color='red', label='Predicted Values', linestyle='--', marker='x')
    plt.xlabel('Sample Number')
    plt.ylabel('Values')
    plt.title('Actual vs Predicted LV ActivePower (kW)')
    plt.legend()

    plt.tight_layout()
    st.pyplot(plt)  # Display the plots in Streamlit

elif model_option == "XGBoost with pipeline":
    pipeline = Pipeline([('scaler_X', scaler_X),('model', XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, objective='reg:squarederror'))])
    X = np.array(X)  # Convert to NumPy arrays if they are not already
    y = np.array(y)

   

    # Initialize scalers
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    # Reshape y to be a 2D column vector
    y_reshaped = y.reshape(-1, 1)

    # Fit and transform the features and target variable
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y_reshaped).ravel() 
    predictions_cv_scaled = cross_val_predict(pipeline, X_scaled, y_scaled, cv=10)
    predictions_cv=scaler_y.inverse_transform(predictions_cv_scaled.reshape(-1,1)).ravel()
    y = pd.Series(y, index=data.index)
    predictions_cv = pd.Series(predictions_cv,index=y.index)

    # Calculate metrics
    cv_mse = cross_val_score(pipeline, X_scaled, y_scaled, cv=10, scoring='neg_mean_squared_error')

    cv_mae = cross_val_score(pipeline, X_scaled, y_scaled, cv=10, scoring='neg_mean_absolute_error')

    cv_r2scores = cross_val_score(pipeline, X_scaled, y_scaled, cv=10)

    # Convert negative MSE to positive MSE
    cv_mse = -cv_mse
    # Convert negative MSE to positive MSE
    cv_mae= -cv_mae


    mse_cv_score = round(np.mean(cv_mse),2)
    mae_cv_score = round(np.mean(cv_mae),2)
    r2_cv_score=round(np.mean(cv_r2scores),2)
    average_rmse = round((np.sqrt(mse_cv_score)),2)

    st.subheader("Metrics")
    st.write(f"Mean Absolute Error (MAE): {mae_cv_score}")
    st.write(f"Mean Squared Error (MSE): {mse_cv_score}")
    st.write(f"Root Mean Squared Error (RMSE): {average_rmse}")
    st.write(f"R² Score: {r2_cv_score}")



    # Display the prediction
    pipeline.fit(X_scaled, y_scaled)  # Fit on the entire data for prediction
    prediction = pipeline.predict(input_data)
    st.subheader("Prediction")
    st.write(f"Predicted Wind Power (kW): {prediction[0]:.2f}")

    # Plot Actual vs Predicted
    st.subheader("Plots")

    # Create DataFrame for plotting
    df = pd.DataFrame({
        'Actual': y,
        'Predicted': predictions_cv
    }).reset_index()  # Reset index to avoid plotting issues

    # Calculate R^2 value
    r2 = r2_score(y, predictions_cv)

    # Create the plots
    plt.figure(figsize=(14, 7))

    # Plot 1: Regression Plot
    plt.subplot(1, 2, 1)
    sns.regplot(x='Actual', y='Predicted', data=df, ci=None, 
                scatter_kws={"color": "blue", "s": 10, "marker": "*"},  # Star markers
                line_kws={"color": "red", "alpha": 0.7})
    plt.xlabel('Actual Values', color='blue')
    plt.ylabel('Predicted Values', color='blue')
    plt.title(f'Regression Plot: Actual vs Predicted\n$R^2 = {r2:.4f}$', color='blue')

    # Plot 2: Line Plot of Actual vs Predicted
    y_limited = y.head(100).reset_index(drop=True)
    predictions_cv_limited = predictions_cv.head(100).reset_index(drop=True)
    sample_numbers = np.arange(len(y_limited))

    plt.subplot(1, 2, 2)
    plt.plot(sample_numbers, y_limited, color='blue', label='Actual Values', linestyle='-', marker='o')
    plt.plot(sample_numbers, predictions_cv_limited, color='red', label='Predicted Values', linestyle='--', marker='x')
    plt.xlabel('Sample Number')
    plt.ylabel('Values')
    plt.title('Actual vs Predicted LV ActivePower (kW)')
    plt.legend()

    plt.tight_layout()
    st.pyplot(plt)  # Display the plots in Streamlit

elif model_option == "LSTM":
    # Scaling
    scaler_X = MinMaxScaler(feature_range=(0, 1))
    scaler_y = MinMaxScaler(feature_range=(0, 1))
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1))
    y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1))

    # Reshape X data for LSTM
    X_train_scaled = X_train_scaled.reshape(X_train_scaled.shape[0], 1, X_train_scaled.shape[1])
    X_test_scaled = X_test_scaled.reshape(X_test_scaled.shape[0], 1, X_test_scaled.shape[1])

    # Build LSTM model
    model_load_path = 'best_lstm_model_0.92.h5'
   # Load the model
    model = tf.keras.models.load_model(model_load_path)
    predictions_scaled=model.predict(X_test_scaled)

    # Predict and inverse scale the predictions
    y_test_actual = scaler_y.inverse_transform(y_test_scaled)

    predictions = scaler_y.inverse_transform(model.predict(X_test_scaled))

    # Calculate metrics
    mae_lstm = round(mean_absolute_error(y_test_scaled, predictions_scaled), 2)
    mse_lstm = round(mean_squared_error(y_test_scaled, predictions_scaled), 2)
    rmse_lstm = round(np.sqrt(mse_lstm), 2)
    r2_lstm = round(r2_score(y_test_scaled, predictions_scaled), 2)

    st.subheader("Prediction")
    prediction = scaler_y.inverse_transform(model.predict(scaler_X.transform(input_data).reshape(1, 1, -1)))
    st.write(f"Predicted Wind Power (kW): {prediction[0][0]:.2f}")

    st.subheader("Metrics")
    st.write(f"Mean Absolute Error (MAE): {mae_lstm}")
    st.write(f"Mean Squared Error (MSE): {mse_lstm}")
    st.write(f"Root Mean Squared Error (RMSE): {rmse_lstm}")
    st.write(f"R² Score: {r2_lstm}")

    # Plotting Actual vs Predicted for the first 100 samples
    results_df = pd.DataFrame({
        'Actual Values': y_test_actual[:100].flatten(),
        'Predicted Values': predictions[:100].flatten()
    })

    plt.figure(figsize=(14, 7))
    plt.plot(results_df.index, results_df['Actual Values'], label='Actual Values', color='blue', linestyle='-', marker='o')
    plt.plot(results_df.index, results_df['Predicted Values'], label='Predicted Values', color='red', linestyle='--', marker='x')
    plt.title('Actual vs Predicted Values with LSTM model (First 100 Samples)')
    plt.xlabel('Sample Index')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)


