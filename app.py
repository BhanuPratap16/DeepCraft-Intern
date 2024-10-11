import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import math

# Title for the app
st.title("Predicting Stock Prices")
st.set_option('deprecation.showPyplotGlobalUse', False)
# Load the dataset
df = pd.read_csv("stock_price.csv", index_col='Date', parse_dates=True)
df = df.iloc[::-1]  # Reversing the data so that the most recent data is at the end

# Plotting the actual stock prices
st.subheader("Close Stock Price Graph Over Time")  # Title for the close data plot
plt.plot(df['Close'])
plt.title('Stock Prices')
st.pyplot()  # Use st.pyplot() instead of plt.show()

# Add Streamlit sliders for n_input and LSTM neurons
n_input = st.slider("Select n_input (days for forecasting)", 20, 100, 50)
neurons_1 = st.slider("Select number of neurons for first LSTM layer", 64, 120, 100)
neurons_2 = st.slider("Select number of neurons for second LSTM layer", 64, 120, 100)
neurons_3 = st.slider("Select number of neurons for third LSTM layer", 64, 120, 100)

# Prepare the data for forecasting
df1 = df.iloc[:-n_input, 0]  # Data for training
df2 = df.iloc[-n_input:, 0]  # Data for testing

# Normalize the dataset using MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
df1 = scaler.fit_transform(np.array(df1).reshape(-1, 1))

# Split the dataset into training and testing sets
training_size = int(len(df1) * 0.7)  # 70% for training
test_size = len(df1) - training_size
train_data, test_data = df1[0:training_size], df1[training_size:len(df1)]

# Function to create a dataset matrix suitable for LSTM
def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - time_step - 1):
        a = dataset[i:(i + time_step)]
        dataX.append(a)
        dataY.append(dataset[i + time_step])
    return np.array(dataX), np.array(dataY)

# Reshape the data for LSTM input
time_step = n_input
X_train, y_train = create_dataset(train_data, time_step)
X_test, ytest = create_dataset(test_data, time_step)

# Reshape input to be [samples, time steps, features] as required for LSTM
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Build the Stacked LSTM model
model = Sequential()
model.add(LSTM(neurons_1, return_sequences=True, input_shape=(n_input, 1)))  # First LSTM layer
model.add(LSTM(neurons_2, return_sequences=True))  # Second LSTM layer
model.add(LSTM(neurons_3))  # Third LSTM layer
model.add(Dense(1))  # Output layer
model.compile(loss='mean_squared_error', optimizer='adam')  # Compile the model

# Train the model
model.fit(X_train, y_train, validation_data=(X_test, ytest), epochs=20, batch_size=200, verbose=1)

# Predictions for both training and test data
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Transform predictions back to original form
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)

# Calculate RMSE for both training and test data
train_rmse = math.sqrt(mean_squared_error(y_train, train_predict))
test_rmse = math.sqrt(mean_squared_error(ytest, test_predict))

# Shift train and test predictions for plotting
look_back = n_input
trainPredictPlot = np.empty_like(df1)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(train_predict) + look_back, :] = train_predict

testPredictPlot = np.empty_like(df1)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(train_predict) + (look_back * 2) + 1:len(df1) - 1, :] = test_predict

# Plotting the training and test predictions
st.subheader("Training and Test Data Predictions")  # Title for the plot
plt.figure(figsize=(10, 6))
df12 = pd.DataFrame(df1, index=df.iloc[:-n_input].index)
plt.plot(df12.index, scaler.inverse_transform(df1), label='Actual Data')
plt.plot(df12.index, trainPredictPlot, label='Train Predictions')
plt.plot(df12.index, testPredictPlot, label='Test Predictions')

# Add labels and legend
plt.xlabel('Date' if df12.index.dtype == 'datetime64[ns]' else 'Index')
plt.ylabel('Stock Prices')
plt.title('Stock Price Forecasting')
plt.legend()
st.pyplot()  # Show the plot in Streamlit

# Generating predictions for the next n_input days
st.subheader("Stock Price Forecast for the Next n_input Days")  # Title for the forecast plot
test_predictions = []
first_eval_batch = df1[-n_input:]  # Last n_input days of training data
current_batch = first_eval_batch.reshape((1, n_input, 1))

for i in range(len(df2)):
    current_pred = model.predict(current_batch)[0]
    test_predictions.append(current_pred)
    current_batch = np.append(current_batch[:, 1:, :], [[current_pred]], axis=1)

# Inverse transform the predictions
final_pred = scaler.inverse_transform(test_predictions)

# Plot the final predictions
plt.figure(figsize=(12, 6))
plt.plot(df['Close'], label='Actual Data')
plt.plot(pd.DataFrame(final_pred, index=df.iloc[-n_input:].index), label='Prediction', linewidth=3.5)
plt.xlabel('Date' if df.index.dtype == 'datetime64[ns]' else 'Index')
plt.ylabel('Stock Prices')
plt.title('Stock Price Forecasting')

plt.legend()
st.pyplot()  # Show the plot in Streamlit

# Calculate RMSE for the final predictions
actual_values = df['Close'].iloc[-n_input:].values
predicted_values = final_pred.flatten()

rmse = math.sqrt(mean_squared_error(actual_values, predicted_values))

# Display results
st.write(f'Training RMSE: {train_rmse:.2f}')
st.write(f'Test RMSE: {test_rmse:.2f}')
st.write(f'Final RMSE: {rmse:.2f}')
