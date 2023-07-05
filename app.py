
# install and import al the important libraries       

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import datetime
from datetime import date, timedelta
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.layers import GRU
from sklearn.preprocessing import MinMaxScaler

app_name = "Stock Market Forecasting App"
st.title(app_name)
st.subheader("This app is created to forecast the selected company")
st.image("https://tradebrains.in/features/wp-content/uploads/2021/07/stock-market-news-trade-brains.jpg")

# Sidebar
st.sidebar.title("Select the Parameters Below")
start_date = st.sidebar.date_input("Start Date", date(2020, 1, 1))
end_date = st.sidebar.date_input("End Date", date(2020, 12, 31))

# Ticker selection
ticker_list = ["AAPL", "GOOG", "GOOGL", "MSFT", "AMZN", "META", "TSLA", "NVDA", "ADBE", "PYPL", "INTC", "NFLX", "PEP"]
ticker = st.sidebar.selectbox("Select the Ticker", ticker_list)

# Fetch Data from Yahoo Finance
data = yf.download(ticker, start=start_date, end=end_date)

# Make date as a column
data.insert(0, "Date", data.index, True)
data.reset_index(drop=True, inplace=True)
st.write("Date from", start_date, "to", end_date)
st.write(data)



# Data visualization
st.header("Data visualization")

# Plot the data using Plotly
fig = px.line(data, x="Date", y=data.columns, title="Closing price of the stock", width=1000, height=800)
st.plotly_chart(fig)

# Select the column to be used in forecasting
columns = st.selectbox('Select the column to be used in forecasting', data.columns[1:])
data = data[["Date", columns]]
st.write("Selected data")
st.write(data)

# Check whether the data is stationary or not
st.header('Is Data stationary')
st.write("**Note**: If the p-value is less than 0.5, then the data is stationary; else, the data is not stationary.")
st.write(adfuller(data[columns])[1]<0.5)


# Decompose the data
st.header('Decomposition')
decompose = seasonal_decompose(data[columns], model='additive', period=12)
st.write(decompose.plot())

# Make the decomposition in Plotly
st.write('## Plotting the Decomposition in Plotly')
st.plotly_chart(px.line(x=data["Date"], y=decompose.trend, title='Trend', labels={"x": "Date", "y": "Price"}).update_traces(line_color="Red"))
st.plotly_chart(px.line(x=data["Date"], y=decompose.seasonal, title='Seasonality', labels={"x": "Date", "y": "Price"}).update_traces(line_color="Green"))
st.plotly_chart(px.line(x=data["Date"], y=decompose.resid, title='Residual', labels={"x": "Date", "y": "Price"}).update_traces(line_dash="dot"))

# Take the input from the user for the three parameters of the SARIMA model and seasonal order
p = st.slider('Select the value of p', 0, 5, 2)
d = st.slider('Select the value of d', 0, 5, 1)
q = st.slider('Select the value of q', 0, 5, 2)
seasonal_order = st.number_input('Select the value of seasonal p', 0, 24, 12)

# Fit the SARIMA model
model = sm.tsa.statespace.SARIMAX(data[columns], order=(p, d, q), seasonal_order=(p, d, q, seasonal_order))
model = model.fit()

# Print the model summary
st.header('SARIMA Model Summary')
st.write(model.summary())
st.write("----")

# Predict the values using SARIMA
st.header('SARIMA Forecasting')
forecast_period = st.number_input('Select the number of days to forecast', 1, 365, 10)
sarima_predictions = model.get_prediction(start=len(data), end=len(data) + forecast_period - 1).predicted_mean

# Add index to the SARIMA predictions dataframe as dates
sarima_predictions.index = pd.date_range(start=end_date, periods=len(sarima_predictions), freq='D')
sarima_predictions = pd.DataFrame(sarima_predictions)
sarima_predictions.insert(0, 'Date', sarima_predictions.index)
sarima_predictions.reset_index(drop=True, inplace=True)
st.write("## SARIMA Predictions")
st.write(sarima_predictions)
st.write("## Actual Data")
st.write(data)
st.write("---")

# Plot the SARIMA predictions
fig_sarima = go.Figure()
fig_sarima.add_trace(go.Scatter(x=data["Date"], y=data[columns], name="Actual Data"))
fig_sarima.add_trace(go.Scatter(x=sarima_predictions["Date"], y=sarima_predictions["predicted_mean"], name="SARIMA Predictions", mode="lines", line=dict(color="Red")))
fig_sarima.update_layout(title_text="Actual Data vs SARIMA Predictions", xaxis_title="Date", yaxis_title="Price", width=1000, height=400)
st.plotly_chart(fig_sarima)

# LSTM Model
st.header('LSTM Model')
st.write('**Note**: LSTM Model is trained and predicted on the selected column of the data.')

# Prepare the data for LSTM model
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data[columns].values.reshape(-1, 1))

# Split data into train and test sets
train_size = int(len(scaled_data) * 0.8)
test_size = len(scaled_data) - train_size
train_data, test_data = scaled_data[:train_size, :], scaled_data[train_size:, :]

# Create the LSTM model
lstm_model = Sequential()
lstm_model.add(LSTM(50, return_sequences=True, input_shape=(1, 1)))
lstm_model.add(LSTM(50))
lstm_model.add(Dense(1))

# Compile and train the LSTM model
lstm_model.compile(loss='mean_squared_error', optimizer='adam')
lstm_model.fit(train_data.reshape(-1, 1, 1), train_data.reshape(-1, 1), epochs=10, batch_size=1, verbose=2)

# lstm predictions
lstm_predictions = lstm_model.predict(test_data.reshape(-1, 1, 1))
lstm_predictions = scaler.inverse_transform(lstm_predictions)

# Convert the index to datetime
test_dates = pd.to_datetime(data["Date"].values[train_size:])

# Create a new DataFrame with date and predictions
lstm_predictions_df = pd.DataFrame({"Date": test_dates, "Predicted_Price": lstm_predictions.flatten()})

# Display the LSTM predictions
st.write("## LSTM Predictions")
st.write(lstm_predictions_df)
st.write("---")

# Plot the LSTM predictions
fig_lstm = go.Figure()
fig_lstm.add_trace(go.Scatter(x=data["Date"], y=data[columns], name="Actual Data"))
fig_lstm.add_trace(go.Scatter(x=lstm_predictions_df["Date"], y=lstm_predictions_df["Predicted_Price"], name="LSTM Predictions", mode="lines", line=dict(color="Red")))
fig_lstm.update_layout(title_text="Actual Data vs LSTM Predictions", xaxis_title="Date", yaxis_title="Price", width=1000, height=400)
st.plotly_chart(fig_lstm)




#...GRU MODEL
st.header('GRU Model')
st.write('**Note**: GRU Model is trained and predicted on the selected column of the data.')

# Create the GRU model
gru_model = Sequential()
gru_model.add(GRU(50, return_sequences=True, input_shape=(1, 1)))
gru_model.add(GRU(50))
gru_model.add(Dense(1))

# Compile and train the GRU model
gru_model.compile(loss='mean_squared_error', optimizer='adam')
gru_model.fit(train_data.reshape(-1, 1, 1), train_data.reshape(-1, 1), epochs=10, batch_size=1, verbose=2)

# Make predictions using the GRU model
# Make predictions using the GRU model
gru_predictions = gru_model.predict(test_data.reshape(-1, 1, 1))
gru_predictions = scaler.inverse_transform(gru_predictions)

# Convert the index to datetime
test_dates = pd.to_datetime(data["Date"].values[train_size:])

# Create a new DataFrame with date and predictions
gru_predictions_df = pd.DataFrame({"Date": test_dates, "Predicted_Price": gru_predictions.flatten()})

# Display the GRU predictions
st.write("## GRU Predictions")
st.write(gru_predictions_df)
st.write("---")

# Plot the GRU predictions
fig_gru = go.Figure()
fig_gru.add_trace(go.Scatter(x=data["Date"], y=data[columns], name="Actual Data"))
fig_gru.add_trace(go.Scatter(x=gru_predictions_df["Date"], y=gru_predictions_df["Predicted_Price"], name="GRU Predictions", mode="lines", line=dict(color="Green")))
fig_gru.update_layout(title_text="Actual Data vs GRU Predictions", xaxis_title="Date", yaxis_title="Price", width=1000, height=400)
st.plotly_chart(fig_gru)