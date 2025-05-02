import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import datetime

# Load sentiment
sentiment_df = pd.read_csv("sentiment.csv")
avg_sentiment = sentiment_df['Sentiment'].values[0]

# Download historical stock data
ticker = "AAPL"
start = "2015-01-01"
end = datetime.date.today().strftime('%Y-%m-%d')
data = yf.download(ticker, start=start, end=end)

# Add sentiment column as a new feature
data['Sentiment'] = avg_sentiment

# Use only Close price and Sentiment for prediction
df = data[['Close', 'Sentiment']].copy()

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df)

# Create training dataset
sequence_length = 60
X, y = [], []

for i in range(sequence_length, len(scaled_data)):
    X.append(scaled_data[i-sequence_length:i])
    y.append(scaled_data[i][0])  # Predicting only the Close price

X, y = np.array(X), np.array(y)

# Build the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 2)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=1))  # Predict the closing price

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X, y, epochs=10, batch_size=32)

# Predict future
test_data = scaled_data[-sequence_length:]
test_input = np.expand_dims(test_data, axis=0)
predicted_price = model.predict(test_input)
predicted_price = scaler.inverse_transform([[predicted_price[0][0], avg_sentiment]])[0][0]

print(f"Predicted next closing price for {ticker}: ${predicted_price:.2f}")

# Plot predictions vs actual
predicted_prices = model.predict(X)
predicted_prices_unscaled = scaler.inverse_transform(
    np.hstack((predicted_prices, np.full((predicted_prices.shape[0], 1), avg_sentiment)))
)[:, 0]

actual_prices_unscaled = scaler.inverse_transform(
    np.hstack((y.reshape(-1, 1), np.full((len(y), 1), avg_sentiment)))
)[:, 0]

plt.plot(actual_prices_unscaled, color='blue', label='Actual Stock Price')
plt.plot(predicted_prices_unscaled, color='red', label='Predicted Stock Price')
plt.title(f'{ticker} Stock Price Prediction with Sentiment')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()
