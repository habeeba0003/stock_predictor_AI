import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd

# Step 1: Get user input
ticker_symbol = input("Enter the stock symbol (e.g., AAPL, MSFT): ").upper()
start_date = input("Enter start date (YYYY-MM-DD): ")
end_date = input("Enter end date (YYYY-MM-DD): ")

import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np

# 1. Fetch historical data
ticker = "AAPL"
data = yf.download(ticker, start="2015-01-01", end="2024-12-31")
data = data[['Close']]
data = data.dropna()

# 2. Create features and target
data['Prediction'] = data[['Close']].shift(-30)  # Predict 30 days into the future
X = np.array(data.drop(['Prediction'], axis=1))[:-30]
y = np.array(data['Prediction'])[:-30]

# 3. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Make future predictions
future = model.predict(np.array(data.drop(['Prediction'], axis=1))[-30:])
print("Future 30-Day Close Price Prediction:\n", future)

# 6. Plot the original and predicted data
plt.figure(figsize=(14,7))
plt.plot(data['Close'], label='Historical Close Price')
plt.plot(range(len(data)-30, len(data)), future, label='Predicted Next 30 Days', linestyle='--')
plt.title(f'{ticker} Stock Price Prediction')
plt.xlabel('Days')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()

