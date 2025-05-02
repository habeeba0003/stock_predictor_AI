import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Step 1: Download stock data
ticker = "AAPL"  # You can change this to any ticker like "TSLA"
data = yf.download(ticker, start="2015-01-01", end="2024-12-31")

# Step 2: Use only the 'Close' column
data = data[['Close']]
data['Prediction'] = data[['Close']].shift(-1)

# Step 3: Prepare the features and labels
X = np.array(data.drop(['Prediction'], axis=1))[:-1]
y = np.array(data['Prediction'])[:-1]

# Step 4: Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 6: Predict the next-day prices
predictions = model.predict(X_test)

# Step 7: Plot results
plt.figure(figsize=(10, 6))
plt.plot(y_test[:50], label="Real Price")
plt.plot(predictions[:50], label="Predicted Price", linestyle='dashed')
plt.title(f"{ticker} Stock Price Prediction (Linear Regression)")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
