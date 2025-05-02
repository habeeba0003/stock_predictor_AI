import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Step 1: Download stock data
ticker = "AAPL"  # You can change this to any stock
data = yf.download(ticker, start="2015-01-01", end="2023-12-31")

# Step 2: Create features and labels
data = data.dropna()
data['Target'] = data['Close'].shift(-1)  # Predict next day's close

# Features
features = ['Open', 'High', 'Low', 'Close', 'Volume']
X = data[features]
y = data['Target']

# Drop last row (it will have NaN target)
X = X[:-1]
y = y[:-1]

# Step 3: Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 4: Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 5: Predict and evaluate
predictions = model.predict(X_test)

print("Mean Squared Error:", mean_squared_error(y_test, predictions))
print("R2 Score:", r2_score(y_test, predictions))

# Step 6: Plot actual vs predicted
plt.figure(figsize=(10, 6))
plt.plot(y_test.values, label="Actual")
plt.plot(predictions, label="Predicted")
plt.title(f"{ticker} Stock Price Prediction (Improved Linear Model)")
plt.xlabel("Samples")
plt.ylabel("Price")
plt.legend()
plt.tight_layout()
plt.show()

