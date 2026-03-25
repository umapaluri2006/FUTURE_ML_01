# -----------------------------
# Import Libraries
# -----------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# -----------------------------
# Load Dataset (or create if not exists)
# -----------------------------
try:
    df = pd.read_csv("sales_data.csv")
except:
    print("Dataset not found. Creating sample dataset...")
    df = pd.DataFrame({
        "Date": pd.date_range(start="2023-01-01", periods=200),
        "Sales": np.random.randint(100, 500, 200)
    })
    df.to_csv("sales_data.csv", index=False)

# -----------------------------
# Data Preprocessing
# -----------------------------
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')
df = df.dropna()

# -----------------------------
# Feature Engineering
# -----------------------------
df['Day'] = df['Date'].dt.day
df['Month'] = df['Date'].dt.month
df['Year'] = df['Date'].dt.year
df['DayOfWeek'] = df['Date'].dt.dayofweek

df['Lag_1'] = df['Sales'].shift(1)
df['Rolling_Mean'] = df['Sales'].rolling(3).mean()

df = df.dropna()

# -----------------------------
# Prepare Data
# -----------------------------
X = df[['Day', 'Month', 'Year', 'DayOfWeek', 'Lag_1', 'Rolling_Mean']]
y = df['Sales']

split = int(len(df) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# -----------------------------
# Train Model
# -----------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# -----------------------------
# Evaluate Model
# -----------------------------
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print("Mean Absolute Error:", mae)

# -----------------------------
# Future Forecast
# -----------------------------
future_dates = pd.date_range(start=df['Date'].max(), periods=30)

future_df = pd.DataFrame({'Date': future_dates})
future_df['Day'] = future_df['Date'].dt.day
future_df['Month'] = future_df['Date'].dt.month
future_df['Year'] = future_df['Date'].dt.year
future_df['DayOfWeek'] = future_df['Date'].dt.dayofweek

future_df['Lag_1'] = df['Sales'].iloc[-1]
future_df['Rolling_Mean'] = df['Rolling_Mean'].iloc[-1]

future_X = future_df[['Day', 'Month', 'Year', 'DayOfWeek', 'Lag_1', 'Rolling_Mean']]
future_df['Predicted_Sales'] = model.predict(future_X)

# =============================
#  VISUALIZATIONS
# =============================

# Actual vs Predicted
plt.figure(figsize=(10,5))
plt.plot(df['Date'], df['Sales'], label='Actual Sales')
plt.plot(df['Date'][split:], y_pred, linestyle='dashed', label='Predicted Sales')
plt.title("Actual vs Predicted Sales")
plt.legend()
plt.grid()
plt.savefig("Actual_vs_Predicted_Sales.png")
plt.show()

# Sales Trend
plt.figure(figsize=(10,5))
plt.plot(df['Date'], df['Sales'])
plt.title("Sales Trend Over Time")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.grid()
plt.savefig("Sales_Trend.png")
plt.show()

# Monthly Sales
monthly_sales = df.groupby(df['Date'].dt.month)['Sales'].mean()

plt.figure(figsize=(8,5))
monthly_sales.plot(kind='bar')
plt.title("Average Monthly Sales")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.savefig("Sales_Comparison.png")
plt.show()

# Sales Distribution
plt.figure(figsize=(8,5))
plt.hist(df['Sales'], bins=20)
plt.title("Sales Distribution")
plt.xlabel("Sales")
plt.ylabel("Frequency")
plt.savefig("Sales_Distribution.png")
plt.show()

# Future Forecast
plt.figure(figsize=(10,5))
plt.plot(future_df['Date'], future_df['Predicted_Sales'])
plt.title("Future Sales Forecast")
plt.xlabel("Date")
plt.ylabel("Predicted Sales")
plt.grid()
plt.savefig("Future_Forecast.png")
plt.show()

# -----------------------------
# Output
# -----------------------------
print("\nFuture Predictions:")
print(future_df.head())