import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# -------------------------------
# Create Historical Sales Data
# -------------------------------
data = {
    "Month":[1,2,3,4,5,6,7,8,9,10,11,12],
    "Sales":[200,220,250,270,300,320,310,330,350,370,390,420]
}

df = pd.DataFrame(data)

print("Historical Sales Data:\n")
print(df)

# -------------------------------
# Graph 1 – Sales Trend (Line Chart)
# -------------------------------
plt.figure()
plt.plot(df["Month"], df["Sales"], marker='o')
plt.title("Monthly Sales Trend")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.show()

# -------------------------------
# Graph 2 – Sales Distribution (Histogram)
# -------------------------------
plt.figure()
plt.hist(df["Sales"], bins=10)
plt.title("Sales Distribution")
plt.xlabel("Sales")
plt.ylabel("Frequency")
plt.show()

# -------------------------------
# Graph 3 – Sales Comparison (Bar Chart)
# -------------------------------
plt.figure()
plt.bar(df["Month"], df["Sales"])
plt.title("Monthly Sales Comparison")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.show()

# -------------------------------
# Train Forecast Model
# -------------------------------
X = df[["Month"]]
y = df["Sales"]

model = LinearRegression()
model.fit(X, y)

# -------------------------------
# Model Evaluation
# -------------------------------
y_pred = model.predict(X)

mae = mean_absolute_error(y, y_pred)
mse = mean_squared_error(y, y_pred)

print("\nModel Evaluation")
print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)

# -------------------------------
# Graph 4 – Actual vs Predicted Sales
# -------------------------------
plt.figure()
plt.scatter(df["Month"], df["Sales"], label="Actual Sales")
plt.plot(df["Month"], y_pred, color="red", label="Predicted Sales")
plt.title("Actual vs Predicted Sales")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.legend()
plt.show()

# -------------------------------
# Predict Future Sales
# -------------------------------
future_months = pd.DataFrame({"Month":[13,14,15,16,17]})
predictions = model.predict(future_months)

forecast = future_months.copy()
forecast["Predicted Sales"] = predictions

print("\nFuture Sales Forecast:\n")
print(forecast)

# -------------------------------
# Graph 5 – Future Forecast
# -------------------------------
plt.figure()

plt.plot(df["Month"], df["Sales"], label="Actual Sales")

plt.plot(future_months["Month"],
         predictions,
         linestyle="dashed",
         marker="o",
         label="Forecast Sales")

plt.title("Sales Forecast")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.legend()
plt.show()