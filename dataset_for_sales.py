import pandas as pd
import numpy as np

# Create dataset
data = {
    "Date": pd.date_range(start="2023-01-01", periods=200, freq='D'),
    "Sales": np.random.randint(100, 500, size=200)
}

df = pd.DataFrame(data)

# Save as CSV
df.to_csv("sales_data.csv", index=False)

print("Dataset created successfully!")