# House Price Prediction using Linear Regression

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------------------------
# 1. Load Dataset
# ---------------------------
data = pd.read_csv("housing.csv")

print("Dataset Loaded Successfully!")
print(data.head())

# ---------------------------
# 2. Handle Categorical Data
# ---------------------------
# Convert categorical columns to numerical using one-hot encoding
data = pd.get_dummies(data, drop_first=True)

# ---------------------------
# 3. Split Features and Target
# ---------------------------
X = data.drop("price", axis=1)
y = data["price"]

# ---------------------------
# 4. Train-Test Split
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------
# 5. Train Linear Regression Model
# ---------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# ---------------------------
# 6. Make Predictions
# ---------------------------
y_pred = model.predict(X_test)

# ---------------------------
# 7. Evaluate Model
# ---------------------------
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance:")
print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("R² Score:", r2)

# ---------------------------
# 8. Visualization
# ---------------------------
plt.scatter(y_test, y_pred)
plt.xlabel("Actual House Prices")
plt.ylabel("Predicted House Prices")
plt.title("Actual vs Predicted House Prices")
plt.show()
