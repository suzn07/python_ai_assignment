import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np


data = pd.read_csv("linreg_data.csv", header=None)
X = data[[0]].values
y = data[1].values


plt.scatter(X, y, color="red", label="Data")


model = LinearRegression()
model.fit(X, y)


y_pred = model.predict(X)


yhat = model.intercept_ + model.coef_[0] * X


RSS = np.sum((y - y_pred) ** 2)
MSE = mean_squared_error(y, y_pred)
RMSE = np.sqrt(MSE)
MAE = mean_absolute_error(y, y_pred)
R2 = r2_score(y, y_pred)


plt.plot(X, y_pred, color="green", label=f"Regression Line (R²={R2:.2f})")
plt.xlabel("X - Axis")
plt.ylabel("Y - Axis")
plt.legend()
plt.title("Scatter plot of X and Y")
plt.show()

print("Values of Yhat:", y_pred)
print("Residual Sum of Squares (RSS):", RSS)
print("Mean Squared Error (MSE):", MSE)
print("Root Mean Squared Error (RMSE):", RMSE)
print("Mean Absolute Error (MAE):", MAE)
print("R-squared (R²):", R2)