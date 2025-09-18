import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

data = pd.read_csv("linreg_data.csv", header=None)


x_col = 0
y_col = 1

Xpd = data[[x_col]].values
ypd = data[y_col].values

plt.scatter(Xpd, ypd, color="red", label="Data")

model = LinearRegression()
model.fit(Xpd, ypd)
y_pred = model.predict(Xpd)

r2 = r2_score(ypd, y_pred)

plt.plot(Xpd, y_pred, color="green", label=f"Regression Line (R²={r2:.2f})")
plt.xlabel(f" X- Axis ")
plt.ylabel(f" Y - Axis ")
plt.legend()
plt.title(f"Scatter plot of X and Y")
plt.show()