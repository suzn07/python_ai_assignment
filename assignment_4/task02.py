import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv('weight-height.csv')

X = df[["Height"]].values
y = df["Weight"].values

plt.scatter(X,y,color='blue')
plt.xlabel('Height')
plt.ylabel('Weight')
plt.title('Scatter Plot of height and weight')
plt.show()

model = LinearRegression()

model.fit(X, y)

y_value_predict = model.predict(X)

plt.scatter(X, y,color='blue')
plt.plot(X, y_value_predict, color='red', label='Linear Regression')
plt.xlabel('Height')
plt.ylabel('Weight')
plt.title('Linear Regression of Height and Weight')
plt.show()

RMSE = np.sqrt(mean_squared_error(y, y_value_predict))
r2 = r2_score(y, y_value_predict)

print(f"RMSE = {RMSE}, r2 ={r2}")
