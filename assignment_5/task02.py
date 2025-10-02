import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

df = pd.read_csv("50_Startups.csv", delimiter=",")
print(df.head())

X = df.drop("Profit", axis=1)
y = df["Profit"]


print("variables inside dataset:", df.columns.tolist())

correlation = df.corr(numeric_only=True)
print("Correlation matrix:", correlation)

for col in ["R&D Spend", "Administration", "Marketing Spend"]:
    plt.scatter(df[col], df["Profit"])
    plt.xlabel(col)
    plt.ylabel("Profit")
    plt.title(f"Profit vs {col}")
    plt.show()

ct = ColumnTransformer(
    [("encoder", OneHotEncoder(drop="first"), ["State"])],
    remainder="passthrough"
)
X = ct.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
train_r2 = r2_score(y_train, y_train_pred)

test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
test_r2 = r2_score(y_test, y_test_pred)

print("RMSE:", train_rmse)
print("R²:", train_r2)
print("RMSE:", test_rmse)
print("R²:", test_r2)

# Answers:
# 1) variables inside dataset: ['R&D Spend', 'Administration', 'Marketing Spend', 'State', 'Profit']

# 2)
# R&D With profit is 0.97(strong)
# Administration with profit is 0.2(weak)
# Marketing Spend with profit is 0.74(medium)

# 3)
# According to the provided data, I tried to predict company profit as R&D Spend, Marketing Spend, State

# 4) Plots: Linear trend for R&D Spend and Marketing

# 5) Data split: 80% training, 20% testing

# 6) Model: Multiple Linear Regression

# 7)  Training R²(0.95), Testing R²(0.93–0.95)



