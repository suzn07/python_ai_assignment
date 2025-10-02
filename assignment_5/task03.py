import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import r2_score

df = pd.read_csv("Auto.csv", delimiter=",")

print(df.head())
df['horsepower'] = df['horsepower'].replace('?', np.nan).astype(float)
df.dropna(inplace=True)

X = df.drop(['mpg', 'name', 'origin'], axis=1)
y = df['mpg']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

alphas = np.logspace(-3, 3, 50)
ridge_scores = []
lasso_scores = []

for alpha in alphas:
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train, y_train)
    ridge_scores.append(r2_score(y_test, ridge.predict(X_test)))

    lasso = Lasso(alpha=alpha, max_iter=10000)
    lasso.fit(X_train, y_train)
    lasso_scores.append(r2_score(y_test, lasso.predict(X_test)))

plt.figure(figsize=(8, 6))
plt.semilogx(alphas, ridge_scores, label="Ridge R²", marker='o')
plt.semilogx(alphas, lasso_scores, label="Lasso R²", marker='s')
plt.xlabel("Alpha (log scale)")
plt.ylabel("R² Score")
plt.title("Ridge vs Lasso Regression Performance")
plt.legend()
plt.grid(True)
plt.show()

ridge_alpha_value = alphas[np.argmax(ridge_scores)]
lasso_alpha_value = alphas[np.argmax(lasso_scores)]

print("Ridge alpha value:", ridge_alpha_value, " R² =", max(ridge_scores))
print("Lasso alpha value:", ridge_alpha_value, " R² =", max(lasso_scores))


# 1)
#     mpg  cylinders  displacement  horsepower  weight  acceleration  model year  origin                       name
# 0  18.0          8         307.0       130.0  3504.0         12.0          70       1  chevrolet chevelle malibu
# 1  15.0          8         350.0       165.0  3693.0         11.5          70       1          buick skylark 320
# 2  18.0          8         318.0       150.0  3436.0         11.0          70       1         plymouth satellite
# 3  16.0          8         304.0       150.0  3433.0         12.0          70       1              amc rebel sst
# 4  17.0          8         302.0       140.0  3449.0         10.5          70       1                ford torino

# 2)
# X = df.drop(['mpg', 'name', 'origin'], axis=1)
# y = df['mpg']

# 3)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4) for this i have use for loop in the code for alpha in alphas.

# 5)
# Ridge alpha value: 0.001 with R² = 0.7942348920666245
# Lasso alpha value: 0.001 with R² = 0.7941834683177982
# 5)
# R² = 0.7942348920666245
# R² = 0.7941834683177982
