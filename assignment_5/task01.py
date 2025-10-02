import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

data = load_diabetes(as_frame=True)
df = data['frame']

plt.hist(df['target'], 25)
plt.xlabel("target")
plt.show()

sns.heatmap(data=df.corr().round(2), annot=True)
plt.show()

plt.subplot(1,2,1)
plt.scatter(df['bmi'], df['target'])
plt.xlabel("bmi")
plt.ylabel("target")
plt.subplot(1,2,2)
plt.scatter(df['s5'], df['target'])
plt.xlabel("s5")
plt.ylabel("target")
plt.show()

X1 = df[['bmi', 's5']]
y = df['target']
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y, random_state=5, test_size=0.5)

lm1 = LinearRegression()
lm1.fit(X1_train, y1_train)
y1_pred = lm1.predict(X1_test)

rmse1 = np.sqrt(mean_squared_error(y1_test, y1_pred))
r2_1 = r2_score(y1_test, y1_pred)

print(f"Model 1 (bmi and s5): RMSE={rmse1}, R²={r2_1}")

X2 = df[['bmi', 's5', 'bp']]
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y, random_state=5, test_size=0.5)

lm2 = LinearRegression()
lm2.fit(X2_train, y2_train)
y2_pred = lm2.predict(X2_test)

rmse2 = np.sqrt(mean_squared_error(y2_test, y2_pred))
r2_2 = r2_score(y2_test, y2_pred)

print(f"Model 2 (bmi and s5 + bp): RMSE={rmse2}, R²={r2_2}")

X_all = df.drop(columns='target')
X_all_train, X_all_test, y_all_train, y_all_test = train_test_split(X_all, y, random_state=5, test_size=0.5)

lm_all = LinearRegression()
lm_all.fit(X_all_train, y_all_train)
y_all_pred = lm_all.predict(X_all_test)

rmse_all = np.sqrt(mean_squared_error(y_all_test, y_all_pred))
r2_all = r2_score(y_all_test, y_all_pred)

print(f"Model 3 (features): RMSE={rmse_all}, R²={r2_all}")


# a) Which variable would you add next? Why?
# I have chosen the bp(blood pressure) because it is closely related to the diabetes complication.
#
# b) How does adding it affect the model’s performance?
# In the model,
# - Model 1: RMSE(57.0541) and R²(0.45730)
# - Model 2: RMSE(57.48231) and R²(0.4491)
#
# From the output I have observe a small decrease in RMSE and a slight increase in R²,
# which means adding 'bp' improves the model performance a little.
#
# d) Does it help if you add even more variables?
# ------------------------------------------------
# Yes. When we add all even more variables, RMSE decreases further and R² increases more.
