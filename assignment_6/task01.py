import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

df = pd.read_csv('bank.csv', delimiter=';')
print(df.head())
print(df.dtypes)
print(df.shape)
print(list(df.columns))

cols = ['y', 'job', 'marital', 'default', 'housing', 'poutcome']
df2 = df[cols].copy()

df3 = pd.get_dummies(df2, columns=['job', 'marital', 'default', 'housing', 'poutcome'], drop_first=True)

if df3['y'].dtype == 'object':
    df3['y'] = df3['y'].map({'no': 0, 'yes': 1})

corr = df3.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', square=False)
plt.title("Correlation Matrix")
plt.tight_layout()
plt.show()

corr_with_y = corr['y'].drop('y').abs().sort_values(ascending=False)
print("\nTop 10 Absolute Correlations with Target 'y':")
print(corr_with_y.head(10))

y = df3['y']
X = df3.drop('y', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

logistic_reg = LogisticRegression(max_iter=1000)
logistic_reg.fit(X_train, y_train)
y_pred_lr = logistic_reg.predict(X_test)

confusion_matrix_logistic_regression = confusion_matrix(y_test, y_pred_lr)
accuracy_logistic_regression = accuracy_score(y_test, y_pred_lr)

print("\nLogistic Regression:")
print("Confusion Matrix:\n", confusion_matrix_logistic_regression)
print("Accuracy:", accuracy_logistic_regression)
metrics.ConfusionMatrixDisplay.from_estimator(logistic_reg, X_test, y_test)
plt.show()
print(classification_report(y_test, y_pred_lr))

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
y_predict_knn = knn.predict(X_test)

cm_knn = confusion_matrix(y_test, y_predict_knn)
acc_knn = accuracy_score(y_test, y_predict_knn)

print("\nK-Nearest Neighbors (k=3):")
print("Confusion Matrix:\n", cm_knn)
print("Accuracy:", acc_knn)
metrics.ConfusionMatrixDisplay.from_estimator(knn, X_test, y_test)
plt.show()
print(classification_report(y_test, y_predict_knn))

print("\nModel Comparison:")
print(f"Logistic Regression Accuracy: {accuracy_logistic_regression}")
print(f"KNN Accuracy: {acc_knn}")


# 4)
# the heat map of correlation coefficients for all variables in df3 is:-
# poutcome_success    0.283481
# poutcome_unknown    0.162038
# housing_yes         0.104683
# job_retired         0.086675
# job_blue-collar     0.068147
# marital_married     0.064643
# poutcome_other      0.051908
# job_student         0.047809
# marital_single      0.045815
# job_management      0.032634
#
# here, the most correlation is less than the 0.3. so, the correalation is generally low here.

# 9)
# Confusion Matrix:
#  [[974  32]
#  [110  15]]
# Accuracy: 0.8744473
#
# 10)
#
# The Logistic Regression model performed slightly better than the K-Nearest Neighbors model,
# with an accuracy of about 89.8% means 0.898320(Logistic Regression) compared to 87.4% means 0.874473(KNN).
# Logistic Regression was more reliable and consistent in its predictions, while KNN struggled a bit more to identify customers who said yes.
# Both models had difficulty predicting the minority class due to the data imbalance,
# but overall, Logistic Regression gave clearer and more accurate results for this task.

