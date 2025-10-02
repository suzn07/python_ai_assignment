# confusion matrix

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split
import seaborn as sns

df = pd.read_csv("exams.csv", skiprows=0, delimiter=",")
print(df)

X = df.iloc[:, 0:2]
y = df.iloc[:, -1]


admit_yes = df.loc[y == 1]
admit_no = df.loc[y == 0]

plt.scatter(admit_no.iloc[:,0], admit_no.iloc[:,1], label="admit no")
plt.scatter(admit_yes.iloc[:,0],admit_yes.iloc[:,1], label="admit yes")
plt.xlabel("exam1")
plt.ylabel("exam2")
plt.legend()
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)
print(X_train.shape)

model = LogisticRegression()
model.fit(X_train, y_train)
Y_pred = model.predict(X_test)
metrics.ConfusionMatrixDisplay.from_estimator(model, X_test, Y_pred)
plt.show()
cnf_metrix = metrics.confusion_matrix(y_test,Y_pred)
print(cnf_metrix)

print("Accuracy:",metrics.accuracy_score(y_test,Y_pred))
print("Precision:",metrics.precision_score(y_test,Y_pred))
print("Recall:",metrics.recall_score(y_test,Y_pred))


