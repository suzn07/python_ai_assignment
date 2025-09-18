import numpy as np
from matplotlib import pyplot as plt

from sklearn.linear_model import LinearRegression

my_data = np.genfromtxt("linreg_data.csv", delimiter= ",")

xp = my_data[:,0]

yp = my_data[:,1]

xp = xp.reshape(-1,1)

yp = yp.reshape(-1,1)

regr = LinearRegression()

regr.fit(xp, yp)

print(regr.coef_,regr.intercept_)
xval = np.linspace(-1,1,100).reshape(-1,1)
yval = regr.predict(xval)
plt.plot(xval,yval)
plt.scatter(xp,yp)
plt.show()

from sklearn import metrics

yhat = regr.predict(xp)

print('Mean Absolute Error:', metrics.mean_absolute_error(yp, yhat))

print('Mean Squared Error:', metrics.mean_squared_error(yp, yhat))

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(yp, yhat)))

print('R2 value:',metrics.r2_score(yp, yhat))