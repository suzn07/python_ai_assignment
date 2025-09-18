# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# df = pd.read_csv('linreg_data.csv')
#
# # df = np.loadtxt("linreg_data.csv", delimiter=",")
#
# df = np.array(df)
#
# x = df[:, 0]
# y = df[:, 1]
#
#
# plt.scatter(x,y,color='red', marker='+')
#
# plt.show()


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('linreg_data.csv',names=['x','y'], skiprows=0)
xgiven = df['x']
ygiven = df['y']
term1xy = xgiven*ygiven
print(term1xy)