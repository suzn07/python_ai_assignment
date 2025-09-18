import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('linreg_data.csv',names=['x','y'], skiprows=0)
xgiven = df['x']
ygiven = df['y']
term1xy = xgiven*ygiven
print(term1xy)