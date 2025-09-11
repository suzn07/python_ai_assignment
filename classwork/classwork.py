import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0,10,10)
# y = a +bx  is the equation
y1 = 1 + 2 * x
y2 = 1 + 3 * x
y3 = 1 + 4 * x

plt.plot(x,y1, 'r-', label = '1+2x' )
plt.plot(x,y2, 'b-', label = '1+3x' )
plt.plot(x,y3, 'g-', label = '1+4x' )
plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.legend()
plt.grid(True)
plt.show()