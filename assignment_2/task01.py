import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0,7,10)

y1 = 2*x + 1
y2 = 2*x + 2
y3 = 2*x + 3

plt.plot(x,y1, 'r-', label= 'y= 2x + 1')
plt.plot(x,y2, 'b-', label = 'y= 2x + 2')
plt.plot(x,y3, 'g-', label = 'y= 2x + 3')

plt.xlabel('x-axis')
plt.ylabel('y-axis')

plt.title('Graph for  y=2x+1,  y=2x+2,  y=2x+3')
plt.legend()
plt.grid(True)
plt.show()