import numpy as np

def quadratic_regression(x, y):
    n = len(x)

    Sx = np.sum(x)
    Sx2  = np.sum(x**2)
    Sx3 = np.sum(x**3)
    Sx4 = np.sum(x**4)
    Sy = np.sum(y)
    Sxy = np.sum(x*y)
    Sx2y = np.sum((x**2)*y)

    D = (Sx2 * (Sx4) - (Sx3)**2)

    a = (Sx2y * Sx2 - Sxy * Sx3) / D
    b = (Sxy * Sx4 - Sx2y * Sx3) / D
    c = (Sy - b * Sx - a * Sx2) / n

    return a, b, c

x = np.array([1, 2, 3, 4, 5], dtype=float)
y = np.array([2.2, 4.8, 7.9, 11.3, 17.2], dtype=float)

a, b, c = quadratic_regression(x, y)

print(f"Quadratic model: y = {a:.4f}x² + {b:.4f}x + {c:.4f}")
