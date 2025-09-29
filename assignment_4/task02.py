import numpy as np
import matplotlib.pyplot as plt

valuesOfN = [500, 1000, 2000, 5000, 10000, 15000, 20000, 50000, 100000]

for n in valuesOfN:
    firstDice = np.random.randint(1, 7, n)
    secondDice = np.random.randint(1, 7, n)

    sumOfDices = firstDice + secondDice

    h,h2 = np.histogram(sumOfDices, range(2, 14))

    plt.bar(h2[:-1], h/n)

    plt.title(f"n={n}")

    plt.show()