import numpy as np
import matplotlib.pyplot as plt

values = [ 500, 1000, 2000, 5000, 10000, 15000, 20000, 50000, 100000 ]

for v in values:
    dice1 = np.random.randint(1, 7, v)

    dice2 = np.random.randint(1, 7, v)

    sum_of_dices = dice1 + dice2

    h, h2 = np.histogram(sum_of_dices, range(2, 14))

    plt.bar(h2[:-1], h/v)
    plt.xlabel('sum of dices')
    plt.ylabel('probability')
    plt.title(f"n={v}")

    plt.show()

