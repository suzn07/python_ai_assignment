import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import resample

probabilities_value = {
    2: 1/36,  3: 2/36,  4: 3/36,  5: 4/36, 6: 5/36,  7: 6/36,  8: 5/36,  9: 4/36, 10: 3/36, 11: 2/36, 12: 1/36
}

values = [ 500, 1000, 2000, 5000, 10000, 15000, 20000, 50000, 100000 ]

for v in values:
    dices = np.arange(1,7)
    dice1 = resample(dices, replace=True, n_samples=v, random_state=None)
    dice2 = resample(dices, replace=True, n_samples=v, random_state=None)
    sum_of_dices = dice1 + dice2

    counts = np.array([np.sum(sum_of_dices==i) for i in range(2,13)])
    simulated_probabilities = counts /v

    plt.bar(range(2, 13), simulated_probabilities, width=0.4, alpha=0.7)
    plt.plot(list(probabilities_value.keys()), list(probabilities_value.values()), "yo-")
    plt.title(f"Two Dice Sum Distribution (n={v})")
    plt.show()

