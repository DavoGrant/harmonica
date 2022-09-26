import numpy as np
import matplotlib.pyplot as plt


for k in [1, 3, 5, 7]:
    cs = []
    for i in range(10000):
        a = np.random.normal(loc=0, scale=1, size=k)
        b = a**2
        c = np.sum(b)
        cs.append(c)

    print(k, np.mean(cs))
    plt.hist(cs, bins=np.linspace(0, 20, 30), histtype='step', lw=2)
plt.show()
