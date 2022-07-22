import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


data = pd.DataFrame(
    [['f14b12b2d197e7df078dc29edd468412e3103367', 'Start benchmarking', 9.15e-06, 2.55e-05],
     ['a8d295310ea3d97b6e998d88da3cc15b4f2001bc', 'Class inheritance', 9.08e-06, 2.54e-05],],
    columns=['commit', 'note', 'speed', 'speed_wgrad'])

# todo final no final.
fig, ax1 = plt.subplots(1, 1, figsize=(7, 5))
x = np.arange(len(data))
width = 0.3
ax1.bar(x - width/2, data['speed'] * 1.e2 * 1.e3,
        width=width, color='#bc5090', label='Model')
ax1.bar(x + width/2, data['speed_wgrad'] * 1.e2 * 1.e3,
        width=width, color='#58508d', label='Model + gradients')

ax1.set_xticks(x, data['note'])
ax1.set_xlabel('Commit history')
ax1.set_ylabel('Runtime per 100 point light curve / ms')
ax1.legend(loc='upper right')

plt.tight_layout()
plt.show()
