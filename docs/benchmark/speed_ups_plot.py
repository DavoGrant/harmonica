import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


data = pd.DataFrame(
    [['nan',
      'Undo pre-compute', 9.21e-06, 2.61e-05],
     ['f14b12b2d197e7df078dc29edd468412e3103367',
      'Start benchmarking', 9.15e-06, 2.55e-05],
     ['a8d295310ea3d97b6e998d88da3cc15b4f2001bc',
      'Class inheritance', 9.08e-06, 2.54e-05],
     ['d367e5cc994e2d64786cda4c19e384b192984c03',
      'Selective pass by const reference', 8.76e-06, 2.35e-05],
     ['d367e5cc994e2d64786cda4c19e384b192984c03',
      'Remove grad branching for ld law', 8.76e-06, 2.31e-05],
     ['825a7f78033f5e67f9f6297e29659b1afe1f616e',
      'Compile flag -ffast-math can get away with.', 3.70e-06, 1.25e-05],],
    columns=['commit', 'note', 'speed', 'speed_grad'])

fig, ax1 = plt.subplots(1, 1, figsize=(7, 5))
x = np.arange(len(data))
width = 0.3
ax1.bar(x - width/2, data['speed'] * 1.e3 * 1.e3,
        width=width, color='#bc5090', label='Model')
ax1.bar(x + width/2, data['speed_grad'] * 1.e3 * 1.e3,
        width=width, color='#58508d', label='Model + gradients')

ax1.set_xticks(x, data['note'], rotation=90, fontsize=2)
ax1.set_xlabel('Commit history')
ax1.set_ylabel('Runtime per 1000 point light curve / ms')
ax1.legend(loc='upper right')

plt.tight_layout()
plt.show()
