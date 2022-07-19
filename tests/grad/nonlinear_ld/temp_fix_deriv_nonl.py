import numpy as np

from harmonica import bindings


us = np.array([0.1, 0.2, 0.3, 0.4])
rs = np.array([0.1, -0.003, 0.003, 0.001, 0.001, 0.001, 0.001])
ds = np.array([0.95])
nus = np.array([0.1])
fs = np.empty(ds.shape, dtype=np.float64, order='C')
fs_grad = np.empty(ds.shape + (6 + 4 + 7,), dtype=np.float64, order='C')
bindings.temp_light_curve(1, us, rs, ds, nus,
                          fs, fs_grad, 50, 500)
print(fs_grad[0, -7])
print(fs_grad[0, -6])
print(fs_grad[0, -5])
print(fs_grad[0, -4])
print(fs_grad[0, -3])
print(fs_grad[0, -2])
print(fs_grad[0, -1])

"""

-0.061994951405994744
-0.061994951405994744

-0.03713111912633512
-0.037131119126334926

-0.004968043394569293
-0.0049680433945692905

-0.0012377294375696241
-0.001237729437569849

-0.0018062816657323467
-0.001806281665732505

0.003750825135750732
0.0037508251357505308

0.00045618382548980247
0.0004561838254899683

"""