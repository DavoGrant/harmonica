import numpy as np

from harmonica import bindings


us = np.array([0.1, 0.2, 0.3, 0.4])
rs = np.array([0.1, -0.003, 0.003])
ds = np.array([0.85])
nus = np.array([0.1])
fs = np.empty(ds.shape, dtype=np.float64, order='C')
fs_grad = np.empty(ds.shape + (6 + 4 + 3,), dtype=np.float64, order='C')
bindings.temp_light_curve(1, us, rs, ds, nus,
                          fs, fs_grad, 50, 500)
print(fs_grad[0, 0])


"""

-9.526865777045171e-05
-9.526865777043954e-05

"""
