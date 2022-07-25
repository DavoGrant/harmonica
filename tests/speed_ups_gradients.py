import jax
import time
import numpy as np

from harmonica.jax import harmonica_transit_quad_ld


# Config.
np.random.seed(123)
jax.random.PRNGKey(12345)
n_ints = 500
n_repeats = 1000

# System params.
t0 = 0.
period = 4.055259
a = 11.55
inc = 87.83 * np.pi / 180.
us = np.array([0.02711153, 0.23806882])
rs = np.array([0.1457, -0.001, 0., 0.001, 0.])
times = np.linspace(-0.19, 0.19, n_ints)
fs = np.empty(times.shape, dtype=np.float64, order='C')

runtimes = []
for i in range(n_repeats):
    start = time.time()
    harmonica_transit_quad_ld(times, t0, period, a, inc, 0., 0.,
                              us[0], us[1], rs[0], rs[1], rs[2], rs[3], rs[4])
    end = time.time()
    runtimes.append(end - start)

print('Total runtime w/ gradients: ', np.median(runtimes))
print('Runtime w/ gradients per data point: ', np.median(runtimes) / n_ints)
