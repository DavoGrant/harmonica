import time
import numpy as np

from harmonica import bindings


# Config.
np.random.seed(123)
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
    bindings.light_curve(t0, period, a, inc, 0., 0.,
                         0, us, rs, times, fs, 20, 50)
    end = time.time()
    runtimes.append(end - start)

print('Total runtime: ', np.median(runtimes))
print('Runtime per data point: ', np.median(runtimes) / n_ints)
