import time
import numpy as np

from harmonica import HarmonicaTransit


# Config.
np.random.seed(123)
n_ints = 500
n_repeats = 1000

# todo: adding const ref everywhere possible
# todo: adding const to variables where possible too
# todo: adding const as method labels where possible to
# todo: less precompute slow down

# System params.
t0 = 0.
period = 4.055259
a = 11.55
inc = 87.83 * np.pi / 180.
us = np.array([0.02711153, 0.23806882])
rs = np.array([0.1457, -0.001, 0., 0.001, 0.])
times = np.linspace(-0.19, 0.19, n_ints)
fs = np.empty(times.shape, dtype=np.float64, order='C')
ht = HarmonicaTransit(times, pnl_c=20, pnl_e=50)

runtimes = []
for i in range(n_repeats):
    start = time.time()
    ht.set_orbit(t0, period, a, inc, 0., 0.)
    ht.set_stellar_limb_darkening(us, limb_dark_law='quadratic')
    ht.set_planet_transmission_string(rs)
    ht.get_transit_light_curve()
    end = time.time()
    runtimes.append(end - start)

print('Total runtime: ', np.median(runtimes))
print('Runtime per data point: ', np.median(runtimes) / n_ints)
