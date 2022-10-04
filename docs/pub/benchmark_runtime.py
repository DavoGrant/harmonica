import time
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from exotic_ld import StellarLimbDarkening

from harmonica import bindings
from harmonica import HarmonicaTransit


# Config.
np.random.seed(123)
n_repeats = 100

# System params.
t0 = 0.
period = 3.73548546
a = 7.025
inc = 86.9 * np.pi / 180.

sld = StellarLimbDarkening(
    M_H=-0.25, Teff=6550., logg=4.2, ld_model='3D',
    ld_data_path='/Users/davidgrant/research/data/limb_darkening')
u1, u2 = sld.compute_quadratic_ld_coeffs(
    wavelength_range=((4.4 - 0.25) * 1e4, (4.4 + 0.25) * 1e4),
    mode='JWST_NIRSpec_G395H')
us = np.array([u1, u2])

tss = [np.array([0.125]),
       np.array([0.125, -0.002, 0.]),
       np.array([0.125, -0.002, 0., 0.001, 0.]),
       np.array([0.125, -0.002, 0., 0.001, 0., 0.001, 0.]),
       np.array([0.125, -0.002, 0., 0.001, 0., 0.001, 0., 0.001, 0.])]

# Test runtime vs d.
n_evals = 500
n_times = np.linspace(0, 0.099, 15)
ds_1 = []
res_1 = []
done_ds = False
for rs in tss:

    rs_res = []
    for n_t in n_times:
        times = np.ones(n_evals) * n_t

        if not done_ds:
            ds = np.empty(times.shape, dtype=np.float64)
            zs = np.empty(times.shape, dtype=np.float64)
            nus = np.empty(times.shape, dtype=np.float64)
            bindings.orbit(t0, period, a, inc, 0., 0., times, ds, zs, nus)
            ds_1.append(np.median(ds))

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
        print(rs, np.median(ds), 'Runtime: ', np.median(runtimes))

        rs_res.append(np.median(runtimes) / n_evals)

    done_ds = True
    res_1.append(rs_res)

# Test runtime vs n data points.
n_ints = np.logspace(0, 5, 15)
res_2 = []
for rs in tss:

    rs_res = []
    for n_i in n_ints:
        times = np.linspace(-0.21, 0.21, int(n_i))
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
        print(rs, n_i, 'Runtime: ', np.median(runtimes))

        rs_res.append(np.median(runtimes))

    res_2.append(rs_res)

cmap = matplotlib.cm.get_cmap('viridis')
fig = plt.figure(figsize=(6, 11))
ax1 = plt.subplot(2, 1, 1)
ax2 = plt.subplot(2, 1, 2)

ax1.plot(ds_1, res_1[0],
         c=cmap(0.0), alpha=0.85, label='$N_c = 0$', linestyle='--', marker='o')
ax1.plot(ds_1, res_1[1],
         c=cmap(0.275), alpha=0.85, label='$N_c = 1$', linestyle='--', marker='o')
ax1.plot(ds_1, res_1[2],
         c=cmap(0.5), alpha=0.85, label='$N_c = 2$', linestyle='--', marker='o')
ax1.plot(ds_1, res_1[3],
         c=cmap(0.725), alpha=0.85, label='$N_c = 3$', linestyle='--', marker='o')
ax1.plot(ds_1, res_1[4],
         c=cmap(0.95), alpha=0.85, label='$N_c = 4$', linestyle='--', marker='o')

ax2.plot(n_ints, res_2[0],
         c=cmap(0.0), alpha=0.85, label='$N_c = 0$', linestyle='--', marker='o')
ax2.plot(n_ints, res_2[1],
         c=cmap(0.275), alpha=0.85, label='$N_c = 1$', linestyle='--', marker='o')
ax2.plot(n_ints, res_2[2],
         c=cmap(0.5), alpha=0.85, label='$N_c = 2$', linestyle='--', marker='o')
ax2.plot(n_ints, res_2[3],
         c=cmap(0.725), alpha=0.85, label='$N_c = 3$', linestyle='--', marker='o')
ax2.plot(n_ints, res_2[4],
         c=cmap(0.95), alpha=0.85, label='$N_c = 4$', linestyle='--', marker='o')

ax1.tick_params(axis='both', which='both', labelsize=12)
ax2.tick_params(axis='both', which='both', labelsize=12)

ax1.set_yscale('log')
ax2.set_yscale('log')
ax2.set_xscale('log')

ax1.set_xlabel('$d$ / stellar radii', fontsize=14)
ax1.set_ylabel('Runtime per data point / seconds', fontsize=14)
# ax1.legend(loc='upper left', fontsize=12)

ax2.set_xlabel('Number of data points', fontsize=14)
ax2.set_ylabel('Runtime per light curve / seconds', fontsize=14)
ax2.legend(loc='upper left', fontsize=12)

plt.tight_layout()
plt.show()
