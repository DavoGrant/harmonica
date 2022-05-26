import time
import numpy as np
import matplotlib.pyplot as plt
from harmonica import HarmonicaTransit


# todo: update all tests for mean of many trials.
# todo: update all tests on diff machine too.
# todo: update all tests for wasp17 params, and time span from hst-like data.
# todo: do this for with and without grad.
# todo: add mean line as the final number to write.


# n_dp = 1000
# us = np.array([0.40, 0.29])
# n_rs = np.logspace(0, 2, 10).astype(int)
# n_rs = np.array([n + 1 if n % 2 == 0 else n for n in n_rs])
# rs_base = [0.1]
# ts = np.linspace(4.5, 5.5, n_dp)
# sample_rt = []
# ht = HarmonicaTransit(times=ts, require_gradients=False)
# for n_r in n_rs:
#     rs = rs_base + [0.005] * (n_r - 1)
#     s = time.time()
#     ht.set_orbit(t0=5., period=10., a=7., inc=88. * np.pi / 180.,
#                  ecc=0., omega=0. * np.pi / 180)
#     ht.set_stellar_limb_darkening(us, limb_dark_law='quadratic')
#     ht.set_planet_transmission_string(rs)
#     ht.get_transit_light_curve()
#     sample_rt.append((time.time() - s) / n_dp)
#
# fig = plt.figure(figsize=(8, 6))
# ax1 = plt.subplot(1, 1, 1)
#
# ax1.plot(n_rs, sample_rt, '--o', c='#000000')
#
# ax1.set_xscale('log')
# ax1.set_yscale('log')
# ax1.set_xlabel('N Fourier parameters')
# ax1.set_ylabel('Runtime / s')
#
# plt.tight_layout()
# plt.show()

# n_dp = 1000
# us = np.array([0.40, 0.29])
# rs = np.array([0.1, -0.005, 0.005, -0.005, 0.005])
# sample_ds = np.linspace(0.22, 1.39, 100)
# sample_rt = []
# for d in sample_ds:
#     ds = np.ones(n_dp) * d
#     nus = np.linspace(-np.pi, np.pi, n_dp)
#     ht = HarmonicaTransit(ds=ds, nus=nus, require_gradients=False)
#     s = time.time()
#     ht.set_stellar_limb_darkening(us, limb_dark_law='quadratic')
#     ht.set_planet_transmission_string(rs)
#     ht.get_transit_light_curve()
#     sample_rt.append((time.time() - s) / n_dp)
#
# fig = plt.figure(figsize=(8, 6))
# ax1 = plt.subplot(1, 1, 1)
#
# ax1.plot(sample_ds, sample_rt, c='#000000')
#
# ax1.set_yscale('log')
# ax1.set_xlabel('$d$ / stellar radii')
# ax1.set_ylabel('Runtime / s')
#
# plt.tight_layout()
# plt.show()

fig = plt.figure(figsize=(8, 6))
ax1 = plt.subplot(1, 1, 1)

n_iter = 10
n_dps = np.logspace(0, 4, 10)
us = np.array([0.40, 0.29])
rs = np.array([0.1, -0.005, 0.005, -0.005, 0.005])
nls = [20, 50, 100, 200, 500, 0]
colours = ['#ffa600', '#ff6361', '#bc5090', '#58508d', '#003f5c', '#000000']
for nl, colour in zip(nls, colours):
    sample_rt = []
    for n_dp in n_dps:
        iter_rt = []
        ts = np.linspace(4.5, 5.5, int(n_dp))
        ht = HarmonicaTransit(ts, require_gradients=False)
        for i in range(n_iter):
            s = time.time()
            ht.set_orbit(t0=5., period=10., a=7., inc=88. * np.pi / 180.,
                         ecc=0., omega=0. * np.pi / 180)
            ht.set_stellar_limb_darkening(us, limb_dark_law='quadratic')
            ht.set_planet_transmission_string(rs)
            ht.get_precision_estimate(nl)
            iter_rt.append(time.time() - s)
        sample_rt.append(np.mean(iter_rt))
    ax1.plot(n_dps, sample_rt, '--o', c=colour, label='$N_l = {}$'.format(nl))

ax1.set_yscale('log')
ax1.set_xscale('log')
ax1.set_xlabel('N data points')
ax1.set_ylabel('Runtime / s')
ax1.legend(loc='upper left')

plt.tight_layout()
plt.show()
