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

times = np.linspace(-0.16, 0.16, 10000)
colours = ['#FFD200', '#3cb557', '#007c82', '#003f5c']

fig = plt.figure(figsize=(8, 8))
ax1 = plt.subplot(1, 1, 1)
for i, c in zip(np.linspace(-0.002, 0.002, 4), colours):

    ht = HarmonicaTransit(times, pnl_c=500, pnl_e=500)
    ht.set_orbit(t0, period, a, inc, 0., 0.)
    ht.set_stellar_limb_darkening(us, limb_dark_law='quadratic')

    # Circle.
    rs = np.array([0.3])
    ht.set_planet_transmission_string(rs)
    circle_lc = ht.get_transit_light_curve()
    rs = np.array([0.3 + i])
    ht.set_planet_transmission_string(rs)
    ts_lc = ht.get_transit_light_curve()

    ax1.plot(times, np.zeros(times.shape),
             lw=3.5, c='#000000', ls='--')
    ax1.plot(times, ts_lc - circle_lc,
             lw=3.5, c=c, alpha=0.8)

plt.axis('off')
ax1.set_xlim(-0.17, 0.17)
# ax1.set_ylim(-0.00025, 0.00025)
plt.savefig('/Users/davidgrant/Desktop/residuals_n0.png', transparent=True)
# plt.show()

fig = plt.figure(figsize=(8, 8))
ax1 = plt.subplot(1, 1, 1)
for i, c in zip(np.linspace(-0.002, 0.002, 4), colours):

    ht = HarmonicaTransit(times, pnl_c=500, pnl_e=500)
    ht.set_orbit(t0, period, a, inc, 0., 0.)
    ht.set_stellar_limb_darkening(us, limb_dark_law='quadratic')

    # Circle.
    rs = np.array([0.3])
    ht.set_planet_transmission_string(rs)
    circle_lc = ht.get_transit_light_curve()
    rs = np.array([0.3, i, 0.])
    ht.set_planet_transmission_string(rs)
    ts_lc = ht.get_transit_light_curve()

    ax1.plot(times, np.zeros(times.shape),
             lw=3.5, c='#000000', ls='--')
    ax1.plot(times, ts_lc - circle_lc,
             lw=3.5, c=c, alpha=0.8)

plt.axis('off')
ax1.set_xlim(-0.17, 0.17)
# ax1.set_ylim(-0.00025, 0.00025)
plt.savefig('/Users/davidgrant/Desktop/residuals_n1_a.png', transparent=True)
# plt.show()

fig = plt.figure(figsize=(8, 8))
ax1 = plt.subplot(1, 1, 1)
for i, c in zip(np.linspace(-0.002, 0.002, 4), colours):

    ht = HarmonicaTransit(times, pnl_c=500, pnl_e=500)
    ht.set_orbit(t0, period, a, inc, 0., 0.)
    ht.set_stellar_limb_darkening(us, limb_dark_law='quadratic')

    # Circle.
    rs = np.array([0.3])
    ht.set_planet_transmission_string(rs)
    circle_lc = ht.get_transit_light_curve()
    rs = np.array([0.3, 0., i])
    ht.set_planet_transmission_string(rs)
    ts_lc = ht.get_transit_light_curve()

    ax1.plot(times, np.zeros(times.shape),
             lw=3.5, c='#000000', ls='--')
    ax1.plot(times, ts_lc - circle_lc,
             lw=3.5, c=c, alpha=0.8)

plt.axis('off')
ax1.set_xlim(-0.17, 0.17)
# ax1.set_ylim(-0.00025, 0.00025)
plt.savefig('/Users/davidgrant/Desktop/residuals_n1_b.png', transparent=True)
# plt.show()

fig = plt.figure(figsize=(8, 8))
ax1 = plt.subplot(1, 1, 1)
for i, c in zip(np.linspace(-0.002, 0.002, 4), colours):

    ht = HarmonicaTransit(times, pnl_c=500, pnl_e=500)
    ht.set_orbit(t0, period, a, inc, 0., 0.)
    ht.set_stellar_limb_darkening(us, limb_dark_law='quadratic')

    # Circle.
    rs = np.array([0.3])
    ht.set_planet_transmission_string(rs)
    circle_lc = ht.get_transit_light_curve()
    rs = np.array([0.3, 0., 0., i, 0.])
    ht.set_planet_transmission_string(rs)
    ts_lc = ht.get_transit_light_curve()

    ax1.plot(times, np.zeros(times.shape),
             lw=3.5, c='#000000', ls='--')
    ax1.plot(times, ts_lc - circle_lc,
             lw=3.5, c=c, alpha=0.8)

plt.axis('off')
ax1.set_xlim(-0.17, 0.17)
# ax1.set_ylim(-0.00025, 0.00025)
plt.savefig('/Users/davidgrant/Desktop/residuals_n2_a.png', transparent=True)
# plt.show()

fig = plt.figure(figsize=(8, 8))
ax1 = plt.subplot(1, 1, 1)
for i, c in zip(np.linspace(-0.002, 0.002, 4), colours):

    ht = HarmonicaTransit(times, pnl_c=500, pnl_e=500)
    ht.set_orbit(t0, period, a, inc, 0., 0.)
    ht.set_stellar_limb_darkening(us, limb_dark_law='quadratic')

    # Circle.
    rs = np.array([0.3])
    ht.set_planet_transmission_string(rs)
    circle_lc = ht.get_transit_light_curve()
    rs = np.array([0.3, 0., 0., 0., i])
    ht.set_planet_transmission_string(rs)
    ts_lc = ht.get_transit_light_curve()

    ax1.plot(times, np.zeros(times.shape),
             lw=3.5, c='#000000', ls='--')
    ax1.plot(times, ts_lc - circle_lc,
             lw=3.5, c=c, alpha=0.8)

plt.axis('off')
ax1.set_xlim(-0.17, 0.17)
# ax1.set_ylim(-0.00025, 0.00025)
plt.savefig('/Users/davidgrant/Desktop/residuals_n2_b.png', transparent=True)
# plt.show()

fig = plt.figure(figsize=(8, 8))
ax1 = plt.subplot(1, 1, 1)
for i, c in zip(np.linspace(-0.002, 0.002, 4), colours):

    ht = HarmonicaTransit(times, pnl_c=500, pnl_e=500)
    ht.set_orbit(t0, period, a, inc, 0., 0.)
    ht.set_stellar_limb_darkening(us, limb_dark_law='quadratic')

    # Circle.
    rs = np.array([0.3])
    ht.set_planet_transmission_string(rs)
    circle_lc = ht.get_transit_light_curve()
    rs = np.array([0.3, 0., 0., 0., 0., i, 0.])
    ht.set_planet_transmission_string(rs)
    ts_lc = ht.get_transit_light_curve()

    ax1.plot(times, np.zeros(times.shape),
             lw=3.5, c='#000000', ls='--')
    ax1.plot(times, ts_lc - circle_lc,
             lw=3.5, c=c, alpha=0.8)

plt.axis('off')
ax1.set_xlim(-0.17, 0.17)
# ax1.set_ylim(-0.00025, 0.00025)
plt.savefig('/Users/davidgrant/Desktop/residuals_n3_a.png', transparent=True)
# plt.show()

fig = plt.figure(figsize=(8, 8))
ax1 = plt.subplot(1, 1, 1)
for i, c in zip(np.linspace(-0.002, 0.002, 4), colours):

    ht = HarmonicaTransit(times, pnl_c=500, pnl_e=500)
    ht.set_orbit(t0, period, a, inc, 0., 0.)
    ht.set_stellar_limb_darkening(us, limb_dark_law='quadratic')

    # Circle.
    rs = np.array([0.3])
    ht.set_planet_transmission_string(rs)
    circle_lc = ht.get_transit_light_curve()
    rs = np.array([0.3, 0., 0., 0., 0., 0., i])
    ht.set_planet_transmission_string(rs)
    ts_lc = ht.get_transit_light_curve()

    ax1.plot(times, np.zeros(times.shape),
             lw=3.5, c='#000000', ls='--')
    ax1.plot(times, ts_lc - circle_lc,
             lw=3.5, c=c, alpha=0.8)

plt.axis('off')
ax1.set_xlim(-0.17, 0.17)
# ax1.set_ylim(-0.00025, 0.00025)
plt.savefig('/Users/davidgrant/Desktop/residuals_n3_b.png', transparent=True)
# plt.show()
