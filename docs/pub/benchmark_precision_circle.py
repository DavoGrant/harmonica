import matplotlib
import numpy as np
from harmonica import bindings
import matplotlib.pyplot as plt
from harmonica import HarmonicaTransit
from exotic_ld import StellarLimbDarkening
from exoplanet_core import quad_limbdark_light_curve


# Config.
t0 = 0.
period = 3.73548546
a = 7.025
inc = 86.9 * np.pi / 180.
_as = np.array([0.1, -0.002, 0.001])
_bs = np.array([0., 0.])
ts = np.linspace(-0.091, 0.091, 500)

sld = StellarLimbDarkening(
    M_H=-0.25, Teff=6550., logg=4.2, ld_model='3D',
    ld_data_path='/Users/davidgrant/research/data/limb_darkening')
u1, u2 = sld.compute_quadratic_ld_coeffs(
    wavelength_range=((4.4 - 0.25) * 1e4, (4.4 + 0.25) * 1e4),
    mode='JWST_NIRSpec_G395H')

us = np.array([u1, u2])
rs = np.array([0.1])

# Harmonica transit light curves.
ht = HarmonicaTransit(times=ts, pnl_c=500, pnl_e=500)
ht.set_orbit(t0, period, a, inc)
ht.set_stellar_limb_darkening(us, limb_dark_law='quadratic')
ht.set_planet_transmission_string(rs)
fs_500 = ht.get_transit_light_curve()
ht = HarmonicaTransit(times=ts, pnl_c=200, pnl_e=200)
ht.set_orbit(t0, period, a, inc)
ht.set_stellar_limb_darkening(us, limb_dark_law='quadratic')
ht.set_planet_transmission_string(rs)
fs_200 = ht.get_transit_light_curve()
ht = HarmonicaTransit(times=ts, pnl_c=100, pnl_e=100)
ht.set_orbit(t0, period, a, inc)
ht.set_stellar_limb_darkening(us, limb_dark_law='quadratic')
ht.set_planet_transmission_string(rs)
fs_100 = ht.get_transit_light_curve()
ht = HarmonicaTransit(times=ts, pnl_c=50, pnl_e=50)
ht.set_orbit(t0, period, a, inc)
ht.set_stellar_limb_darkening(us, limb_dark_law='quadratic')
ht.set_planet_transmission_string(rs)
fs_50 = ht.get_transit_light_curve()
ht = HarmonicaTransit(times=ts, pnl_c=20, pnl_e=20)
ht.set_orbit(t0, period, a, inc)
ht.set_stellar_limb_darkening(us, limb_dark_law='quadratic')
ht.set_planet_transmission_string(rs)
fs_20 = ht.get_transit_light_curve()
ht = HarmonicaTransit(times=ts)
ht.set_orbit(t0, period, a, inc)
ht.set_stellar_limb_darkening(us, limb_dark_law='quadratic')
ht.set_planet_transmission_string(rs)
fs = ht.get_transit_light_curve()

# Get orbit.
ds = np.empty(ts.shape, dtype=np.float64)
zs = np.empty(ts.shape, dtype=np.float64)
nus = np.empty(ts.shape, dtype=np.float64)
bindings.orbit(t0, period, a, inc, 0., 0., ts, ds, zs, nus)

# Exoplanet package transit light curve.
fs_exo = 1. + quad_limbdark_light_curve(us[0], us[1], ds, rs[0])

cmap = matplotlib.cm.get_cmap('viridis')
fig = plt.figure(figsize=(6, 9))
ax1 = plt.subplot(1, 1, 1)

ax1.plot(ds, np.abs(fs_20 - fs_exo),
         c=cmap(0.05), label='$N_l = 20$')
ax1.plot(ds, np.abs(fs_50 - fs_exo),
         c=cmap(0.275), label='$N_l = 50$')
ax1.plot(ds, np.abs(fs_100 - fs_exo),
         c=cmap(0.5), label='$N_l = 100$')
ax1.plot(ds, np.abs(fs_200 - fs_exo),
         c=cmap(0.725), label='$N_l = 200$')
ax1.plot(ds, np.abs(fs_500 - fs_exo),
         c=cmap(0.95), label='$N_l = 500$')

ax1.axhline(10.e-6, ls='--', color='#000000', alpha=0.5)
ax1.text(0.9, 15.e-6, 'JWST noise floor', color='#000000', alpha=0.6)

ax1.set_ylim(1.e-17, 7.e-3)
ax1.set_yscale('log')
ax1.set_xlabel('$d$ / stellar radii')
ax1.set_ylabel('Error')
ax1.legend(loc='upper left')

plt.tight_layout()
plt.show()
