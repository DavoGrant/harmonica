import numpy as np
from harmonica import bindings
import matplotlib.pyplot as plt
from harmonica import HarmonicaTransit
from exoplanet_core import quad_limbdark_light_curve


# Config.
t0 = 5.
period = 10.
a = 7.
inc = 88. * np.pi / 180.
us = np.array([0.07, 0.123])
rs = np.array([0.2])
ts = np.linspace(4.7, 5.3, 1000)

# Harmonica transit light curve.
ht = HarmonicaTransit(times=ts, pnl_c=500, pnl_e=500)
ht.set_orbit(t0, period, a, inc)
ht.set_stellar_limb_darkening(us, limb_dark_law='quadratic')
ht.set_planet_transmission_string(rs)
fs = ht.get_transit_light_curve()

ds = np.empty(ts.shape, dtype=np.float64)
zs = np.empty(ts.shape, dtype=np.float64)
nus = np.empty(ts.shape, dtype=np.float64)
bindings.orbit(t0, period, a, inc, 0., 0., ts, ds, zs, nus)

# Exoplanet package transit light curve.
fs_exo = 1. + quad_limbdark_light_curve(us[0], us[1], ds, rs[0])

fig = plt.figure(figsize=(8, 6))
ax1 = plt.subplot(1, 1, 1)

ax1.plot(ds, np.abs(fs - fs_exo),
         c='#000000', label='exo-core comparison')

ax1.set_ylim(1.e-17, 1.e-7)
ax1.set_yscale('log')
ax1.set_xlabel('$d$ / stellar radii')
ax1.set_ylabel('Error')
ax1.legend(loc='upper left')

plt.tight_layout()
plt.show()

