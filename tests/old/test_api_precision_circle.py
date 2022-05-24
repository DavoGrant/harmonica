import numpy as np
import matplotlib.pyplot as plt
from harmonica import HarmonicaTransit
from exoplanet_core import quad_limbdark_light_curve


# Config.
t0 = 5.
period = 10.
a = 7.
inc = 88. * np.pi / 180.
us = np.array([0.40, 0.29])
rs = np.array([0.1])
ts = np.linspace(4.7, 5.3, 10000)

# Harmonica transit light curve.
ht = HarmonicaTransit(times=ts, require_gradients=False)
ht.set_orbit(t0, period, a, inc)
ht.set_stellar_limb_darkening(us, limb_dark_law='quadratic')
ht.set_planet_transmission_string(rs)
fs = ht.get_transit_light_curve()

# Exoplanet package transit light curve.
fs_exo = 1. + quad_limbdark_light_curve(us[0], us[1], ht.ds, rs[0])

plt.plot(ht.ds, fs - fs_exo, lw=1.5, c='#000000')
plt.ylabel('Error')
plt.xlabel('d / stellar radii')
plt.tight_layout()
plt.show()

