import numpy as np
import matplotlib.pyplot as plt
from harmonica import HarmonicaTransit


# Config.
t0 = 5.
period = 10.
a = 7.
inc = 88. * np.pi / 180.
us = np.array([0.40, 0.29])
rs = np.array([0.1, 0.001, 0.001, 0.001, 0.001])
ts = np.linspace(4.7, 5.3, 10000)

# Harmonica transit light curve.
ht = HarmonicaTransit(times=ts, require_gradients=False)
ht.set_orbit(t0, period, a, inc)
ht.set_stellar_limb_darkening(us, limb_dark_law='quadratic')
ht.set_planet_transmission_string(rs)
fs = ht.get_transit_light_curve()

# Harmonica transit light curve, max precision.
ht = HarmonicaTransit(times=ts, require_gradients=False)
ht.set_orbit(t0, period, a, inc)
ht.set_stellar_limb_darkening(us, limb_dark_law='quadratic')
ht.set_planet_transmission_string(rs)
fs_mp = ht.get_precision_estimate()

plt.plot(ts, fs - fs_mp, lw=1.5, c='#000000')
plt.tight_layout()
plt.show()
