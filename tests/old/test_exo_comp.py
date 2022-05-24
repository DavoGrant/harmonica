import time
import numpy as np
import matplotlib.pyplot as plt
from harmonica import HarmonicaTransit
from exoplanet_core import quad_limbdark_light_curve


# Harmonica transit light curve.
s = time.time()
ts = np.linspace(4.5, 5.5, 1000000)
ht = HarmonicaTransit(times=ts, n_l=10, require_gradients=False)
ht.set_orbit(t0=5., period=10., a=7., inc=90. * np.pi / 180.,
             ecc=0., omega=0. * np.pi / 180)
ht.set_stellar_limb_darkening(np.array([0.40, 0.29]), limb_dark_law='quadratic')
ht.set_planet_transmission_string(np.array([0.1]))
fs = ht.get_transit_light_curve()
print((time.time() - s) / 1)

# Exoplanet package transit light curve.
fs_exo = 1. + quad_limbdark_light_curve(0.40, 0.29, ht.ds, 0.1)

plt.plot(ts, fs - fs_exo, lw=1.5, c='#000000')
plt.ylabel('Error')
plt.xlabel('Time / days')
plt.tight_layout()
plt.show()

