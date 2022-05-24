import time
import numpy as np
from harmonica import HarmonicaTransit


# Harmonica transit light curve.
s = time.time()
ts = np.linspace(4.5, 5.5, 1000)
ht = HarmonicaTransit(times=ts, require_gradients=False)
for i in range(1000):
    ht.set_orbit(t0=5., period=10., a=7., inc=88. * np.pi / 180.,
                 ecc=0., omega=0. * np.pi / 180)
    ht.set_stellar_limb_darkening(np.array([0.40, 0.29]), limb_dark_law='quadratic')
    ht.set_planet_transmission_string(np.array([0.1, 0.001, 0.001, 0.001, 0.001]))
    ht.get_transit_light_curve()
print((time.time() - s) / 1000)