# First light curve: orbit, ld, rs, create.
# Maybe show some residuals from circular.


import numpy as np
import matplotlib.pyplot as plt
from harmonica import HarmonicaTransit


# Input parameters.
t0 = 5.
period = 10.
a = 7.
inc = 88. * np.pi / 180.
times = np.linspace(4.4, 5.6, 1000)
us = np.array([0.40, 0.29])
rs = np.array([0.1, -0.005, 0.005, -0.005, 0.005])

# Harmonica api example.
ht = HarmonicaTransit(times)
ht.set_orbit(t0=t0, period=period, a=a, inc=inc)
ht.set_stellar_limb_darkening(us, limb_dark_law='quadratic')
ht.set_planet_transmission_string(rs)
fluxes = ht.get_transit_light_curve()

# Draw model.
fig = plt.figure(figsize=(8, 6))
ax1 = plt.subplot(1, 1, 1)
ax1.plot(times, fluxes, c='#000000')
ax1.set_xlabel('Time / days')
ax1.set_ylabel('Relative flux')
plt.tight_layout()
plt.show()
