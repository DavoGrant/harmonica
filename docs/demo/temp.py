import numpy as np
import matplotlib.pyplot as plt
from harmonica import HarmonicaTransit

# Generate data.
np.random.seed(1)
n_obs = 20000
times = np.linspace(-0.22, 0.22, n_obs)
us = np.array([0.074, 0.193])
rs = np.array([0.1, 0., 0.01])
# rs = np.zeros((n_obs, 11))
# rs[:, 0] = 0.1
# rs[:, 4] = 0.01

ht = HarmonicaTransit(times, pnl_c=20, pnl_e=50)
ht.set_orbit(t0=0., period=5., a=7., inc=89. * np.pi / 180.)
ht.set_stellar_limb_darkening(us, limb_dark_law='quadratic')
ht.set_planet_transmission_string(rs)

theta = np.linspace(-np.pi, np.pi, 10000)
transmission_string = ht.get_planet_transmission_string(theta)
observed_fluxes = ht.get_transit_light_curve()
precisions = ht.get_precision_estimate()

fig = plt.figure(figsize=(8, 6))
ax1 = plt.subplot(1, 1, 1)
ax1.plot(theta, transmission_string)
ax1.set_xlabel('Theta')
ax1.set_ylabel('r_p')
plt.show()

fig = plt.figure(figsize=(8, 6))
ax1 = plt.subplot(1, 1, 1)
ax1.plot(times, observed_fluxes)
ax1.set_xlabel('Time / days')
ax1.set_ylabel('Relative flux')
plt.show()

fig = plt.figure(figsize=(8, 6))
ax1 = plt.subplot(1, 1, 1)
ax1.plot(times, precisions)
ax1.set_xlabel('Time / days')
ax1.set_ylabel('Precisions')
plt.show()
