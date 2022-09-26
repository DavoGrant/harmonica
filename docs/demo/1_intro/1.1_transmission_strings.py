# Intro to definition of transmission strings.
# Visualisation video.
# Parameterisation.
# Using harmonica to create them.


import numpy as np
import matplotlib.pyplot as plt
from harmonica import HarmonicaTransit




# fig = plt.figure(figsize=(8, 8))
# ax1 = plt.subplot(1, 1, 1)
# ax1.set_aspect('equal', 'box')
#
# theta = np.linspace(-np.pi, np.pi, 1000)
#
# ht = HarmonicaTransit()
# rs = np.array([0.1, -0.02, 0.01, 0.01, 0.005, -0.009, -0.003, -0.001, 0.001])
# ht.set_planet_transmission_string(rs)
# transmission_string_truht = ht.get_planet_transmission_string(theta)
#
#
# ax1.plot(transmission_string_truht * np.cos(theta),
#          transmission_string_truht * np.sin(theta),
#          lw=2.5, c='#ffffff', ls='--')
# plt.axis('off')
# ax1.set_xlim(-0.175, 0.175)
# ax1.set_ylim(-0.175, 0.175)
# plt.savefig('/Users/davidgrant/Downloads/fourier_{}.png'.format(00), transparent=True)
#
# for _rr in range(1, 10, 2):
#     fig = plt.figure(figsize=(8, 8))
#     ax1 = plt.subplot(1, 1, 1)
#     ax1.set_aspect('equal', 'box')
#
#     theta = np.linspace(-np.pi, np.pi, 1000)
#
#     ht = HarmonicaTransit()
#     rs_new = np.copy(rs[:_rr])
#     ht.set_planet_transmission_string(rs_new)
#     transmission_string = ht.get_planet_transmission_string(theta)
#
#
#     ax1.plot(transmission_string_truht * np.cos(theta),
#              transmission_string_truht * np.sin(theta),
#              lw=2.5, c='#ffffff', ls='--')
#
#     ax1.plot(transmission_string * np.cos(theta),
#              transmission_string * np.sin(theta),
#              lw=2.5, c='#FFD200')
#     plt.axis('off')
#     ax1.set_xlim(-0.175, 0.175)
#     ax1.set_ylim(-0.175, 0.175)
#
#     plt.savefig('/Users/davidgrant/Downloads/fourier_{}.png'.format(_rr), transparent=True)
#
# exit()



# Input parameters.
rs = np.array([0.1, -0.005, 0., 0.005, 0.])
theta = np.linspace(-np.pi, np.pi, 1000)

# Harmonica api example.
ht = HarmonicaTransit()
ht.set_planet_transmission_string(rs)
transmission_string = ht.get_planet_transmission_string(theta)

# Draw transmission string.
fig = plt.figure(figsize=(12, 5))
ax1 = plt.subplot(1, 2, 1)
ax2 = plt.subplot(1, 2, 2)
ax1.set_aspect('equal', 'box')

ax1.plot(transmission_string * np.cos(theta),
         transmission_string * np.sin(theta),
         lw=1.5, c='#FFD200')
ax1.plot(rs[0] * np.cos(theta),
         rs[0] * np.sin(theta),
         lw=1.5, c='#D5D6D2', ls='--')
ax2.plot(theta * 180. / np.pi, transmission_string,
         c='#FFD200', lw=1.5, label='Transmission string')
ax2.plot(theta * 180. / np.pi, np.ones(theta.shape[0]) * rs[0],
         c='#D5D6D2', lw=1.5, ls='--', label='Circle')

ax1.set_xlabel('x / stellar radii')
ax1.set_ylabel('y / stellar radii')
ax2.set_xlabel('$\\theta$ / degrees')
ax2.set_ylabel('$r_{\\rm{p}}$ / stellar radii')
ax2.legend()

plt.tight_layout()
plt.show()

