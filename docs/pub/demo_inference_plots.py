import jax
import corner
import pickle
import numpy as np
import matplotlib.pyplot as plt
from exotic_ld import StellarLimbDarkening
from mpl_toolkits.axes_grid1 import make_axes_locatable

from harmonica import HarmonicaTransit


# Make reproducible.
rng_key = jax.random.PRNGKey(17)
np.random.seed(17)

# Observations config.
n_obs = 587
y_sigma = 71.e-6

# Injected orbit.
times = np.linspace(-0.205, 0.205, n_obs)
t0 = 0.
period = 3.73548546
a = 7.025
inc = 86.9 * np.pi / 180.

# Limb darkening.
sld = StellarLimbDarkening(
    M_H=-0.25, Teff=6550., logg=4.2, ld_model='3D',
    ld_data_path='/Users/davidgrant/research/data/limb_darkening')
u1, u2 = sld.compute_quadratic_ld_coeffs(
    wavelength_range=(2.7 * 1e4, 3.7 * 1e4),
    mode='JWST_NIRSpec_G395H')
us = np.array([u1, u2])

# Planet shape.
rs = np.array([0.125, -0.002, 0., 0.001, 0.])
var_names = ['r0', 'r1', 'r2', 'r3', 'r4']

# Make data.
ht = HarmonicaTransit(times)
ht.set_orbit(t0, period, a, inc)
ht.set_stellar_limb_darkening(us)
ht.set_planet_transmission_string(rs)
observed_fluxes = ht.get_transit_light_curve()
y_errs = np.random.normal(loc=0., scale=y_sigma, size=n_obs)
observed_fluxes += y_errs

# Load mcmc chain.
with open('numpyro_v1.p', 'rb') as in_f:
    chain = pickle.load(in_f)

# Histograms and covariance.
fig = corner.corner(chain, bins=25,
                    truths=np.array([0.125, -0.002, 0., 0.001, 0.]),
                    truth_color="#3cb557",
                    labels=['$a_0$', '$a_1$', '$b_1$', '$a_2$', '$b_2$'],
                    label_kwargs={"fontsize": 14}, labelpad=0.03,
                    show_titles=True, title_fmt='.4f', title_kwargs={"fontsize": 13},
                    color='k')

# Model inset.
sub_axes = plt.axes([0.54, 0.68, 0.428, 0.288])
# divider = make_axes_locatable(sub_axes)
# axr = divider.append_axes('bottom', size='30%', pad=0)
# axr.get_shared_x_axes().join(sub_axes, axr)

sub_axes.scatter(times, observed_fluxes, s=5, color='#000000', alpha=0.8)
for sample in chain[np.random.randint(len(chain), size=50), :]:
    ht = HarmonicaTransit(times)
    ht.set_orbit(t0, period, a, inc)
    ht.set_stellar_limb_darkening(us)
    ht.set_planet_transmission_string(sample)
    lc = ht.get_transit_light_curve()
    sub_axes.plot(times, lc, color='#3cb557', alpha=0.1, linewidth=2)

sub_axes.tick_params(axis='both', which='major', labelsize=10)
sub_axes.tick_params(axis='both', which='minor', labelsize=10)
sub_axes.set_ylabel('Relative Flux', fontsize=14, labelpad=5)
sub_axes.set_xlabel('Time / days', fontsize=14, labelpad=5)
plt.show()

# # Alternative model inset.
# sub_axes = plt.axes([0.59, 0.59, 0.39, 0.39])
# theta = np.linspace(-np.pi, np.pi, 1000)
# for sample in chain[np.random.randint(len(chain), size=100), :]:
#     ht = HarmonicaTransit(times)
#     ht.set_planet_transmission_string(sample)
#     transmission_string = ht.get_planet_transmission_string(theta)
#     transmission_string -= 0.11
#     sub_axes.plot(transmission_string * np.cos(theta), transmission_string * np.sin(theta), color='#3cb557', alpha=0.1)
#
# sub_axes.set_aspect('equal', 'box')
# sub_axes.axis('off')
# plt.show()


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.set_aspect('equal', 'box')
labelled = False
theta = np.linspace(-np.pi, np.pi, 1000)

ht.set_planet_transmission_string(rs)
transmission_string = ht.get_planet_transmission_string(theta)
ax1.plot(transmission_string * np.cos(theta), transmission_string * np.sin(theta), lw=2.0, c='#000000')
ax2.plot(theta * 180. / np.pi, transmission_string, c='#000000', lw=2.0, label='True transmission string')

ht = HarmonicaTransit()
for sample in chain[np.random.randint(len(chain), size=50), :]:
    ht.set_planet_transmission_string(sample)
    transmission_string = ht.get_planet_transmission_string(theta)

    ax1.plot(transmission_string * np.cos(theta), transmission_string * np.sin(theta), color='#3cb557', alpha=0.1)
    ax2.plot(theta * 180. / np.pi, transmission_string, color='#3cb557', alpha=0.1)

ht.set_planet_transmission_string(np.median(chain[:, :], axis=0))
transmission_string = ht.get_planet_transmission_string(theta)
ax2.plot(theta * 180. / np.pi, transmission_string, color='#3cb557', lw=2.0, alpha=1.0, label='Inferred transmission string')

ax1.plot(rs[0] * np.cos(theta), rs[0] * np.sin(theta), lw=1.5, c='#b3b4b1', ls='--')
ax2.plot(theta * 180. / np.pi, np.ones(theta.shape[0]) * rs[0], c='#b3b4b1', lw=2.0, ls='--', label='Circle')

ax2.legend(loc='upper center', fontsize=12)
ax1.set_xlabel('x / stellar radii', fontsize=14)
ax1.set_ylabel('y / stellar radii', fontsize=14)
ax2.set_xlabel('$\\theta$ / degrees', fontsize=14)
ax2.set_ylabel('$r_{\\rm{p}}$ / stellar radii', fontsize=14)
plt.tight_layout()
plt.show()


# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
# ax1.set_aspect('equal', 'box')
# labelled = False
# theta = np.linspace(-np.pi, np.pi, 1000)
#
# ht.set_planet_transmission_string(rs)
# transmission_string = ht.get_planet_transmission_string(theta)
# ax1.plot(transmission_string * np.cos(theta), transmission_string * np.sin(theta), lw=1.5, c='#000000')
# ax2.plot(theta * 180. / np.pi, transmission_string, c='#000000', lw=1.5, label='True transmission string')
#
# ax1.plot(rs[0] * np.cos(theta), rs[0] * np.sin(theta), lw=1.5, c='#b3b4b1', ls='--')
# ax2.plot(theta * 180. / np.pi, np.ones(theta.shape[0]) * rs[0], c='#b3b4b1', lw=1.5, ls='--', label='Circle')
#
# ht = HarmonicaTransit()
# for sample in chain[np.random.randint(len(chain), size=50), :]:
#     ht.set_planet_transmission_string(sample)
#     transmission_string = ht.get_planet_transmission_string(theta)
#     ax1.plot(transmission_string * np.cos(theta), transmission_string * np.sin(theta), color='#3cb557', alpha=0.15)
#
# tss = []
# for sample in chain:
#     ht.set_planet_transmission_string(sample)
#     transmission_string = ht.get_planet_transmission_string(theta)
#     tss.append(transmission_string)
# ts_2 = np.percentile(tss, 2.5, axis=0)
# ts_16 = np.percentile(tss, 16., axis=0)
# ts_50 = np.percentile(tss, 50., axis=0)
# ts_84 = np.percentile(tss, 84., axis=0)
# ts_98 = np.percentile(tss, 98.5, axis=0)
#
# ax2.fill_between(theta * 180. / np.pi, ts_84, ts_16, color="#3cb557", alpha=0.2)
# # ax2.fill_between(theta * 180. / np.pi, ts_98, ts_2, color="#FFD200", alpha=0.2)
# ax2.plot(theta * 180. / np.pi, ts_50, color="#3cb557", lw=2)
#
# ax2.legend(loc='upper center')
# ax1.set_xlabel('x / stellar radii')
# ax1.set_ylabel('y / stellar radii')
# ax2.set_xlabel('$\\theta$ / degrees')
# ax2.set_ylabel('$r_{\\rm{p}}$ / stellar radii')
# plt.tight_layout()
# plt.show()


# # Plot of covariance of the transmission string at different angles.
# theta_samples = np.linspace(0, np.pi, 2)
# rp_samples = []
# for sample in chain:
#     ht.set_planet_transmission_string(sample)
#     transmission_string = ht.get_planet_transmission_string(theta_samples)
#     rp_samples.append(transmission_string)
#
# rp_samples = np.array(rp_samples)
# fig = corner.corner(rp_samples, bins=25,
#                     labels=np.round(theta_samples / np.pi * 180, 1),
#                     color='k')
# plt.show()

