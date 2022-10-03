import os
import jax
import time
import corner
import pickle
import numpyro
import arviz as az
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpyro.distributions as dist
from exotic_ld import StellarLimbDarkening
from numpyro.infer import MCMC, NUTS, init_to_median

# Set number of cores.
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"
numpyro.set_host_device_count(jax.local_device_count())

from harmonica import HarmonicaTransit
from harmonica.jax import harmonica_transit_quad_ld


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


# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
# ax1.set_aspect('equal', 'box')
# labelled = False
# theta = np.linspace(-np.pi, np.pi, 1000)
#
# ht.set_planet_transmission_string(rs)
# transmission_string = ht.get_planet_transmission_string(theta)
# ax1.plot(transmission_string * np.cos(theta), transmission_string * np.sin(theta), lw=1.5, c='#d5d6d2')
# ax2.plot(theta * 180. / np.pi, transmission_string, c='#ffffff', lw=1.5, label='True transmission string')
#
# ax1.plot(rs[0] * np.cos(theta), rs[0] * np.sin(theta), lw=1.5, c='#d5d6d2', ls='--')
# ax2.plot(theta * 180. / np.pi, np.ones(theta.shape[0]) * rs[0], c='#d5d6d2', lw=1.5, ls='--', label='Circle')
#
# ax2.legend(loc='upper center')
# ax1.set_xlabel('x / stellar radii', fontsize=15)
# ax1.set_ylabel('y / stellar radii', fontsize=15)
# ax2.set_xlabel('$\\theta$ / degrees', fontsize=15)
# ax2.set_ylabel('$r_{\\rm{p}}$ / stellar radii', fontsize=15)
# plt.tight_layout()
# plt.show()
#
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
# ax1.set_aspect('equal', 'box')
# labelled = False
# theta = np.linspace(-np.pi, np.pi, 1000)
#
# ht.set_planet_transmission_string(rs)
# transmission_string = ht.get_planet_transmission_string(theta)
# ax1.plot(rs[0] * np.cos(theta), rs[0] * np.sin(theta), lw=1.5, c='#d5d6d2', ls='--')
# ax1.plot(transmission_string * np.cos(theta), transmission_string * np.sin(theta), lw=1.5, c='#ffffff')
#
# ax2.errorbar(times, observed_fluxes, yerr=y_sigma, fmt="#ffffff", capsize=0)
# ax2.set_xlabel('Time / days', fontsize=15)
# ax2.set_ylabel('Relative flux', fontsize=15)
# ax1.set_xlabel('x / stellar radii', fontsize=15)
# ax1.set_ylabel('y / stellar radii', fontsize=15)
# plt.tight_layout()
# plt.show()


def numpyro_model(t, obs_err, f_obs=None):
    """ Numpyro model. """
    # NB: non-centred r0 or uniform,
    #     rn n > 0 as normal fraction of r0,
    #     dense_mass=true,
    #     max_tree_depth=default 10,
    #     target_accept_prob=0.65.

    # # Kipping 2013 quad ld.
    # q1 = numpyro.sample('q1', dist.Uniform(0., 1.))
    # q2 = numpyro.sample('q2', dist.Uniform(0., 1.))
    # u1 = numpyro.deterministic('u1', 2. * jnp.sqrt(q1) * q2)
    # u2 = numpyro.deterministic('u2', jnp.sqrt(q1) * (1. - 2. * q2))

    # Zeroth-order planet radius: r0.
    r0 = numpyro.sample('r0', dist.Uniform(0.10, 0.14))

    # Higher-order radius harmonics: r1/r0, r2/r0, r3/r0, r4/r0.
    rn_frac = numpyro.sample('rn_frac', dist.Normal(0.0, 0.1), sample_shape=(4,))

    # Transmission string parameter vector.
    r = numpyro.deterministic('r', jnp.concatenate([jnp.array([r0]), rn_frac * r0]))

    # Model evaluation: this is our custom JAX primitive.
    fs = harmonica_transit_quad_ld(
        t, t0, period, a, inc, u1=us[0], u2=us[1], r=r)

    # Condition on the observations
    numpyro.sample('obs', dist.Normal(fs, obs_err), obs=f_obs)


# Define NUTS kernel.
nuts_kernel = NUTS(
    numpyro_model,
    dense_mass=True, adapt_mass_matrix=True,
    max_tree_depth=10, target_accept_prob=0.65,
    init_strategy=init_to_median(),
)

# Define HMC sampling strategy.
mcmc = MCMC(nuts_kernel,
            num_warmup=2000, num_samples=5000,
            num_chains=2, progress_bar=False)
start = time.time()
mcmc.run(rng_key, times, y_sigma, f_obs=observed_fluxes)
numpyro_time = time.time() - start

numpyro_trace = az.from_numpyro(mcmc)
samples = mcmc.get_samples()
# chain = np.concatenate([np.array(samples['u1'])[..., np.newaxis],
#                         np.array(samples['u2'])[..., np.newaxis],
#                         np.array(samples['r'])], axis=1)
chain = np.array(samples['r'])


print(az.summary(numpyro_trace, var_names=['r'], round_to=6).to_string())

numpyro_per_eff = numpyro_time / np.mean(az.summary(numpyro_trace)["ess_bulk"])
print('Numpyro runtime = {} s'.format(
    round(numpyro_time, 2)))
print('Numpyro runtime per effective sample = {} ms'.format(
    round(numpyro_per_eff * 1.e3, 2)))

ht = HarmonicaTransit(times)
ht.set_orbit(t0, period, a, inc)
ht.set_stellar_limb_darkening(us)
ht.set_planet_transmission_string(np.median(chain[:, :], axis=0))
lc = ht.get_transit_light_curve()
print('reduced chi-squared', np.sum(((observed_fluxes - lc) / y_sigma)**2) / (n_obs - 5))
print('model non-linear? check residuals or cross-validation')

with open('numpyro_v1.p', 'wb') as out_f:
    pickle.dump(chain, out_f)
