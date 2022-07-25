import os
import jax
import emcee
import corner
import numpyro
import arviz as az
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from numpyro.infer.reparam import LocScaleReparam
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, init_to_value, init_to_median

from harmonica import bindings
from harmonica.jax import harmonica_transit


def hlc_generate(t0=0., period=3.735, a=7.025, inc=86.9 * np.pi / 180.,
                 _us=None, _rs=None, _times=None):
    _fs = np.empty(_times.shape)
    bindings.light_curve(
        t0, period, a, inc, 0., 0., 0, _us, _rs, _times, _fs, 20, 50)
    return _fs


# Set number of cores.
os.environ["XLA_FLAGS"] = (
    "--xla_force_host_platform_device_count=1"
)
n_cores = jax.local_device_count()
numpyro.set_host_device_count(n_cores)
print('N cores = ', n_cores)


# Make reproducible.
rng_key = jax.random.PRNGKey(112)
np.random.seed(112)

n_obs = 500
times = np.linspace(-0.22, 0.22, n_obs)
us = np.array([0.074, 0.193])
rs = np.array([0.1, -0.003, 0., 0.003, 0.])
var_names = ['r0', 'r1', 'r2', 'r3', 'r4', 'u1', 'u2']
y_sigma = 15.e-6
y_errs = np.random.normal(loc=0., scale=y_sigma, size=n_obs)

observed_fluxes = hlc_generate(_us=us, _rs=rs, _times=times)
observed_fluxes += y_errs

# fig = plt.figure(figsize=(8, 6))
# ax1 = plt.subplot(1, 1, 1)
# ax1.errorbar(times, observed_fluxes, yerr=y_sigma, fmt=".k", capsize=0)
# ax1.set_xlabel('Time / days')
# ax1.set_ylabel('Relative flux')
# plt.show()


# def log_prob(params):
#     """ Typical Gaussian likelihood. """
#     # Ln prior.
#     ln_prior = -0.5 * np.sum(((params[0] - 0.1) / 0.01)**2)
#     ln_prior += -0.5 * np.sum((params[1:-2] / 0.01)**2)
#     ln_prior += -0.5 * np.sum((params[-2:] / 1.)**2)
#
#     # Ln likelihood.
#     model = hlc_generate(_us=params[-2:], _rs=params[:-2], _times=times)
#     ln_like = -0.5 * np.sum((observed_fluxes - model)**2 / y_sigma**2
#                             + np.log(2 * np.pi * y_sigma**2))
#
#     return ln_like + ln_prior
#
#
# coords = np.array([0.1, 0., 0., 0., 0., 0.1, 0.1]) + 1.e-5 * np.random.randn(16, len(rs) + len(us))
# sampler = emcee.EnsembleSampler(coords.shape[0], coords.shape[1], log_prob)
# state = sampler.run_mcmc(coords, 6000, progress=True)
# chain = sampler.get_chain(discard=2000, flat=True)
#
# emcee_data = az.from_emcee(sampler, var_names)
# print(az.summary(emcee_data, var_names, round_to=6).to_string())
#
# # figure = corner.corner(chain, truths=rs, labels=var_names)
# figure = corner.corner(chain)
# plt.show()


def numpyro_model(t, obs_err, f_obs=None):
    """ Numpyro model. """
    q1 = numpyro.sample('q1', dist.Uniform(0., 1.))
    q2 = numpyro.sample('q2', dist.Uniform(0., 1.))
    u1 = numpyro.deterministic('u1', 2. * jnp.sqrt(q1) * q2)
    u2 = numpyro.deterministic('u2', jnp.sqrt(q1) * (1. - 2. * q2))
    # u1 = numpyro.sample('u1', dist.Uniform(0., 1.))
    # u2 = numpyro.sample('u2', dist.Uniform(0., 1.))

    r0_tilde = numpyro.sample('r0_hat', dist.Normal(0., 1.))
    r0 = numpyro.deterministic('r0', 0.1 + r0_tilde * 0.01)

    r1_frac = numpyro.sample('r1_frac', dist.Normal(0.0, 0.1))
    r2_frac = numpyro.sample('r2_frac', dist.Normal(0.0, 0.1))
    r3_frac = numpyro.sample('r3_frac', dist.Normal(0.0, 0.1))
    r4_frac = numpyro.sample('r4_frac', dist.Normal(0.0, 0.1))

    r1 = numpyro.deterministic('r1', r1_frac * r0)
    r2 = numpyro.deterministic('r2', r2_frac * r0)
    r3 = numpyro.deterministic('r3', r3_frac * r0)
    r4 = numpyro.deterministic('r4', r4_frac * r0)

    # # Model evaluation: this is our custom JAX primitive.
    fs = harmonica_transit(
        t, 0., 3.735, 7.025, 86.9 * np.pi / 180., 0., 0.,
        'quadratic', u1, u2, r0, r1, r2, r3, r4)

    # Condition on the observations
    numpyro.sample('obs', dist.Normal(fs, obs_err), obs=f_obs)


# from numpyro.handlers import reparam
# reparam_model = reparam(numpyro_model, config={
#     "r0": LocScaleReparam(0),
#     "r1": LocScaleReparam(0),
#     "r2": LocScaleReparam(0),
#     "r3": LocScaleReparam(0),
#     "r4": LocScaleReparam(0),})

# Define NUTS kernel.
nuts_kernel = NUTS(
    numpyro_model,
    dense_mass=True,
    adapt_mass_matrix=True,
    max_tree_depth=10,
    target_accept_prob=0.65,
    init_strategy=init_to_median(),
)

# Define HMC sampling strategy.
mcmc = MCMC(nuts_kernel, num_warmup=1000, num_samples=1000,
            num_chains=2, progress_bar=True)

mcmc.run(rng_key, times, y_sigma, f_obs=observed_fluxes)

numpyro_data = az.from_numpyro(mcmc)
# print(az.summary(numpyro_data, var_names=var_names, round_to=6).to_string())
print(az.summary(numpyro_data, round_to=6).to_string())

mcmc.print_summary()
samples = mcmc.get_samples()

# # fig = corner.corner(samples, truths=rs, labels=var_names)
# fig = corner.corner(samples)
# plt.show()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.set_aspect('equal', 'box')
labelled = False
theta = np.linspace(-np.pi, np.pi, 1000)

chain = np.array([samples['r0'], samples['r1'], samples['r2'], samples['r3'], samples['r4']]).T
for sample in chain[np.random.randint(len(chain), size=50)]:
    r_p = np.empty(theta.shape, dtype=np.float64)
    bindings.transmission_string(sample, theta, r_p)
    transmission_string = r_p

    ax1.plot(transmission_string * np.cos(theta), transmission_string * np.sin(theta), color='C0', alpha=0.1)
    ax2.plot(theta * 180. / np.pi, transmission_string, color='C0', alpha=0.1)

r_p = np.empty(theta.shape, dtype=np.float64)
bindings.transmission_string(np.median(chain, axis=0), theta, r_p)
transmission_string = r_p
ax2.plot(theta * 180. / np.pi, transmission_string, color='C0', alpha=1.0)

r_p = np.empty(theta.shape, dtype=np.float64)
bindings.transmission_string(rs, theta, r_p)
transmission_string = r_p
ax1.plot(transmission_string * np.cos(theta), transmission_string * np.sin(theta), lw=1.5, c='#000000')
ax2.plot(theta * 180. / np.pi, transmission_string, c='#000000', lw=1.5, label='True transmission string')

ax1.plot(rs[0] * np.cos(theta), rs[0] * np.sin(theta), lw=1.5, c='#d5d6d2', ls='--')
ax2.plot(theta * 180. / np.pi, np.ones(theta.shape[0]) * rs[0], c='#d5d6d2', lw=1.5, ls='--', label='Circle')

ax2.legend(loc='upper center')
ax1.set_xlabel('x / stellar radii')
ax1.set_ylabel('y / stellar radii')
ax2.set_xlabel('$\\theta$ / degrees')
ax2.set_ylabel('$r_{\\rm{p}}$ / stellar radii')
plt.tight_layout()
plt.show()
