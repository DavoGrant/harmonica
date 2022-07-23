import os
import jax
import emcee
import corner
import numpyro
import arviz as az
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, init_to_value

from harmonica import bindings
from harmonica.jax import harmonica_transit


def hlc_generate(t0=0., period=3.735, a=7.025, inc=86.9 * np.pi / 180.,
                 _us=None, _rs=None, _times=None):
    _fs = np.empty(_times.shape)
    bindings.light_curve(
        t0, period, a, inc, 0., 0., 0, us, rs, _times, _fs, 20, 50)
    return _fs


# Set number of cores.
os.environ["XLA_FLAGS"] = (
    "--xla_force_host_platform_device_count=1"
)
n_cores = jax.local_device_count()
numpyro.set_host_device_count(n_cores)
print('N cores = ', n_cores)


# Make reproducible.
rng_key = jax.random.PRNGKey(12345)
np.random.seed(12345)

n_obs = 300
times = np.linspace(-0.22, 0.22, n_obs)
us = np.array([0.074, 0.193])
rs = np.array([0.1])
var_names = ['r0']
y_sigma = 70.e-6
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
#     # ln_prior += -0.5 * np.sum((params[1:] / 0.01)**2)
#
#     # Ln likelihood.
#     model = hlc_generate(_us=us, _rs=params, _times=times)
#     ln_like = -0.5 * np.sum((observed_fluxes - model)**2 / y_sigma**2
#                             + np.log(2 * np.pi * y_sigma**2))
#
#     return ln_like + ln_prior
#
#
# coords = np.array([0.1]) + 1.e-5 * np.random.randn(36, len(rs))
# sampler = emcee.EnsembleSampler(coords.shape[0], coords.shape[1], log_prob)
# state = sampler.run_mcmc(coords, 4000, progress=True)
# chain = sampler.get_chain(discard=1000, flat=True)
#
# emcee_data = az.from_emcee(sampler, var_names)
# print(az.summary(emcee_data, var_names, round_to=6).to_string())
#
# figure = corner.corner(chain, truths=rs, labels=var_names)
# plt.show()


def numpyro_model(t, obs_err, f_obs=None):
    """ Numpyro model. """
    r0 = numpyro.sample('r0', dist.Uniform(0.09, 0.11))
    # r1 = numpyro.sample('r1', dist.Uniform(-0.01, 0.01))
    # r2 = numpyro.sample('r2', dist.Uniform(-0.01, 0.01))
    # r3 = numpyro.sample('r3', dist.Uniform(-0.01, 0.01))
    # r4 = numpyro.sample('r4', dist.Uniform(-0.01, 0.01))

    # # Model evaluation: this is our custom JAX primitive.
    fs = harmonica_transit(
        t, 0., 3.735, 7.025, 86.9 * np.pi / 180., 0., 0.,
        'quadratic', us[0], us[1], r0)

    # Condition on the observations
    numpyro.sample('obs', dist.Normal(fs, obs_err), obs=f_obs)


# config = {"betas": LocScaleReparam(centered=0)}
# _rep_hs_model2 = numpyro.handlers.reparam(numpyro_model, config=config)

# Define NUTS kernel.
iinit = {'r0': 0.1}
nuts_kernel = NUTS(
    numpyro_model,
    dense_mass=True,
    max_tree_depth=10,
    target_accept_prob=0.80,
    init_strategy=init_to_value(
        values=iinit),
)

# Define HMC sampling strategy.
mcmc = MCMC(nuts_kernel, num_warmup=1000, num_samples=1000,
            num_chains=2, progress_bar=True)

mcmc.run(rng_key, times, y_sigma, f_obs=observed_fluxes)

numpyro_data = az.from_numpyro(mcmc)
print(az.summary(numpyro_data, var_names=var_names, round_to=6).to_string())

samples = mcmc.get_samples()
fig = corner.corner(samples, truths=rs, labels=var_names)
plt.show()

# todo: try (data - 1.) * 1.e3
# see https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html#checking-against-numerical-differences
# and https://jax.readthedocs.io/en/latest/notebooks/Custom_derivative_rules_for_Python_code.html#handling-non-differentiable-arguments
