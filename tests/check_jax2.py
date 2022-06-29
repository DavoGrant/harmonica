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
rng_key = jax.random.PRNGKey(123)
np.random.seed(1)

n_obs = 200
times = np.linspace(-0.22, 0.22, n_obs)
us = np.array([0.074, 0.193])
rs = np.array([0.1, -0.003, 0., 0.003, 0.])
var_names = ['r0', 'r1', 'r2', 'r3', 'r4']
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
#     ln_prior += -0.5 * np.sum((params[1:] / 0.005)**2)
#
#     # Ln likelihood.
#     model = hlc_generate(_us=us, _rs=params, _times=times)
#     ln_like = -0.5 * np.sum((observed_fluxes - model)**2 / y_sigma**2
#                             + np.log(2 * np.pi * y_sigma**2))
#
#     return ln_like + ln_prior
#
#
# coords = np.array([0.1, 0., 0., 0., 0.]) + 1.e-5 * np.random.randn(18, len(rs))
# sampler = emcee.EnsembleSampler(coords.shape[0], coords.shape[1], log_prob)
# state = sampler.run_mcmc(coords, 2000, progress=True)
# chain = sampler.get_chain(discard=1000, flat=True)
#
# emcee_data = az.from_emcee(sampler, var_names)
# print(az.summary(emcee_data, var_names, round_to=6).to_string())
#
# figure = corner.corner(chain, truths=rs, labels=var_names)
# plt.show()


def numpyro_model(t, obs_err, f_obs=None):
    """ Numpyro model. """
    # Parameters
    r0 = numpyro.sample('r0', dist.Normal(0.1, 0.01))
    r1 = numpyro.sample('r1', dist.Normal(0., 0.005))
    r2 = numpyro.sample('r2', dist.Normal(0., 0.005))
    r3 = numpyro.sample('r3', dist.Normal(0., 0.005))
    r4 = numpyro.sample('r4', dist.Normal(0., 0.005))

    # u1 = numpyro.sample('u1', dist.Normal(0.074, 0.1))
    # u2 = numpyro.sample('u2', dist.Normal(0.193, 0.1))

    # # Model evaluation: this is our custom JAX primitive.
    fs = harmonica_transit(
        t, 0., 3.735, 7.025, 86.9 * np.pi / 180., 0., 0.,
        'quadratic', us[0], us[1], r0, r1, r2, r3, r4)

    # Condition on the observations
    numpyro.sample('obs', dist.Normal(fs, obs_err), obs=f_obs)


# Define NUTS kernel.
iinit = {'r0': 0.1, 'r1': -0.003, 'r2': 0., 'r3': 0.003, 'r4': 0.}
nuts_kernel = NUTS(
    numpyro_model,
    # dense_mass=True,
    # step_size=0.1,
    # adapt_step_size=False,
    # forward_mode_differentiation=True,
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

# todo: check gradients working with finite diff.
# todo: check gradients shape for jacobian
# todo: check finite diffs on hello world first perhasp,
# see https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html#checking-against-numerical-differences
# and https://jax.readthedocs.io/en/latest/notebooks/Custom_derivative_rules_for_Python_code.html#handling-non-differentiable-arguments
