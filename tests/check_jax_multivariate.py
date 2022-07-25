import os
import jax
import corner
import numpyro
import arviz as az
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, init_to_median

from harmonica import bindings
from harmonica.jax import harmonica_transit_quad_ld


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
rng_key = jax.random.PRNGKey(123)
np.random.seed(123)

n_obs = 300
times = np.linspace(-0.22, 0.22, n_obs)
us = np.array([0.074, 0.193])
rs = np.array([0.1, -0.003, 0., 0.003, 0.])
var_names = ['r0', 'r1', 'r2', 'r3', 'r4']
y_sigma = 35.e-6
y_errs = np.random.normal(loc=0., scale=y_sigma, size=n_obs)

observed_fluxes = hlc_generate(_us=us, _rs=rs, _times=times)
observed_fluxes += y_errs


def numpyro_model(t, obs_err, f_obs=None):
    """ Numpyro model. """
    r0_tilde = numpyro.sample('r0_hat', dist.Normal(0., 1.))
    r0 = numpyro.deterministic('r0', 0.1 + r0_tilde * 0.01)

    # rh_frac = numpyro.sample('rh_frac', dist.MultivariateNormal(jnp.zeros(4), jnp.diag(jnp.ones(4) * 0.1)))
    rh_frac = numpyro.sample('rh_frac', dist.Normal(0.0, 0.1), sample_shape=(4,))
    rh = numpyro.deterministic('rh', rh_frac * r0)

    # # Model evaluation: this is our custom JAX primitive.
    fs = harmonica_transit_quad_ld(
        t, 0., 3.735, 7.025, 86.9 * np.pi / 180., 0., 0.,
        us[0], us[1], jnp.array([r0, rh[0], rh[1], rh[2], rh[3]]))

    # Condition on the observations
    numpyro.sample('obs', dist.Normal(fs, obs_err), obs=f_obs)


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
mcmc = MCMC(nuts_kernel, num_warmup=500, num_samples=500,
            num_chains=2, progress_bar=True)

mcmc.run(rng_key, times, y_sigma, f_obs=observed_fluxes)

numpyro_data = az.from_numpyro(mcmc)
# print(az.summary(numpyro_data, var_names=var_names, round_to=6).to_string())
print(az.summary(numpyro_data, round_to=6).to_string())

samples = mcmc.get_samples()
# fig = corner.corner(samples, truths=rs, labels=var_names)
fig = corner.corner(samples)
plt.show()

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