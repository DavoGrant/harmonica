import emcee
import numpy as np
import arviz as az
import multiprocessing
import matplotlib.pyplot as plt
from harmonica import HarmonicaTransit


# Generate data.
np.random.seed(1)
n_obs = 1000
times = np.linspace(-0.22, 0.22, n_obs)
us = np.array([0.062, 0.151])
rs = np.array([0.1, -0.006, 0., 0.003, 0.])
var_names = ['a_0', 'a_1', 'b_1', 'a_2', 'b_2']
y_sigma = 100.e-6
y_errs = np.random.normal(loc=0., scale=y_sigma, size=n_obs)

ht = HarmonicaTransit(times, pnl_c=20, pnl_e=20)
ht.set_orbit(t0=0., period=3.735, a=7.025, inc=86.9 * np.pi / 180.)
ht.set_stellar_limb_darkening(us, limb_dark_law='quadratic')
ht.set_planet_transmission_string(rs)
observed_fluxes = ht.get_transit_light_curve() + y_errs

# # Plot synthetic data.
# fig = plt.figure(figsize=(8, 6))
# ax1 = plt.subplot(1, 1, 1)
# ax1.errorbar(times, observed_fluxes, yerr=y_sigma, fmt=".k", capsize=0)
# ax1.set_xlabel('Time / days')
# ax1.set_ylabel('Relative flux')
# plt.show()


def log_prob(params):
    """ Typical Gaussian likelihood. """
    # Ln prior.
    ln_prior = -0.5 * np.sum((params[0] / 0.05)**2)
    ln_prior += -0.5 * np.sum((params[1:] / 0.05)**2)

    # Ln likelihood.
    ht.set_planet_transmission_string(params)
    model = ht.get_transit_light_curve()
    ln_like = -0.5 * np.sum((observed_fluxes - model)**2 / y_sigma**2
                            + np.log(2 * np.pi * y_sigma**2))

    return ln_like + ln_prior


if __name__ == '__main__':

    print('Using {} cores.'.format(multiprocessing.cpu_count()))
    coords = np.array([0.1, 0., 0., 0., 0.]) + 1.e-5 * np.random.randn(12, len(rs))
    with multiprocessing.Pool() as pool:
        sampler = emcee.EnsembleSampler(coords.shape[0], coords.shape[1], log_prob, pool=pool)
        state = sampler.run_mcmc(coords, 10000, progress=True)
    chain = sampler.get_chain(discard=5000, flat=True)

    # Display sampling metrics.
    emcee_data = az.from_emcee(sampler, var_names)
    print(az.summary(emcee_data, var_names, round_to=8).to_string())

    # Plot transit light curve fits.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.errorbar(times, observed_fluxes, yerr=y_sigma, color='#000000', markersize=2, capsize=0)
    ht.set_planet_transmission_string(np.median(chain, axis=0))
    ax2.scatter(times, ht.get_transit_light_curve() - observed_fluxes, color='#000000', s=2)
    for sample in chain[np.random.randint(len(chain), size=50)]:
        ht.set_planet_transmission_string(sample)
        ax1.plot(times, ht.get_transit_light_curve(),
                 color='C0', alpha=0.1)
    ax1.set_xlabel('Time / days')
    ax1.set_ylabel('Relative flux')
    ax2.set_xlabel('Time / days')
    ax2.set_ylabel('Residuals')
    plt.tight_layout()
    plt.show()

    # Plot inferred transmission string.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.set_aspect('equal', 'box')
    labelled = False
    theta = np.linspace(-np.pi, np.pi, 1000)
    for sample in chain[np.random.randint(len(chain), size=50)]:
        ht.set_planet_transmission_string(sample)
        transmission_string = ht.get_planet_transmission_string(theta)
        ax1.plot(transmission_string * np.cos(theta), transmission_string * np.sin(theta), color='C0', alpha=0.1)
        ax2.plot(theta * 180. / np.pi, transmission_string, color='C0', alpha=0.1)
    ht.set_planet_transmission_string(rs)
    ax1.plot(transmission_string * np.cos(theta), transmission_string * np.sin(theta), lw=1.5, c='#000000')
    ax2.plot(theta * 180. / np.pi, transmission_string, c='#000000', lw=1.5, label='True transmission string')
    ax1.plot(rs[0] * np.cos(theta), rs[0] * np.sin(theta), lw=1.5, c='#d5d6d2', ls='--')
    ax2.plot(theta * 180. / np.pi, np.ones(theta.shape[0]) * rs[0], c='#d5d6d2', lw=1.5, ls='--', label='Circle')

    ht.set_planet_transmission_string(np.median(chain, axis=0))
    transmission_string = ht.get_planet_transmission_string(theta)
    ax2.plot(theta * 180. / np.pi, transmission_string, color='C0', alpha=1.0)

    ax2.legend(loc='upper center')
    ax1.set_xlabel('x / stellar radii')
    ax1.set_ylabel('y / stellar radii')
    ax2.set_xlabel('$\\theta$ / degrees')
    ax2.set_ylabel('$r_{\\rm{p}}$ / stellar radii')
    plt.tight_layout()
    plt.show()
