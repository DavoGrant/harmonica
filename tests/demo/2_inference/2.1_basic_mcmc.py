import emcee
import numpy as np
import arviz as az
import multiprocessing
import matplotlib.pyplot as plt
from harmonica import HarmonicaTransit


# Generate data.
np.random.seed(1)
n_obs = 600
times = np.linspace(-0.22, 0.22, n_obs)
us = np.array([0.062, 0.151])

rs = np.array([0.1])
# var_names = ['a_0']
# rs = np.array([0.1, -0.005, 0.])
# var_names = ['a_0', 'a_1', 'b_1']
rs = np.array([0.1, -0.005, 0., 0.005, 0.])
var_names = ['a_0', 'a_1', 'b_1', 'a_2', 'b_2']

y_sigma = 10.e-6
y_errs = np.random.normal(loc=0., scale=y_sigma, size=n_obs)

# Harmonica api example.
ht = HarmonicaTransit(times, pnl_c=50, pnl_e=500)
ht.set_orbit(t0=0., period=3.735, a=7.025, inc=86.9 * np.pi / 180.)
ht.set_stellar_limb_darkening(us, limb_dark_law='quadratic')
ht.set_planet_transmission_string(rs)
observed_fluxes = ht.get_transit_light_curve() + y_errs

# # Draw model.
# fig = plt.figure(figsize=(8, 6))
# ax1 = plt.subplot(1, 1, 1)
# ax1.errorbar(times, observed_fluxes, yerr=y_sigma, fmt=".k", capsize=0)
# ax1.set_xlabel('Time / days')
# ax1.set_ylabel('Relative flux')
# plt.tight_layout()
# plt.show()


# Todo: pnl tuning.
# Todo: priors.


def log_prob(params):
    """ Typical Gaussian likelihood. """
    # Ln prior.
    ln_prior = -0.5 * np.sum((params[0] / 0.05)**2)
    ln_prior += -0.5 * np.sum((params[1:] / 0.01)**2)

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
        state = sampler.run_mcmc(coords, 5000, progress=True)

    emcee_data = az.from_emcee(sampler, var_names)

    print(az.summary(emcee_data, var_names, round_to=8).to_string())

    plt.errorbar(times, observed_fluxes, yerr=y_sigma, fmt=".k", capsize=0)
    times = np.linspace(-0.22, 0.22, n_obs * 10)
    ht = HarmonicaTransit(times, pnl_c=50, pnl_e=500)
    ht.set_orbit(t0=0., period=3.735, a=7.025, inc=86.9 * np.pi / 180.)
    ht.set_stellar_limb_darkening(us, limb_dark_law='quadratic')

    chain = sampler.get_chain(discard=2000, flat=True)
    for sample in chain[np.random.randint(len(chain), size=50)]:
        ht.set_planet_transmission_string(sample)
        plt.plot(times, ht.get_transit_light_curve(), color="C0", alpha=0.1)

    plt.show()
