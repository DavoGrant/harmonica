import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from exotic_ld import StellarLimbDarkening

from harmonica import HarmonicaTransit


# Make reproducible.
for i in range(1000):
    print(i)
    np.random.seed(i)

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


    def model(_, *params):
        params = np.array(params)
        _r = np.concatenate([np.array([params[0]]), params[1:] * params[0]])
        ht.set_planet_transmission_string(_r)
        light_curve = ht.get_transit_light_curve()

        return light_curve


    popt, pcov = curve_fit(
        model, times, observed_fluxes, sigma=np.ones(observed_fluxes.shape) * y_sigma,
        p0=[0.125, -0., 0., 0., 0.0001],
        method='lm')


    # fig, ax1 = plt.subplots(1, 1, figsize=(12, 5))
    #
    # ax1.errorbar(times, observed_fluxes, yerr=y_sigma, color="#000000", capsize=0, alpha=0.5)
    # _r = np.concatenate([np.array([popt[0]]), popt[1:] * popt[0]])
    # ht.set_planet_transmission_string(_r)
    # light_curve = ht.get_transit_light_curve()
    # ax1.plot(times, light_curve, color='#FFD200', alpha=1.)
    #
    # ax1.set_xlabel('Time / days', fontsize=15)
    # ax1.set_ylabel('Relative flux', fontsize=15)
    # plt.tight_layout()
    # plt.show()


    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.set_aspect('equal', 'box')
    labelled = False
    theta = np.linspace(-np.pi, np.pi, 1000)

    ht = HarmonicaTransit()
    _r = np.concatenate([np.array([popt[0]]), popt[1:] * popt[0]])
    ht.set_planet_transmission_string(_r)
    transmission_string = ht.get_planet_transmission_string(theta)
    ax1.plot(transmission_string * np.cos(theta), transmission_string * np.sin(theta), color='#FFD200', alpha=1.)
    ax2.plot(theta * 180. / np.pi, transmission_string, color='#FFD200', alpha=1.)

    ht.set_planet_transmission_string(rs)
    transmission_string = ht.get_planet_transmission_string(theta)
    ax1.plot(transmission_string * np.cos(theta), transmission_string * np.sin(theta), lw=1.5, c='#000000')
    ax2.plot(theta * 180. / np.pi, transmission_string, c='#000000', lw=1.5, label='True transmission string')

    ax1.plot(rs[0] * np.cos(theta), rs[0] * np.sin(theta), lw=1.5, c='#d5d6d2', ls='--')
    ax2.plot(theta * 180. / np.pi, np.ones(theta.shape[0]) * rs[0], c='#d5d6d2', lw=1.5, ls='--', label='Circle')

    ax2.legend(loc='upper center')
    ax1.set_xlabel('x / stellar radii', fontsize=15)
    ax1.set_ylabel('y / stellar radii', fontsize=15)
    ax2.set_xlabel('$\\theta$ / degrees', fontsize=15)
    ax2.set_ylabel('$r_{\\rm{p}}$ / stellar radii', fontsize=15)
    plt.tight_layout()
    plt.show()
