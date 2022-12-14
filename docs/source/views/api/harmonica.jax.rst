harmonica.jax
-------------

.. currentmodule:: harmonica.jax

.. automodule:: harmonica.jax

.. function:: harmonica_transit_quad_ld(times, t0, period, a, inc, ecc=0., omega=0., u1=0., u2=0., r=jnp.array([0.1]))

    Harmonica transits with jax -- quadratic limb darkening.

    :param times: 1D array of model evaluation times [days].
    :type times: ndarray

    :param t0: Time of transit [days].
    :type t0: float

    :param period: Orbital period [days].
    :type period: float

    :param a: Semi-major axis [stellar radii].
    :type a: float

    :param inc: Orbital inclination [radians].
    :type inc: float

    :param ecc: Eccentricity [], 0 <= ecc < 1. Default=0.
    :type ecc: float, optional

    :param omega: Argument of periastron [radians]. Default=0.
    :type omega: float, optional

    :param u1: Quadratic limb-darkening coefficient.
    :type u1: float.

    :param u2: Quadratic limb-darkening coefficient.
    :type u2: float.

    :param r: Transmission string coefficients. 1D array of N Fourier coefficients that specify the planet radius as a function of angle in the sky-plane. The length of r must be odd, and the final two coefficients must not both be zero.

              .. math::
                  r_{\rm{p}}(\theta) = \sum_{n=0}^N a_n \cos{(n \theta)} + \sum_{n=1}^N b_n \sin{(n \theta)}

             The input array is given as r=[a_0, a_1, b_1, a_2, b_2,..].
    :type r: ndarray

    :return: Normalised transit light curve fluxes [].
    :rtype: array

.. function:: harmonica_transit_nonlinear_ld(times, t0, period, a, inc, ecc=0., omega=0., u1=0., u2=0., u3=0., u4=0., r=jnp.array([0.1]))

    Harmonica transits with jax -- non-linear limb darkening.

    :param times: 1D array of model evaluation times [days].
    :type times: ndarray

    :param t0: Time of transit [days].
    :type t0: float

    :param period: Orbital period [days].
    :type period: float

    :param a: Semi-major axis [stellar radii].
    :type a: float

    :param inc: Orbital inclination [radians].
    :type inc: float

    :param ecc: Eccentricity [], 0 <= ecc < 1. Default=0.
    :type ecc: float, optional

    :param omega: Argument of periastron [radians]. Default=0.
    :type omega: float, optional

    :param u1: Non-linear limb-darkening coefficient.
    :type u1: float.

    :param u2: Non-linear limb-darkening coefficient.
    :type u2: float.

    :param u3: Non-linear limb-darkening coefficient.
    :type u3: float.

    :param u4: Non-linear limb-darkening coefficient.
    :type u4: float.

    :param r: Transmission string coefficients. 1D array of N Fourier coefficients that specify the planet radius as a function of angle in the sky-plane. The length of r must be odd, and the final two coefficients must not both be zero.

              .. math::
                  r_{\rm{p}}(\theta) = \sum_{n=0}^N a_n \cos{(n \theta)} + \sum_{n=1}^N b_n \sin{(n \theta)}

             The input array is given as r=[a_0, a_1, b_1, a_2, b_2,..].
    :type r: ndarray

    :return: Normalised transit light curve fluxes [].
    :rtype: array
