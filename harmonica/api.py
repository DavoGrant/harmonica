import numpy as np

from harmonica import bindings


class HarmonicLimbMap(object):
    """
    Harmonic limb mapping class.

    Compute light curves for exoplanet transmission mapping through
    parameterising the planet shape as a Fourier series.

    Parameters
    ----------
    t0 : float
        Time of transit [days].
    period : float
        Orbital period [days].
    a : float
        Semi-major axis [stellar radii].
    inc : float
        Orbital inclination [radians].
    b : float
        Impact parameter []. Only one of inc and b is required.
    ecc : float
        Eccentricity [], 0 <= ecc < 1.
    omega : float
        Argument of periastron [radians]. If ecc is not None or 0.
        then omega must also be given.
    u :  (N,) array_like
        Limb-darkening coefficients. 1D array of coefficients that
        correspond to the limb darkening law specified by the
        limb_dark_law.
    limb_dark_law : string; `integers` or `half-integers`
        The limb darkening law. `integers` corresponds to ``I/I_0 =
        1 - \sum_{n=1}^N u_n (1 - \mu)^n``, or `half-integers`,
        corresponds to ``I/I_0 = 1 - \sum_{n=1}^N u_n (1 - \mu^n)``,
        where N is the length of u. Default is 'integers'.
    r :  (M,) array_like
         Harmonic limb map coefficients. 1D array of Fourier
         coefficients that specify the planet radius as a function
         of angle in the sky-plane.

    Methods
    -------
    set_orbit()
    set_stellar_limb_darkening()
    set_planet_harmonic_limb_map()
    get_transit_light_curve()
    get_planet_harmonic_limb_map()
    get_precision_estimate()

    Notes
    -----
    Some notes about where the method is described.
    Perhaps a further note about the limb darkening for common uses.
    For example a quadratic law; u=[0.1, 0.2], limb_dark_law=`integers`,
    or a 4-param non-linear law u=[0.1, 0.2, 0.1, 0.2], limb_dark_law=`
    half-integers`.
    Perhaps a further note about the r coeffs intuition. If only r=[r0]
    is given then r0 is the radius of a circular planet.
    Perhaps a further note about the use of the require_gradients arg.

    """

    def __init__(self, t0=None, period=None, a=None, inc=None, b=None,
                 ecc=None, omega=None, u=None, limb_dark_law='integers',
                 r=None, require_gradients=False, verbose=False):
        # Orbital parameters.
        self.t0 = t0
        self.period = period
        self.a = a
        self.inc = inc
        self.b = b
        self.ecc = ecc
        self.omega = omega

        # Stellar parameters.
        self.u = u
        self.limb_dark_mode = limb_dark_law

        # Planet parameters.
        self.r = r

        self._require_gradients = require_gradients
        self._verbose = verbose

    def __repr__(self):
        return '<Harmonic limb mapper: require_gradients={}>'.format(
            self._require_gradients)

    def set_orbit(self):
        """ Set/update orbital parameters. """
        return

    def set_stellar_limb_darkening(self):
        """ Set/update stellar limb darkening parameters. """
        return

    def set_planet_harmonic_limb_map(self):
        """ Set/update planet harmonic limb map parameters. """
        return

    def get_transit_light_curve(self):
        """ Get transit light curve. """
        return

    def get_planet_harmonic_limb_map(self):
        """ Get harmonic limb map. """
        return

    def get_precision_estimate(self):
        """ Get light curve precision estimates. """
        return
