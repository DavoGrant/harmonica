import numpy as np

from harmonica import bindings


class HarmonicLimbMap(object):
    """
    Harmonic limb mapping class.

    Compute light curves for exoplanet transmission mapping through
    parameterising the planet shape as a Fourier series.

    # Todo: update doc strings for ndarray or tensors not iterable. must match require gradient.
    # Todo: gradients.
    # Todo: finite exposure time?
    # Todo: light travel time?
    # Todo: update readme labels with this repo links?

    Parameters
    ----------
    t0 : float
        Time of transit [days].
    period : float
        Orbital period [days].
    a : float
        Semi-major axis [stellar radii].
    inc : float
        Orbital inclination [radians]. Only one of inc and b is
        required.
    b : float
        Impact parameter []. Only one of inc and b is required.
    ecc : float
        Eccentricity [], 0 <= ecc < 1.
    omega : float
        Argument of periastron [radians]. If ecc is not None or 0.
        then omega must also be given.
    u : (N,) array_like
        Limb-darkening coefficients. 1d array of coefficients that
        correspond to the limb darkening law specified by the
        limb_dark_law.
    limb_dark_law : string; `integers` or `half-integers`
        The limb darkening law. `integers` corresponds to ``I/I_0 =
        1 - \sum_{n=1}^N u_n (1 - \mu)^n``, or `half-integers`,
        corresponds to ``I/I_0 = 1 - \sum_{n=1}^N u_n (1 - \mu^n)``,
        where N is the length of u. Default is 'integers'.
    r :  (N,) or (N, M) array_like
        Harmonic limb map coefficients. 1D array of N Fourier
        coefficients that specify the planet radius as a function
        of angle in the sky-plane. Coefficients correspond to
        ``r_{\rm{p}}(\theta) = \sum_{n=0}^N a_n \cos{(n \theta)}
        + \sum_{n=1}^N b_n \csin{(n \theta)}`` where the resulting
        input is r=[a_0, a_1, b_1, a_2, b_2,..]. For time-dependent
        planet shapes, use a 2d array with N Fourier coefficients
        and M time steps, where M is equal to the number of model
        evaluation epochs.

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
        self._t0 = t0
        self._period = period
        self._a = a
        self._inc = inc
        self._b = b
        self._ecc = ecc
        self._omega = omega
        self._orbit_updated = True

        # Stellar parameters.
        self._u = u
        self._limb_dark_mode = limb_dark_law
        self._limb_dark_updated = True

        # Planet parameters.
        self._r = r
        self._limb_map_updated = True

        self.xs = None
        self.ys = None
        self.phis = None
        self.lc = None
        self._require_gradients = require_gradients
        self._verbose = verbose

    def __repr__(self):
        return '<Harmonic limb mapper: require_gradients={}>'.format(
            self._require_gradients)

    def set_orbit(self, t0=None, period=None, a=None, inc=None,
                  b=None, ecc=None, omega=None):
        """
        Set/update orbital parameters.

        Parameters
        ----------
        t0 : float
            Time of transit [days].
        period : float
            Orbital period [days].
        a : float
            Semi-major axis [stellar radii].
        inc : float
            Orbital inclination [radians]. Only one of inc and b is
            required.
        b : float
            Impact parameter []. Only one of inc and b is required.
        ecc : float
            Eccentricity [], 0 <= ecc < 1.
        omega : float
            Argument of periastron [radians]. If ecc is not None or 0.
            then omega must also be given.

        """
        self._t0 = t0
        self._period = period
        self._a = a
        self._inc = inc
        self._b = b
        self._ecc = ecc
        self._omega = omega
        self._orbit_updated = True

    def set_stellar_limb_darkening(self, u=None, limb_dark_law='integers'):
        """
        Set/update stellar limb darkening parameters.

        Parameters
        ----------
        u :  (N,) array_like
            Limb-darkening coefficients. 1D array of coefficients that
            correspond to the limb darkening law specified by the
            limb_dark_law.
        limb_dark_law : string; `integers` or `half-integers`
            The limb darkening law. `integers` corresponds to ``I/I_0 =
            1 - \sum_{n=1}^N u_n (1 - \mu)^n``, or `half-integers`,
            corresponds to ``I/I_0 = 1 - \sum_{n=1}^N u_n (1 - \mu^n)``,
            where N is the length of u. Default is 'integers'.

        """
        self._u = u
        self._limb_dark_mode = limb_dark_law
        self._limb_dark_updated = True

    def set_planet_harmonic_limb_map(self, r=None):
        """
        Set/update planet harmonic limb map parameters.

        Parameters
        ----------
        r :  (N,) or (N, M) array_like
            Harmonic limb map coefficients. 1D array of N Fourier
            coefficients that specify the planet radius as a function
            of angle in the sky-plane. Coefficients correspond to
            ``r_{\rm{p}}(\theta) = \sum_{n=0}^N a_n \cos{(n \theta)}
            + \sum_{n=1}^N b_n \csin{(n \theta)}`` where the resulting
            input is r=[a_0, a_1, b_1, a_2, b_2,..]. For time-dependent
            planet shapes, use a 2d array with N Fourier coefficients
            and M time steps, where M is equal to the number of model
            evaluation epochs.

        """
        self._r = r
        self._limb_map_updated = True

    def get_transit_light_curve(self, times=None, positions=None, phis=None):
        """
        Get transit light curve.

        Parameters
        ----------
        times : type
            Description of parameter.

        positions : type
            Description of parameter.

        phis : type
            Description of parameter.

        Returns
        -------
        if self._require_gradients == False:
            name : type
                Description of return object.
        else
            name : type
                Description of return object.

        """
        # Get orbit (if updated).
        if self._orbit_updated:
            np_in = np.array([1, 2, 3, 4])
            np_out = np.array([5, 6, 7, 8])
            np_in = np.ascontiguousarray(np_in, dtype=np.float64)
            np_out = np.ascontiguousarray(np_out, dtype=np.float64)
            bindings.orbit(1., 0., self._require_gradients, np_in, np_out)
            print(np_in)
            print(np_out)

        # Get light curve.

        # Reset update tracking flags to False. NB. this saves computation
        # time in subsequent calls to get_transit_light_curve() if some of
        # the parameters are the same as in the previous call.
        self._orbit_updated = False
        self._limb_dark_updated = False
        self._limb_map_updated = False

        return

    def get_planet_harmonic_limb_map(self):
        """
        Get harmonic limb map.

        Parameters
        ----------
        name : type
            Description of parameter.

        Returns
        -------
        name : type
            Description of return object.

        """
        return

    def get_precision_estimate(self):
        """
        Get light curve precision estimates.

        Parameters
        ----------
        name : type
            Description of parameter.

        Returns
        -------
        name : type
            Description of return object.

        """
        return
