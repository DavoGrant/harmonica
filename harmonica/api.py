import numpy as np

from harmonica import bindings


class HarmonicaTransit(object):
    """
    Harmonica transit class.

    Compute light curves for exoplanet transmission strings through
    parameterising the planet shape as a Fourier series.

    # Todo: update doc strings for ndarray or tensors not iterable. must match require gradient.
    # Todo: gradients.
    # Todo: update doc string for latest api args.
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
        Orbital inclination [radians].
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

    def __init__(self, times=None, ds=None, nus=None, t0=None,
                 period=None, a=None, inc=None, ecc=None, omega=None,
                 u=None, limb_dark_law='integers', r=None,
                 require_gradients=False, verbose=False):
        self._verbose = verbose

        # Orbital parameters.
        self._t0 = t0
        self._period = period
        self._a = a
        self._inc = inc
        self._ecc = ecc
        self._omega = omega

        # Stellar parameters.
        self._u = u
        self._limb_dark_mode = limb_dark_law
        self._limb_dark_updated = True

        # Planet parameters.
        self._r = r
        self._limb_map_updated = True

        # Evaluation arrays.
        if times is not None:
            self.times = np.ascontiguousarray(times, dtype=np.float64)
            self.ds = np.empty(times.shape, dtype=np.float64, order='C')
            self.nus = np.empty(times.shape, dtype=np.float64, order='C')
            self._orbit_updated = True
            self.lc = np.empty(times.shape, dtype=np.float64, order='C')
        else:
            self.ds = np.ascontiguousarray(ds, dtype=np.float64)
            self.nus = np.ascontiguousarray(nus, dtype=np.float64)
            self._orbit_updated = False
            self.lc = np.empty(ds.shape, dtype=np.float64, order='C')

        self._require_gradients = require_gradients
        if require_gradients:
            n_od = times.shape + (6,)
            n_lcd = times.shape + (4,)
            self.ds_grad = np.empty(n_od, dtype=np.float64, order='C')
            self.nus_grad = np.empty(n_od, dtype=np.float64, order='C')
            self.lc_grad = np.empty(n_lcd, dtype=np.float64, order='C')
        else:
            self.ds_grad = None
            self.nus_grad = None
            self.lc_grad = None

    def __repr__(self):
        return '<Harmonica transit: require_gradients={}>'.format(
            self._require_gradients)

    def set_orbit(self, t0=None, period=None, a=None, inc=None,
                  ecc=None, omega=None):
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
            Orbital inclination [radians].
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

    def get_transit_light_curve(self):
        """
        Get transit light curve.

        Returns
        -------
        if self._require_gradients == False:
            name : type
                Description of return object.
        else
            name : type
                Description of return object.

        """
        # Get orbit (if parameters updated).
        if self._orbit_updated:
            bindings.orbit(self._t0, self._period, self._a,
                           self._inc, self._ecc, self._omega,
                           self.times, self.ds, self.nus,
                           self.ds_grad, self.nus_grad,
                           require_gradients=self._require_gradients)

        import matplotlib.pyplot as plt
        plt.plot(self.times, self.ds_grad[:, 0])
        plt.show()

        # Get light curve.
        # NB. is odd term gauss-legendre faster as a whole,
        # or splitting into each term?

        # Reset update tracking flags to False. NB. this saves computation
        # time in subsequent calls to get_transit_light_curve() if some of
        # the parameters are the same as in the previous call.
        self._orbit_updated = False
        self._limb_dark_updated = False
        self._limb_map_updated = False

        return

    def get_planet_transmission_string(self):
        """
        Get harmonic transmission string evaluated at given angles.

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
