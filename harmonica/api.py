import numpy as np
from harmonica import bindings


class HarmonicaTransit(object):
    """ Harmonica transit class.

    Compute transit light curves for a given transmission string
    through parameterising the planet shape as a Fourier series.

    Parameters
    ----------
    times : ndarray, optional
        1D array of model evaluation times [days].
    pnl_c and pnl_e : int, optional
        Number of legendre roots used to approximate the integrals
        with no closed form solution. pnl_c corresponds to when the
        planet lies entirely inside the stellar disc, and pnl_e
        corresponds to when the planet intersects the stellar limb.
        Allowed values = {10, 20, 100, 200, 500}. Use
        get_precision_estimate() to check model precision.

    Methods
    -------
    set_orbit()
    set_stellar_limb_darkening()
    set_planet_transmission_string()
    get_transit_light_curve()
    get_planet_transmission_string()
    get_precision_estimate()

    Notes
    -----
    The algorithm is detailed in Grant and Wakeford 2022.
    Todo: add link.

    """

    def __init__(self, times=None, pnl_c=20, pnl_e=50):
        # Orbital parameters.
        self._t0 = None
        self._period = None
        self._a = None
        self._inc = None
        self._ecc = None
        self._omega = None

        # Stellar parameters.
        self._u = None
        self._ld_mode = None

        # Planet parameters.
        self._r = None
        self._time_dep_strings = False

        # Precision: number of legendre roots at centre and edges.
        self._pnl_c = pnl_c
        self._pnl_e = pnl_e

        # Evaluation arrays.
        if times is not None:
            self.times = np.ascontiguousarray(times, dtype=np.float64)
            self.fs = np.empty(times.shape, dtype=np.float64)
        else:
            self.times = None
            self.fs = None

    def __repr__(self):
        if self.times is not None:
            return '<Harmonica transit: {} eval points>'.format(
                self.times.shape[0])
        else:
            return '<Harmonica transit: transmission strings mode>'

    def set_orbit(self, t0=None, period=None, a=None, inc=None,
                  ecc=0., omega=0.):
        """ Set/update orbital parameters.

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
        ecc : float, optional
            Eccentricity [], 0 <= ecc < 1. Default=0.
        omega : float, optional
            Argument of periastron [radians]. Default=0.

        """
        self._t0 = t0
        self._period = period
        self._a = a
        self._inc = inc
        self._ecc = ecc
        self._omega = omega

    def set_stellar_limb_darkening(self, u=None, limb_dark_law='quadratic'):
        """ Set/update stellar limb darkening parameters.

        Parameters
        ----------
        u : ndarray,
            1D array of limb-darkening coefficients which correspond to
            the limb-darkening law specified by limb_dark_law. The
            quadratic law requires two coefficients and the non-linear
            law requires four coefficients.
        limb_dark_law : string, optional; `quadratic` or `non-linear`
            The stellar limb darkening law. Default=`quadratic`.

        """
        self._u = np.ascontiguousarray(u, dtype=np.float64)
        if limb_dark_law == 'quadratic':
            self._ld_mode = 0
        else:
            self._ld_mode = 1

    def set_planet_transmission_string(self, r=None):
        """ Set/update planet transmission string parameters.

        Parameters
        ----------
        r : ndarray (N,) or (M, N)
            Transmission string coefficients. 1D array of N Fourier
            coefficients that specify the planet radius as a function
            of angle in the sky-plane.

            ``r_{\rm{p}}(\theta) = \sum_{n=0}^N a_n \cos{(n \theta)}
            + \sum_{n=1}^N b_n \csin{(n \theta)}``

            The input array is given as r=[a_0, a_1, b_1, a_2, b_2,..].
            For time-dependent transmission strings, use a 2D array
            with M time steps and N Fourier coefficients, where M is
            equal to the number of model evaluation times.

        """
        self._r = np.ascontiguousarray(r, dtype=np.float64)
        if r.ndim == 1:
            self._time_dep_strings = False
        else:
            self._time_dep_strings = True

    def get_transit_light_curve(self):
        """ Get transit light curve.

        Returns
        -------
        fluxes : ndarray (M,),
            The transit light curve fluxes evaluated at M times.

        """
        if not self._time_dep_strings:
            bindings.light_curve(
                self._t0, self._period, self._a, self._inc, self._ecc,
                self._omega, self._ld_mode, self._u, self._r,
                self.times, self.fs, self._pnl_c, self._pnl_e)
        else:
            for i in range(self.times.shape[0]):
                t = np.array([self.times[i]])
                f = np.array([self.fs[i]])
                bindings.light_curve(
                    self._t0, self._period, self._a, self._inc, self._ecc,
                    self._omega, self._ld_mode, self._u, self._r[i],
                    t, f, self._pnl_c, self._pnl_e)
                self.fs[i] = f

        return np.copy(self.fs)

    def get_planet_transmission_string(self, theta):
        """ Get transmission string evaluated at an array of angles
            around the planet's terminator.

        Parameters
        ----------
        theta : ndarray,
            1D array of angles at which to evaluate the transmission
            string.

        Returns
        -------
        if r.ndim == 1:
            r_p : ndarray (N,),
                The transmission string, ``r_{\rm{p}}(\theta)``, evaluated
                at N thetas.
        elif r.ndim == 2:
            r_p : ndarray (M, N),
                The transmission strings, ``r_{\rm{p}}(\theta)``, each
                M strings evaluated at N provided thetas.

        """
        theta = np.ascontiguousarray(theta, dtype=np.float64)
        if not self._time_dep_strings:
            r_p = np.empty(theta.shape, dtype=np.float64)
            bindings.transmission_string(self._r, theta, r_p)
        else:
            r_p = np.empty((self._r.shape[0], theta.shape[0]), dtype=np.float64)
            for i in range(r_p.shape[0]):
                bindings.transmission_string(self._r[i], theta, r_p[i])

        return r_p

    def get_precision_estimate(self):
        """ Get light curve precision estimate.

        Returns
        -------
        residuals : ndarray
            Difference between light curve generated at user set precision
            and the light curve at max precision.

        """
        # Get light curve for user set precision.
        lc_user = self.get_transit_light_curve()

        # Get light curve at max precision.
        lc_best = np.empty(lc_user.shape, dtype=np.float64)
        bindings.light_curve(self._t0, self._period, self._a,
                             self._inc, self._ecc, self._omega,
                             self._ld_mode, self._u, self._r,
                             self.times, lc_best,
                             500, 500)

        return lc_user - lc_best
