import numpy as np
from harmonica import bindings


class HarmonicaTransit(object):
    """
    Harmonica transit class.

    Compute transit light curves for a given transmission string
    through parameterising the planet shape as a Fourier series.

    Parameters
    ----------
    times : ndarray, optional
        1D array of model evaluation times [days].
    ds : ndarray, optional
        1D array of planet-star separations [stellar radii].
    nus : ndarray, optional
        1D array of planet-star angles [radians].
    require_gradients : bool, optional
        Compute derivatives of model with respect to input parameters.
        This mode should be enabled when using a gradient-based
        inference methods, such Hamiltonian Monte Carlo.
    pnl_c and pnl_e : int, optional
        Number of legendre roots used to approximate the integrals
        with no closed form solution. pnl_c corresponds to when the
        planet lies entirely inside the stellar disc, and pnl_e
        corresponds to when the planet intersects the stellar limb.

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
    Some notes about where the method is described.

    For light curve modelling, either times or ds and nus should be
    provided.

    Perhaps a further note about the limb darkening for common uses.
    For example a quadratic law; u=[0.1, 0.2], limb_dark_law=`integers`,
    or a 4-param non-linear law u=[0.1, 0.2, 0.1, 0.2], limb_dark_law=`
    half-integers`.

    Perhaps a further note about the r coeffs intuition. If only r=[r0]
    is given then r0 is the radius of a circular planet.

    Perhaps a further note about the use of the require_gradients arg.

    """

    def __init__(self, times=None, ds=None, nus=None,
                 require_gradients=False, pnl_c=50, pnl_e=500):
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

        # todo: update api for new structure.
        # todo: remove redundancy in psotion and derivative arrays.
        # todo: no require gradients flag anymore.

        # Evaluation arrays.
        if times is not None:
            self.times = np.ascontiguousarray(times, dtype=np.float64)
            self.ds = np.empty(times.shape, dtype=np.float64, order='C')
            self.nus = np.empty(times.shape, dtype=np.float64, order='C')
            self._orbit_updated = True
            self.lc = np.empty(times.shape, dtype=np.float64, order='C')
        elif ds is not None and nus is not None:
            self.ds = np.ascontiguousarray(ds, dtype=np.float64)
            self.nus = np.ascontiguousarray(nus, dtype=np.float64)
            self._orbit_updated = False
            self.lc = np.empty(ds.shape, dtype=np.float64, order='C')
        else:
            return

        self._require_gradients = require_gradients
        n_od = self.ds.shape + (6,)
        self.ds_grad = np.zeros(n_od, dtype=np.float64, order='C')
        self.nus_grad = np.zeros(n_od, dtype=np.float64, order='C')
        n_lcd = self.ds.shape + (6 + 3 + 5,)
        self.lc_grad = np.zeros(n_lcd, dtype=np.float64, order='C')

        # Precision: number of legendre roots at centre and edges.
        self._pnl_c = pnl_c
        self._pnl_e = pnl_e

    def __repr__(self):
        return '<Harmonica transit: require_gradients={}>'.format(
            self._require_gradients)

    def set_orbit(self, t0=None, period=None, a=None, inc=None,
                  ecc=0., omega=0.):
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
        self._orbit_updated = True

    def set_stellar_limb_darkening(self, u=None, limb_dark_law='quadratic'):
        """
        Set/update stellar limb darkening parameters.

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
        self._u = u
        if limb_dark_law == 'quadratic':
            self._ld_mode = 0
        else:
            self._ld_mode = 1

    def set_planet_transmission_string(self, r=None):
        """
        Set/update planet transmission string parameters.

        Parameters
        ----------
        r : ndarray (N,) or (N, M)
            Transmission string coefficients. 1D array of N Fourier
            coefficients that specify the planet radius as a function
            of angle in the sky-plane.

            ``r_{\rm{p}}(\theta) = \sum_{n=0}^N a_n \cos{(n \theta)}
            + \sum_{n=1}^N b_n \csin{(n \theta)}``

            The input array is given as r=[a_0, a_1, b_1, a_2, b_2,..].
            For time-dependent transmission strings, use a 2D array
            with N Fourier coefficients and M time steps, where M is
            equal to the number of model evaluation epochs.

        """
        self._r = r

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
            self._orbit_updated = False

        # Get light curve.
        bindings.light_curve(self._ld_mode, self._u, self._r,
                             self.ds, self.nus, self.lc,
                             self.ds_grad, self.nus_grad, self.lc_grad,
                             self._pnl_c, self._pnl_e,
                             require_gradients=self._require_gradients)

        return np.copy(self.lc)

    def get_planet_transmission_string(self, theta):
        """
        Get transmission string evaluated at an array of angles.

        Parameters
        ----------
        theta : ndarray,
            1D array of angles at which to evaluate the transmission
            string.

        Returns
        -------
        r_p : ndarray,
            The transmission string, ``r_{\rm{p}}(\theta)``, evaluated
            at the provided thetas.

        """
        transmission_string = np.empty(theta.shape, dtype=np.float64, order='C')
        bindings.transmission_string(self._r, theta, transmission_string)
        return transmission_string

    def get_precision_estimate(self):
        """
        Get light curve precision estimate.

        Returns
        -------
        residuals : ndarray
            Difference between light curve generated at user set precision
            and the light curve at max precision.

        """
        # Get light curve for user set precision.
        lc_user = self.get_transit_light_curve()

        # Get light curve at max precision.
        lc_best = np.empty(lc_user.shape, dtype=np.float64, order='C')
        bindings.light_curve(self._ld_mode, self._u, self._r,
                             self.ds, self.nus, lc_best,
                             self.ds_grad, self.nus_grad, self.lc_grad,
                             500, 500, False)

        return lc_user - lc_best
