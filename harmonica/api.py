import numpy as np

from harmonica import bindings


class HarmonicaTransit(object):
    """
    Harmonica transit class.

    Compute transit light curves for a given transmission string
    through parameterising the planet shape as a Fourier series.

    Parameters
    ----------
    times : type
        Description [units].
    ds : type
        Description [units].

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
        n_od = self.ds.shape + (6,)
        self.ds_grad = np.empty(n_od, dtype=np.float64, order='C')
        self.nus_grad = np.empty(n_od, dtype=np.float64, order='C')
        n_lcd = self.ds.shape + (6 + 3 + 5,)
        self.lc_grad = np.empty(n_lcd, dtype=np.float64, order='C')

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
        ecc : float
            Eccentricity [], 0 <= ecc < 1.
        omega : float
            Argument of periastron [radians]. If ecc is not 0.
            then omega must also be specified.

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
        limb_dark_law : string; `quadratic` or `non-linear`
            The stellar limb darkening law.
        u :  (N,) array_like
            Limb-darkening coefficients. 1D array of coefficients that
            correspond to the limb darkening law specified by the
            limb_dark_law.

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
        r :  (N,) or (N, M) array_like
            Transmission string coefficients. 1D array of N Fourier
            coefficients that specify the planet radius as a function
            of angle in the sky-plane. Coefficients correspond to
            ``r_{\rm{p}}(\theta) = \sum_{n=0}^N a_n \cos{(n \theta)}
            + \sum_{n=1}^N b_n \csin{(n \theta)}`` where the resulting
            input is r=[a_0, a_1, b_1, a_2, b_2,..]. For time-dependent
            transmission strings, use a 2D array with N Fourier
            coefficients and M time steps, where M is equal to the
            number of model evaluation epochs.

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

    def get_planet_transmission_string(self):
        """
        Get transmission string evaluated at given angles.

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

    def get_precision_estimate(self, N_l):
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
