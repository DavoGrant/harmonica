import unittest
import numpy as np

from harmonica import bindings
from harmonica import HarmonicaTransit


class TestFlux(unittest.TestCase):
    """ Test flux computations. """

    def __init__(self, *args, **kwargs):
        super(TestFlux, self).__init__(*args, **kwargs)

        # Make reproducible.
        np.random.seed(3)

        # Example params.
        self.t0 = 5.
        self.period = 10.
        self.a = 10.
        self.inc = 89. * np.pi / 180.
        self.ecc_zero = 0.
        self.ecc_non_zero = 0.1
        self.omega = 0.1 * np.pi / 180.

        # Input data structures.
        self.times = None
        self.fs = None

    def _build_test_data_structures(self, n_dp=100, start=2.5, stop=7.5):
        """ Build test input data structures. """
        self.times = np.ascontiguousarray(
            np.linspace(start, stop, n_dp), dtype=np.float64)
        self.fs = np.empty(self.times.shape, dtype=np.float64)

    def test_bindings_light_curve(self):
        """ Test primary binding, light_curve. """
        rs = np.array([0.1, 0.001, 0.001], dtype=np.float64)
        quad_ld = np.array([0.1, 0.5], dtype=np.float64)
        nl_ld = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64)

        # Check quad and nl limb-darkening cases.
        for ld_law, us in zip([0, 1], [quad_ld, nl_ld]):

            # Check circular and eccentric cases.
            for ecc, w, in zip([self.ecc_zero, self.omega],
                               [self.ecc_non_zero, self.omega]):
                self._build_test_data_structures(n_dp=1000)

                # Check input array types are as expected.
                self.assertIsInstance(us, np.ndarray)
                self.assertIsInstance(rs, np.ndarray)
                self.assertIsInstance(self.times, np.ndarray)
                self.assertIsInstance(self.fs, np.ndarray)

                bindings.light_curve(
                    self.t0, self.period, self.a, self.inc, ecc, w,
                    ld_law, us, rs, self.times, self.fs, 20, 50)

                # Check output array type, shape, and vals.
                self.assertIsInstance(self.fs, np.ndarray)
                self.assertEqual(self.fs.shape, self.times.shape)
                self.assertEqual(np.sum(np.isfinite(self.fs)),
                                 self.fs.shape[0])

    def test_api_light_curve(self):
        """ Test api, transit light curve. """
        rs = np.array([0.1, 0.001, 0.001], dtype=np.float64)
        quad_ld = np.array([0.1, 0.5], dtype=np.float64)
        nl_ld = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64)

        # Check quad and nl limb-darkening cases.
        for ld_law, us in zip([0, 1], [quad_ld, nl_ld]):

            if ld_law == 0:
                limb_dark_law = 'quadratic'
            else:
                limb_dark_law = 'non-linear'

            # Check circular and eccentric cases.
            for ecc, w, in zip([self.ecc_zero, self.omega],
                               [self.ecc_non_zero, self.omega]):
                self._build_test_data_structures(n_dp=1000)

                ht = HarmonicaTransit(self.times)
                ht.set_orbit(self.t0, self.period, self.a, self.inc, ecc, w)
                ht.set_stellar_limb_darkening(us, limb_dark_law)
                ht.set_planet_transmission_string(rs)
                fluxes = ht.get_transit_light_curve()

                # Check output array type, shape, and vals.
                self.assertIsInstance(fluxes, np.ndarray)
                self.assertEqual(fluxes.shape, self.times.shape)
                self.assertEqual(np.sum(np.isfinite(fluxes)),
                                 fluxes.shape[0])

    def test_api_light_curve_time_dependent(self):
        """ Test api, light curve w/ time dependence. """
        rs_ingress = np.array([0.1, 0.001, 0.001], dtype=np.float64)
        rs_egress = np.array([0.1, -0.001, 0.001], dtype=np.float64)
        quad_ld = np.array([0.1, 0.5], dtype=np.float64)
        nl_ld = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64)

        # Check quad and nl limb-darkening cases.
        for ld_law, us in zip([0, 1], [quad_ld, nl_ld]):

            if ld_law == 0:
                limb_dark_law = 'quadratic'
            else:
                limb_dark_law = 'non-linear'

            # Check circular and eccentric cases.
            for ecc, w, in zip([self.ecc_zero, self.omega],
                               [self.ecc_non_zero, self.omega]):
                self._build_test_data_structures(n_dp=1000,
                                                 start=4.4, stop=5.6)

                # Build linear interpolating variable string.
                rs = np.empty(self.times.shape + (3,), dtype=np.float64)
                rs[:, 0] = np.interp(
                    self.times, [4.6, 5.4], [rs_ingress[0], rs_egress[0]])
                rs[:, 1] = np.interp(
                    self.times, [4.6, 5.4], [rs_ingress[1], rs_egress[1]])
                rs[:, 2] = np.interp(
                    self.times, [4.6, 5.4], [rs_ingress[2], rs_egress[2]])
                self.assertEqual(rs.ndim, 2)
                self.assertEqual(rs.shape[0], self.times.shape[0])

                ht = HarmonicaTransit(self.times)
                ht.set_orbit(self.t0, self.period, self.a, self.inc, ecc, w)
                ht.set_stellar_limb_darkening(us, limb_dark_law)
                ht.set_planet_transmission_string(rs)
                fluxes = ht.get_transit_light_curve()

                # Check output array type, shape, and vals.
                self.assertIsInstance(fluxes, np.ndarray)
                self.assertEqual(fluxes.shape, self.times.shape)
                self.assertEqual(np.sum(np.isfinite(fluxes)),
                                 fluxes.shape[0])

    def test_api_transmission_string(self):
        """ Test api, transmission string. """
        # Constant strings of various complexity.
        theta = np.linspace(-np.pi, np.pi, 10000)
        rs = np.array([0.1, 0.001, 0.002, -0.001, -0.002, 0.003, -0.003],
                      dtype=np.float64)
        for n_rs in range(1, 8, 2):
            ht = HarmonicaTransit()
            ht.set_planet_transmission_string(rs[:n_rs])
            r_p = ht.get_planet_transmission_string(theta)

            # Check output array type, shape, and vals.
            self.assertIsInstance(r_p, np.ndarray)
            self.assertEqual(r_p.shape, theta.shape)
            self.assertEqual(np.sum(np.isfinite(r_p)),
                             r_p.shape[0])

    def test_api_transmission_string_time_dependent(self):
        """ Test api, transmission string w/ time dependence. """
        # Time dependent string.
        theta = np.linspace(-np.pi, np.pi, 100)
        self._build_test_data_structures(n_dp=1000, start=4.4, stop=5.6)
        rs_ingress = np.array([0.1, 0.001, 0.001], dtype=np.float64)
        rs_egress = np.array([0.1, -0.001, 0.001], dtype=np.float64)
        rs = np.empty(self.times.shape + (3,), dtype=np.float64)
        rs[:, 0] = np.interp(
            self.times, [4.6, 5.4], [rs_ingress[0], rs_egress[0]])
        rs[:, 1] = np.interp(
            self.times, [4.6, 5.4], [rs_ingress[1], rs_egress[1]])
        rs[:, 2] = np.interp(
            self.times, [4.6, 5.4], [rs_ingress[2], rs_egress[2]])

        ht = HarmonicaTransit()
        ht.set_planet_transmission_string(rs)
        r_ps = ht.get_planet_transmission_string(theta)

        # Check output array type, shape, and vals.
        self.assertIsInstance(r_ps, np.ndarray)
        self.assertEqual(r_ps.shape[0], self.times.shape[0])
        self.assertEqual(r_ps.shape[1], theta.shape[0])
        self.assertEqual(np.sum(np.isfinite(r_ps)),
                         r_ps.shape[0] * r_ps.shape[1])

    def test_api_precision_check(self):
        """ Test api, precision check. """
        quad_ld = np.array([0.1, 0.5], dtype=np.float64)
        rs = np.array([0.1, 0.001, 0.001], dtype=np.float64)
        self._build_test_data_structures(n_dp=1000, start=4.4, stop=5.6)

        ht = HarmonicaTransit(self.times, pnl_c=20, pnl_e=50)
        ht.set_orbit(self.t0, self.period, self.a, self.inc,
                     self.ecc_non_zero, self.omega)
        ht.set_stellar_limb_darkening(quad_ld)
        ht.set_planet_transmission_string(rs)
        errors = ht.get_precision_estimate()

        # Check output array type, shape, and vals.
        self.assertIsInstance(errors, np.ndarray)
        self.assertEqual(errors.shape, self.times.shape)
        self.assertEqual(np.sum(np.isfinite(errors)),
                         errors.shape[0])
        self.assertLess(np.max(np.abs(errors)), 1.e-7)


if __name__ == '__main__':
    unittest.main()
