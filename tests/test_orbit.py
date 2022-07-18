import unittest
import numpy as np

from harmonica import bindings


class TestOrbit(unittest.TestCase):
    """ Test orbital computations. """

    def __init__(self, *args, **kwargs):
        super(TestOrbit, self).__init__(*args, **kwargs)

        # Make reproducible.
        np.random.seed(3)

        # Differential element, epsilon.
        self.epsilon = 1.e-8

        # Example params.
        self.t0 = 5.
        self.period = 10.
        self.a = 7.
        self.inc = 88. * np.pi / 180.
        self.ecc_zero = 0.
        self.ecc_non_zero = 0.22
        self.omega = 101. * np.pi / 180.

        # Input data structures.
        self.times = None
        self.fs = None

    def _build_test_data_structures(self, n_dp=100):
        """ Build test input data structures. """
        self.times = np.ascontiguousarray(np.linspace(2.5, 7.5, n_dp),
                                          dtype=np.float64)
        self.fs = np.empty(self.times.shape, dtype=np.float64)

    def test_orbit_data_structures(self):
        """ Test orbit data structures. """
        # todo: deprecated.
        self._build_test_data_structures(n_dp=100)

        # Check circular and eccentric cases.
        for ecc, w, in zip([self.ecc_zero, self.omega],
                           [self.ecc_non_zero, self.omega]):

            # Check input array types compatible.
            times_shape = self.times.shape
            fs_shape = self.fs.shape

            bindings.light_curve(self.t0, self.period, self.a, self.inc,
                                 ecc, w,
                           self.times, self.ds, self.zs, self.nus)

            # Check output array types unchanged.
            self.assertIsInstance(self.times, np.ndarray)
            self.assertIsInstance(self.ds, np.ndarray)
            self.assertIsInstance(self.zs, np.ndarray)
            self.assertIsInstance(self.nus, np.ndarray)

            # Check updated arrays have consistent shapes.
            self.assertEqual(self.times.shape, times_shape)
            self.assertEqual(self.ds.shape, ds_shape)
            self.assertEqual(self.zs.shape, zs_shape)
            self.assertEqual(self.nus.shape, nus_shape)














    # todo: put these tests elsewhere, in jax checks and derivatives.


    def test_orbit_derivatives_switch(self):
        """ Test orbit derivatives switch. """
        self._build_test_data_structures(n_dp=100)

        # Derivatives switched off.
        ds_grad_a = np.copy(self.ds_grad)
        nus_grad_a = np.copy(self.nus_grad)
        bindings.orbit(self.t0, self.period, self.a,
                       self.inc, self.ecc_zero, self.omega,
                       self.times, self.ds, self.nus,
                       self.ds_grad, self.nus_grad,
                       require_gradients=False)
        self.assertTrue(np.array_equal(
            ds_grad_a, self.ds_grad, equal_nan=True))
        self.assertTrue(np.array_equal(
            nus_grad_a, self.nus_grad, equal_nan=True))

        # Derivatives switched on, circular orbit.
        bindings.orbit(self.t0, self.period, self.a,
                       self.inc, self.ecc_zero, self.omega,
                       self.times, self.ds, self.nus,
                       self.ds_grad, self.nus_grad,
                       require_gradients=True)
        self.assertFalse(np.array_equal(
            ds_grad_a[:, 0:4], self.ds_grad[:, 0:4], equal_nan=True))
        self.assertFalse(np.array_equal(
            nus_grad_a[:, 0:-2], self.nus_grad[:, 0:4], equal_nan=True))
        self.assertTrue(np.array_equal(
            ds_grad_a[:, 4:6], self.ds_grad[:, 4:6], equal_nan=True))
        self.assertTrue(np.array_equal(
            nus_grad_a[:, 4:6], self.nus_grad[:, 4:6], equal_nan=True))

        # Derivatives switched on, eccentric orbit.
        bindings.orbit(self.t0, self.period, self.a,
                       self.inc, self.ecc_non_zero, self.omega,
                       self.times, self.ds, self.nus,
                       self.ds_grad, self.nus_grad,
                       require_gradients=True)
        self.assertFalse(np.array_equal(
            ds_grad_a[:, 0:4], self.ds_grad[:, 0:4], equal_nan=True))
        self.assertFalse(np.array_equal(
            nus_grad_a[:, 0:4], self.nus_grad[:, 0:4], equal_nan=True))
        self.assertFalse(np.array_equal(
            ds_grad_a[:, 4:6], self.ds_grad[:, 4:6], equal_nan=True))
        self.assertFalse(np.array_equal(
            nus_grad_a[:, 4:6], self.nus_grad[:, 4:6], equal_nan=True))

    def test_orbit_derivative_circular_dd_dz(self):
        """ Test orbit derivative circular dd_dz, z={t0, p, a, i}. """
        self._build_test_data_structures(n_dp=100)

        # Check derivatives wrt t0, period, a, and inc.
        z_idxs = [0, 1, 2, 3]
        z_names = ['t0', 'period', 'a', 'inc']
        for param_idx, param_name in zip(z_idxs, z_names):

            # Randomly generate trial geometries.
            for i in range(20):
                params = {'t0': np.random.uniform(-1., 11.),
                          'period': np.random.uniform(5., 100.),
                          'a': np.random.uniform(5., 10.),
                          'inc': np.random.uniform(80., 90.) * np.pi / 180.}

                # Compute orbital separations and derivatives.
                bindings.orbit(params['t0'], params['period'],
                               params['a'], params['inc'], 0., 0.,
                               self.times, self.ds, self.nus,
                               self.ds_grad, self.nus_grad,
                               require_gradients=True)
                ds_a = np.copy(self.ds)
                dd_dz_a = np.copy(self.ds_grad[:, param_idx])

                # Update z by epsilon.
                params[param_name] = params[param_name] + self.epsilon

                # Compute orbital separations at new z.
                bindings.orbit(params['t0'], params['period'],
                               params['a'], params['inc'], 0., 0.,
                               self.times, self.ds, self.nus,
                               self.ds_grad, self.nus_grad,
                               require_gradients=False)
                ds_b = np.copy(self.ds)

                # Check algebraic gradients match numerical.
                res_iter = zip(ds_a, ds_b, dd_dz_a)
                for res_idx, (d_a, d_b, grad) in enumerate(res_iter):
                    if d_a == 5000:
                        # Planet behind star, derivatives not computed.
                        continue
                    delta_d = d_b - d_a
                    residual = np.abs(d_b - (grad * self.epsilon + d_a))
                    tol = max(np.abs(delta_d * 1.e-2), 1.e-13)
                    self.assertLess(
                        residual, tol, msg='dd/d{} failed no.{}.'.format(
                            param_name, res_idx))

    def test_orbit_derivative_circular_dnu_dz(self):
        """ Test orbit derivative circular dnu_dz, z={t0, p, a, i}. """
        self._build_test_data_structures(n_dp=100)

        # Check derivatives wrt t0, period, a, and inc.
        z_idxs = [0, 1, 2, 3]
        z_names = ['t0', 'period', 'a', 'inc']
        for param_idx, param_name in zip(z_idxs, z_names):

            # Randomly generate trial geometries.
            for i in range(20):
                params = {'t0': np.random.uniform(-1., 11.),
                          'period': np.random.uniform(5., 100.),
                          'a': np.random.uniform(5., 10.),
                          'inc': np.random.uniform(80., 90.) * np.pi / 180.}

                # Compute orbital angles and derivatives.
                bindings.orbit(params['t0'], params['period'],
                               params['a'], params['inc'], 0., 0.,
                               self.times, self.ds, self.nus,
                               self.ds_grad, self.nus_grad,
                               require_gradients=True)
                ds_a = np.copy(self.ds)
                nus_a = np.copy(self.nus)
                dnu_dz_a = np.copy(self.nus_grad[:, param_idx])

                # Update z by epsilon.
                params[param_name] = params[param_name] + self.epsilon

                # Compute orbital separations at new z.
                bindings.orbit(params['t0'], params['period'],
                               params['a'], params['inc'], 0., 0.,
                               self.times, self.ds, self.nus,
                               self.ds_grad, self.nus_grad,
                               require_gradients=False)
                nus_b = np.copy(self.nus)

                # Check algebraic gradients match numerical.
                res_iter = zip(ds_a, nus_a, nus_b, dnu_dz_a)
                for res_idx, (d_a, nu_a, nu_b, grad) in enumerate(res_iter):
                    if d_a == 5000:
                        # Planet behind star, derivatives not computed.
                        continue
                    delta_nu = nu_b - nu_a
                    residual = np.abs(nu_b - (grad * self.epsilon + nu_a))
                    tol = max(np.abs(delta_nu * 1.e-2), 1.e-13)
                    self.assertLess(
                        residual, tol, msg='dnu/d{} failed no.{}.'.format(
                            param_name, res_idx))

    def test_orbit_derivative_eccentric_dd_dz(self):
        """ Test orbit derivative eccentric dd_dz, z={t0, p, a, i, e, w}. """
        self._build_test_data_structures(n_dp=100)

        # Check derivatives wrt t0, period, a, inc, e, and w.
        z_idxs = [0, 1, 2, 3, 4, 5]
        z_names = ['t0', 'period', 'a', 'inc', 'e', 'w']
        for param_idx, param_name in zip(z_idxs, z_names):

            # Randomly generate trial geometries.
            for i in range(20):
                params = {'t0': np.random.uniform(-1., 11.),
                          'period': np.random.uniform(5., 100.),
                          'a': np.random.uniform(5., 10.),
                          'inc': np.random.uniform(80., 90.) * np.pi / 180.,
                          'e': np.random.uniform(0., 0.9),
                          'w': np.random.uniform(0., 2. * np.pi)}

                # Compute orbital separations and derivatives.
                bindings.orbit(params['t0'], params['period'], params['a'],
                               params['inc'], params['e'], params['w'],
                               self.times, self.ds, self.nus,
                               self.ds_grad, self.nus_grad,
                               require_gradients=True)
                ds_a = np.copy(self.ds)
                dd_dz_a = np.copy(self.ds_grad[:, param_idx])

                # Update z by epsilon.
                params[param_name] = params[param_name] + self.epsilon

                # Compute orbital separations at new z.
                bindings.orbit(params['t0'], params['period'], params['a'],
                               params['inc'], params['e'], params['w'],
                               self.times, self.ds, self.nus,
                               self.ds_grad, self.nus_grad,
                               require_gradients=False)
                ds_b = np.copy(self.ds)

                # Check algebraic gradients match numerical.
                res_iter = zip(ds_a, ds_b, dd_dz_a)
                for res_idx, (d_a, d_b, grad) in enumerate(res_iter):
                    if d_a == 5000:
                        # Planet behind star, derivatives not computed.
                        continue
                    delta_d = d_b - d_a
                    residual = np.abs(d_b - (grad * self.epsilon + d_a))
                    tol = max(np.abs(delta_d * 1.e-2), 1.e-13)
                    self.assertLess(
                        residual, tol, msg='dd/d{} failed no.{}.'.format(
                            param_name, res_idx))

    def test_orbit_derivative_eccentric_dnu_dz(self):
        """ Test orbit derivative eccentric dnu_dz, z={t0, p, a, i, e, w}. """
        self._build_test_data_structures(n_dp=100)

        # Check derivatives wrt t0, period, a, inc, e, and w.
        z_idxs = [0, 1, 2, 3, 4, 5]
        z_names = ['t0', 'period', 'a', 'inc', 'e', 'w']
        for param_idx, param_name in zip(z_idxs, z_names):

            # Randomly generate trial geometries.
            for i in range(20):
                params = {'t0': np.random.uniform(-1., 11.),
                          'period': np.random.uniform(5., 100.),
                          'a': np.random.uniform(5., 10.),
                          'inc': np.random.uniform(80., 90.) * np.pi / 180.,
                          'e': np.random.uniform(0., 0.9),
                          'w': np.random.uniform(0., 2. * np.pi)}

                # Compute orbital separations and derivatives.
                bindings.orbit(params['t0'], params['period'], params['a'],
                               params['inc'], params['e'], params['w'],
                               self.times, self.ds, self.nus,
                               self.ds_grad, self.nus_grad,
                               require_gradients=True)
                ds_a = np.copy(self.ds)
                nus_a = np.copy(self.nus)
                dnu_dz_a = np.copy(self.nus_grad[:, param_idx])

                # Update z by epsilon.
                params[param_name] = params[param_name] + self.epsilon

                # Compute orbital separations at new z.
                bindings.orbit(params['t0'], params['period'], params['a'],
                               params['inc'], params['e'], params['w'],
                               self.times, self.ds, self.nus,
                               self.ds_grad, self.nus_grad,
                               require_gradients=False)
                nus_b = np.copy(self.nus)

                # Check algebraic gradients match numerical.
                res_iter = zip(ds_a, nus_a, nus_b, dnu_dz_a)
                for res_idx, (d_a, nu_a, nu_b, grad) in enumerate(res_iter):
                    if d_a == 5000:
                        # Planet behind star, derivatives not computed.
                        continue
                    delta_nu = nu_b - nu_a
                    residual = np.abs(nu_b - (grad * self.epsilon + nu_a))
                    tol = max(np.abs(delta_nu * 1.e-2), 1.e-13)
                    self.assertLess(
                        residual, tol, msg='dnu/d{} failed no.{}.'.format(
                            param_name, res_idx))


if __name__ == '__main__':
    unittest.main()
