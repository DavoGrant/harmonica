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

        # Build test input data structures.
        self.times = np.ascontiguousarray(np.linspace(0., 20., 200),
                                          dtype=np.float64)
        self.ds = np.empty(self.times.shape, dtype=np.float64, order='C')
        self.nus = np.empty(self.times.shape, dtype=np.float64, order='C')
        n_od = self.times.shape + (6,)
        self.ds_grad = np.empty(n_od, dtype=np.float64, order='C')
        self.nus_grad = np.empty(n_od, dtype=np.float64, order='C')

    def test_orbit_data_structures(self):
        """ Test orbit data structures. """
        # Check circular and eccentric cases.
        for ecc, w, in zip([0., 0.3], [0., 1.1]):

            # Check input array types compatible.
            times_shape = self.times.shape
            ds_shape = self.ds.shape
            nus_shape = self.nus.shape
            ds_grad_shape = self.ds_grad.shape
            nus_grad_shape = self.nus_grad.shape
            self.assertIsInstance(self.times, np.ndarray)
            self.assertIsInstance(self.ds, np.ndarray)
            self.assertIsInstance(self.nus, np.ndarray)
            self.assertIsInstance(self.ds_grad, np.ndarray)
            self.assertIsInstance(self.nus_grad, np.ndarray)
            bindings.orbit(5., 10., 7., 1.5, ecc, w,
                           self.times, self.ds, self.nus,
                           self.ds_grad, self.nus_grad,
                           require_gradients=True)

            # Check output array types unchanged.
            self.assertIsInstance(self.times, np.ndarray)
            self.assertIsInstance(self.ds, np.ndarray)
            self.assertIsInstance(self.nus, np.ndarray)
            self.assertIsInstance(self.ds_grad, np.ndarray)
            self.assertIsInstance(self.nus_grad, np.ndarray)

            # Check updated arrays have consistent shapes.
            self.assertEqual(self.times.shape, times_shape)
            self.assertEqual(self.ds.shape, ds_shape)
            self.assertEqual(self.nus.shape, nus_shape)
            self.assertEqual(self.ds_grad.shape, ds_grad_shape)
            self.assertEqual(self.nus_grad.shape, nus_grad_shape)

    def test_orbit_trajectory_sensical(self):
        """ Test orbit trajectory is sensical. """
        times = np.ascontiguousarray(np.arange(0., 10.5, 1), dtype=np.float64)
        ds = np.empty(times.shape, dtype=np.float64, order='C')
        nus = np.empty(times.shape, dtype=np.float64, order='C')
        n_od = times.shape + (6,)
        ds_grad = np.ones(n_od, dtype=np.float64, order='C')
        nus_grad = np.ones(n_od, dtype=np.float64, order='C')

        # Edge on, i=90, circular orbit.
        bindings.orbit(5., 10., 7., np.pi/2, 0., 0.,
                       times, ds, nus, ds_grad, nus_grad,
                       require_gradients=False)
        # Planet inline with stellar centre.
        self.assertAlmostEqual(ds[0], 0., delta=1.e-13)
        self.assertAlmostEqual(ds[5], 0., delta=1.e-13)
        self.assertAlmostEqual(ds[-1], 0., delta=1.e-13)
        # Planet orbiting closer/further from stellar centre.
        self.assertGreater(ds[3], ds[4])
        self.assertGreater(ds[7], ds[6])
        # Planet orbiting towards/away from stellar centre.
        self.assertAlmostEqual(nus[4], 0., delta=1.e-13)
        self.assertAlmostEqual(nus[6], np.pi, delta=1.e-13)

        # Just off edge on, i<90, circular orbit.
        bindings.orbit(5., 10., 7., np.pi/2.01, 0., 0.,
                       times, ds, nus, ds_grad, nus_grad,
                       require_gradients=False)
        # Planet not inline with stellar centre.
        self.assertGreater(ds[0], 0.)
        self.assertGreater(ds[5], 0.)
        self.assertGreater(ds[-1], 0.)
        # Planet orbiting closer/further from stellar centre.
        self.assertGreater(ds[3], ds[4])
        self.assertGreater(ds[7], ds[6])
        # Planet orbiting towards/away from just below stellar centre.
        self.assertGreater(nus[4], 0.)
        self.assertLess(nus[6], np.pi)

        # Eccentric orbit, periastron at transit.
        bindings.orbit(5., 10., 7., np.pi/2, 0.3, np.pi/2,
                       times, ds, nus, ds_grad, nus_grad,
                       require_gradients=False)
        # Planet orbit symmetric about transit.
        self.assertAlmostEqual(ds[4], ds[6], delta=1.e-13)

        # Eccentric orbit, periastron between transit and eclipse.
        bindings.orbit(5., 10., 7., np.pi/2, 0.3, 0.,
                       times, ds, nus, ds_grad, nus_grad,
                       require_gradients=False)
        # Planet orbit asymmetric about transit.
        self.assertLess(ds[4], ds[6])

    def test_orbit_circular_derivatives_switch(self):
        """ Test orbit circular derivatives switch. """
        times = np.ascontiguousarray(np.arange(0., 10.5, 1), dtype=np.float64)
        ds = np.empty(times.shape, dtype=np.float64, order='C')
        nus = np.empty(times.shape, dtype=np.float64, order='C')
        n_od = times.shape + (6,)
        ds_grad = np.ones(n_od, dtype=np.float64, order='C')
        nus_grad = np.ones(n_od, dtype=np.float64, order='C')

        ds_grad_a = np.copy(ds_grad)
        nus_grad_a = np.copy(nus_grad)
        bindings.orbit(5., 10., 7., np.pi / 2.1, 0., 0.,
                       times, ds, nus, ds_grad, nus_grad,
                       require_gradients=False)
        self.assertTrue(np.array_equal(ds_grad_a, ds_grad))
        self.assertTrue(np.array_equal(nus_grad_a, nus_grad))

        bindings.orbit(5., 10., 7., np.pi / 2.1, 0.1, 0.1,
                       times, ds, nus, ds_grad, nus_grad,
                       require_gradients=True)
        self.assertFalse(np.array_equal(ds_grad_a, ds_grad))
        self.assertFalse(np.array_equal(nus_grad_a, nus_grad))

    def test_orbit_derivative_circular_dd_dz(self):
        """ Test orbit derivative circular dd_dz, z={t0, p, a, i}. """
        # Check derivatives wrt t0, period, a, and inc.
        z_idxs = [0, 1, 2, 3]
        z_names = ['t0', 'period', 'a', 'inc']
        for param_idx, param_name in zip(z_idxs, z_names):

            # Randomly generate many trial geometries.
            for i in range(20):
                params = {'t0': np.random.uniform(0., 10.),
                          'period': np.random.uniform(0., 10.),
                          'a': np.random.uniform(0., 10.),
                          'inc': np.random.uniform(0., np.pi / 2.)}

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
                    delta_d = d_b - d_a
                    residual = np.abs(d_b - (grad * self.epsilon + d_a))
                    tol = max(np.abs(delta_d * 1.e-2), 1.e-13)
                    self.assertLess(
                        residual, tol, msg='dd/d{} failed no.{}.'.format(
                            param_name, res_idx))

    def test_orbit_derivative_circular_dnu_dz(self):
        """ Test orbit derivative circular dnu_dz, z={t0, p, a, i}. """
        # Check derivatives wrt t0, period, a, and inc.
        z_idxs = [0, 1, 2, 3]
        z_names = ['t0', 'period', 'a', 'inc']
        for param_idx, param_name in zip(z_idxs, z_names):

            # Randomly generate many trial geometries.
            for i in range(20):
                params = {'t0': np.random.uniform(0., 10.),
                          'period': np.random.uniform(0., 10.),
                          'a': np.random.uniform(0., 10.),
                          'inc': np.random.uniform(0., np.pi / 2.)}

                # Compute orbital angles and derivatives.
                bindings.orbit(params['t0'], params['period'],
                               params['a'], params['inc'], 0., 0.,
                               self.times, self.ds, self.nus,
                               self.ds_grad, self.nus_grad,
                               require_gradients=True)
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
                res_iter = zip(nus_a, nus_b, dnu_dz_a)
                for res_idx, (nu_a, nu_b, grad) in enumerate(res_iter):
                    delta_nu = nu_b - nu_a
                    residual = np.abs(nu_b - (grad * self.epsilon + nu_a))
                    tol = max(np.abs(delta_nu * 1.e-2), 1.e-13)
                    self.assertLess(
                        residual, tol, msg='dnu/d{} failed no.{}.'.format(
                            param_name, res_idx))

    def test_orbit_derivative_eccentric_dd_dz(self):
        """ Test orbit derivative eccentric dd_dz, z={t0, p, a, i, e, w}. """
        # Check derivatives wrt t0, period, a, inc, e, and w.
        z_idxs = [0, 1, 2, 3, 4, 5]
        z_names = ['t0', 'period', 'a', 'inc', 'e', 'w']
        for param_idx, param_name in zip(z_idxs, z_names):

            # Randomly generate many trial geometries.
            for i in range(20):
                params = {'t0': np.random.uniform(0., 10.),
                          'period': np.random.uniform(0., 10.),
                          'a': np.random.uniform(0., 10.),
                          'inc': np.random.uniform(0., np.pi / 2.),
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
                    delta_d = d_b - d_a
                    residual = np.abs(d_b - (grad * self.epsilon + d_a))
                    tol = max(np.abs(delta_d * 1.e-2), 1.e-13)
                    self.assertLess(
                        residual, tol, msg='dd/d{} failed no.{}.'.format(
                            param_name, res_idx))

    def test_orbit_derivative_eccentric_dnu_dz(self):
        """ Test orbit derivative eccentric dnu_dz, z={t0, p, a, i, e, w}. """
        # Check derivatives wrt t0, period, a, inc, e, and w.
        z_idxs = [0, 1, 2, 3, 4, 5]
        z_names = ['t0', 'period', 'a', 'inc', 'e', 'w']
        for param_idx, param_name in zip(z_idxs, z_names):

            # Randomly generate many trial geometries.
            for i in range(20):
                params = {'t0': np.random.uniform(0., 10.),
                          'period': np.random.uniform(0., 10.),
                          'a': np.random.uniform(0., 10.),
                          'inc': np.random.uniform(0., np.pi / 2.),
                          'e': np.random.uniform(0., 0.9),
                          'w': np.random.uniform(0., 2. * np.pi)}

                # Compute orbital separations and derivatives.
                bindings.orbit(params['t0'], params['period'], params['a'],
                               params['inc'], params['e'], params['w'],
                               self.times, self.ds, self.nus,
                               self.ds_grad, self.nus_grad,
                               require_gradients=True)
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
                res_iter = zip(nus_a, nus_b, dnu_dz_a)
                for res_idx, (nu_a, nu_b, grad) in enumerate(res_iter):
                    delta_nu = nu_b - nu_a
                    residual = np.abs(nu_b - (grad * self.epsilon + nu_a))
                    tol = max(np.abs(delta_nu * 1.e-2), 1.e-13)
                    self.assertLess(
                        residual, tol, msg='dnu/d{} failed no.{}.'.format(
                            param_name, res_idx))


if __name__ == '__main__':
    unittest.main()
