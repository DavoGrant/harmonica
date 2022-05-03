import unittest
import numpy as np

from harmonica import bindings


class TestOrbit(unittest.TestCase):
    """ Test orbital computations. """

    def __init__(self, *args, **kwargs):
        super(TestOrbit, self).__init__(*args, **kwargs)

        # Make reproducible.
        np.random.seed(32)

        # Differential element, epsilon.
        self.epsilon = 1.e-10

        # Build input data structures.
        self.times = np.ascontiguousarray(np.linspace(0., 20., 200),
                                          dtype=np.float64)
        self.ds = np.empty(self.times.shape, dtype=np.float64, order='C')
        self.nus = np.empty(self.times.shape, dtype=np.float64, order='C')
        n_od = self.times.shape + (6,)
        self.ds_grad = np.empty(n_od, dtype=np.float64, order='C')
        self.nus_grad = np.empty(n_od, dtype=np.float64, order='C')

    def test_orbit_derivative_circular_dd_dz(self):
        """ Test orbit derivative circular dd_dz, z={t0, p, a, i}. """
        # Check derivatives wrt t0, period, a, and inc.
        z_idxs = [0, 1, 2, 3]
        z_names = ['t0', 'period', 'a', 'inc']
        for param_idx, param_name in zip(z_idxs, z_names):

            # Randomly generate many trial geometries.
            for i in range(100):
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
                for d_a, d_b, grad in zip(ds_a, ds_b, dd_dz_a):
                    delta_d = d_b - d_a
                    residual = np.abs(d_b - (grad * self.epsilon + d_a))
                    tol = max(np.abs(delta_d * 1.e-2), 1.e-13)
                    self.assertLess(residual, tol)

    def test_orbit_derivative_circular_dnu_dz(self):
        """ Test orbit derivative circular dnu_dz, z={t0, p, a, i}. """
        # Check derivatives wrt t0, period, a, and inc.
        z_idxs = [0, 1, 2, 3]
        z_names = ['t0', 'period', 'a', 'inc']
        for param_idx, param_name in zip(z_idxs, z_names):

            # Randomly generate many trial geometries.
            for i in range(100):
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
                for d_a, d_b, grad in zip(nus_a, nus_b, dnu_dz_a):
                    delta_d = d_b - d_a
                    residual = np.abs(d_b - (grad * self.epsilon + d_a))
                    tol = max(np.abs(delta_d * 1.e-2), 1.e-13)
                    self.assertLess(residual, tol)


if __name__ == '__main__':
    unittest.main()
