import unittest
import numpy as np

from harmonica import bindings


class TestFlux(unittest.TestCase):
    """ Test flux computations. """

    def __init__(self, *args, **kwargs):
        super(TestFlux, self).__init__(*args, **kwargs)

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
        self.ecc_non_zero = 0.1
        self.omega = 0.1 * np.pi / 180.

        # Input data structures.
        self.times = None
        self.fs = None

    def _build_test_data_structures(self, n_dp=100):
        """ Build test input data structures. """
        self.times = np.ascontiguousarray(np.linspace(0., 7.5, n_dp),
                                          dtype=np.float64)
        self.fs = np.empty(self.times.shape, dtype=np.float64)

    def test_flux_data_structures(self):
        """ Test flux data structures. """
        rs = np.array([0.1, 0.001, 0.001], dtype=np.float64)
        quad_ld = np.array([0.1, 0.5], dtype=np.float64)
        nl_ld = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64)

        # Check quad and nl limb-darkening cases.
        for ld_law, us in zip([0, 1], [quad_ld, nl_ld]):

            # Check circular and eccentric cases.
            for ecc, w, in zip([self.ecc_zero, self.omega],
                               [self.ecc_non_zero, self.omega]):
                self._build_test_data_structures(n_dp=100)

                # Check input array types compatible.
                fs_shape = self.fs.shape
                self.assertIsInstance(us, np.ndarray)
                self.assertIsInstance(rs, np.ndarray)
                self.assertIsInstance(self.times, np.ndarray)
                self.assertIsInstance(self.fs, np.ndarray)

                bindings.light_curve(
                    self.t0, self.period, self.a, self.inc, ecc, w,
                    ld_law, us, rs, self.times, self.fs, 50, 500)

                # Check output array type unchanged.
                self.assertIsInstance(self.fs, np.ndarray)

                # Check updated array has consistent shape.
                self.assertEqual(self.fs.shape, fs_shape)


    # todo: add trasnmisison string test

    def test_flux_derivatives_switch(self):
        """ Test flux derivatives switch. """
        rs = np.array([0.1, 0.001, 0.001], dtype=np.float64)
        quad_ld = np.array([0.1, 0.5], dtype=np.float64)
        nl_ld = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64)

        # Check quad and nl limb-darkening cases.
        for ld_law, us in zip([0, 1], [quad_ld, nl_ld]):

            if ld_law == 0:
                self._build_test_data_structures(n_dp=100, n_us=2, n_rs=3)
            else:
                self._build_test_data_structures(n_dp=100, n_us=4, n_rs=3)

            # Derivatives switched off.
            fs_grad_a = np.copy(self.fs_grad)
            bindings.orbit(self.t0, self.period, self.a,
                           self.inc, self.ecc_zero, self.omega,
                           self.times, self.ds, self.nus,
                           self.ds_grad, self.nus_grad,
                           require_gradients=False)
            bindings.light_curve(ld_law, us, rs,
                                 self.ds, self.nus, self.fs,
                                 self.ds_grad, self.nus_grad, self.fs_grad,
                                 50, 500, require_gradients=False)
            self.assertTrue(np.array_equal(fs_grad_a, self.fs_grad))

            # Derivatives switched on.
            bindings.orbit(self.t0, self.period, self.a,
                           self.inc, self.ecc_zero, self.omega,
                           self.times, self.ds, self.nus,
                           self.ds_grad, self.nus_grad,
                           require_gradients=True)
            bindings.light_curve(ld_law, us, rs,
                                 self.ds, self.nus, self.fs,
                                 self.ds_grad, self.nus_grad, self.fs_grad,
                                 50, 500, require_gradients=True)
            self.assertFalse(np.array_equal(fs_grad_a, self.fs_grad))

    def test_flux_derivative_quad_ld_df_dy(self):
        """ Test flux derivative dd_dy, y={t0, p, a, i, e, w, {us}, {rs}}. """
        # Check derivatives wrt t0, period, a, and inc.
        y_idxs = [0, 1, 2, 3, 4, 5, 6, 8]
        y_names = ['t0', 'period', 'a', 'inc', 'e', 'w', 'us', 'rs']
        for param_idx, param_name in zip(y_idxs, y_names):

            # Randomly generate trial light curves.
            for i in range(20):

                circular_bool = np.random.binomial(1, 0.5)
                if not circular_bool or param_name == 'e':
                    ecc = np.random.uniform(0., 0.9)
                else:
                    ecc = 0.

                us = np.random.uniform(0.01, 0.9, 2)

                n_rs = 2 * (np.random.randint(3, 9) // 2) + 1
                a0 = np.random.uniform(0.05, 1.5)
                rs = np.append([a0], a0 * np.random.uniform(-0.001, 0.001, n_rs - 1))

                params = {'t0': np.random.uniform(-1., 11.),
                          'period': np.random.uniform(5., 100.),
                          'a': np.random.uniform(5., 10.),
                          'inc': np.random.uniform(80., 90.) * np.pi / 180.,
                          'e': ecc,
                          'w': np.random.uniform(0., 2. * np.pi),
                          'us': us,
                          'rs': rs}

                # Compute fluxes.
                self._build_test_data_structures(n_dp=100, n_us=2, n_rs=n_rs)
                bindings.orbit(params['t0'], params['period'], params['a'],
                               params['inc'], params['e'], params['w'],
                               self.times, self.ds, self.nus,
                               self.ds_grad, self.nus_grad,
                               require_gradients=True)
                bindings.light_curve(0, params['us'], params['rs'],
                                     self.ds, self.nus, self.fs,
                                     self.ds_grad, self.nus_grad,
                                     self.fs_grad, 50, 500,
                                     require_gradients=True)
                fs_a = np.copy(self.fs)

                # Update z by epsilon.
                if param_name == 'us':
                    u_idx = np.random.randint(0, 2)
                    us[u_idx] += self.epsilon
                    params['us'] = us
                    _param_idx = param_idx + u_idx
                elif param_name == 'rs':
                    r_idx = np.random.randint(0, n_rs)
                    rs[r_idx] += self.epsilon
                    params['rs'] = rs
                    _param_idx = param_idx + r_idx
                else:
                    params[param_name] = params[param_name] + self.epsilon
                    _param_idx = param_idx

                # Get gradient.
                df_dy_a = np.copy(self.fs_grad[:, _param_idx])

                # Compute fluxes at new z.
                self._build_test_data_structures(n_dp=100, n_us=2, n_rs=n_rs)
                bindings.orbit(params['t0'], params['period'], params['a'],
                               params['inc'], params['e'], params['w'],
                               self.times, self.ds, self.nus,
                               self.ds_grad, self.nus_grad,
                               require_gradients=True)
                bindings.light_curve(0, params['us'], params['rs'],
                                     self.ds, self.nus, self.fs,
                                     self.ds_grad, self.nus_grad,
                                     self.fs_grad, 50, 500,
                                     require_gradients=True)
                fs_b = np.copy(self.fs)

                # Check algebraic gradients match numerical.
                res_iter = zip(fs_a, fs_b, df_dy_a)
                for res_idx, (d_a, d_b, grad) in enumerate(res_iter):
                    delta_d = d_b - d_a
                    residual = np.abs(d_b - (grad * self.epsilon + d_a))
                    tol = max(np.abs(delta_d * 1.e-2), 1.e-13)
                    self.assertLess(
                        residual, tol, msg='df/d{} failed no.{}.'.format(
                            param_name, res_idx))

    def test_temp(self):
        us = np.array([0.1, 0.2, 0.3, 0.4])
        rs = np.array([0.1, -0.003, 0.003])
        times = np.array([4.9])
        fs = np.empty(times.shape, dtype=np.float64, order='C')
        fs_grad = np.empty(times.shape + (6 + 4 + 3,), dtype=np.float64, order='C')
        bindings.temp_light_curve(5., 10., 10., 89.9 * np.pi / 180., 0.01, 0.1,
                                  1, us, rs, times,
                                  fs, fs_grad, 50, 500)
        print(fs)

    def test_flux_derivative_non_linear_ld_df_dy(self):
        """ Test flux derivative dd_dy, y={t0, p, a, i, e, w, {us}, {rs}}. """
        # Check derivatives wrt t0, period, a, and inc.
        # y_idxs = [0, 1, 2, 3, 4, 5, 6, 10]
        # y_names = ['t0', 'period', 'a', 'inc', 'e', 'w', 'us', 'rs']
        y_idxs = [0]
        y_names = ['t0']
        for param_idx, param_name in zip(y_idxs, y_names):

            # Randomly generate trial light curves.
            for i in range(20):

                circular_bool = np.random.binomial(1, 0.5)
                if not circular_bool or param_name == 'e':
                    ecc = np.random.uniform(0., 0.9)
                else:
                    ecc = 0.

                us = np.random.uniform(0.01, 0.9, 4)
                # us = [0., 1., 0., 1.]

                n_rs = 2 * (np.random.randint(3, 9) // 2) + 1
                a0 = np.random.uniform(0.05, 1.5)
                rs = np.append([a0], a0 * np.random.uniform(-0.001, 0.001, n_rs - 1))

                params = {'t0': np.random.uniform(-1., 11.),
                          'period': np.random.uniform(5., 100.),
                          'a': np.random.uniform(5., 10.),
                          'inc': np.random.uniform(80., 90.) * np.pi / 180.,
                          'e': ecc,
                          'w': np.random.uniform(0., 2. * np.pi),
                          'us': us,
                          'rs': rs}

                # Compute fluxes.
                self._build_test_data_structures(n_dp=100, n_us=4, n_rs=n_rs)
                bindings.orbit(params['t0'], params['period'], params['a'],
                               params['inc'], params['e'], params['w'],
                               self.times, self.ds, self.nus,
                               self.ds_grad, self.nus_grad,
                               require_gradients=True)
                bindings.light_curve(1, params['us'], params['rs'],
                                     self.ds, self.nus, self.fs,
                                     self.ds_grad, self.nus_grad,
                                     self.fs_grad, 50, 500,
                                     require_gradients=True)
                fs_a = np.copy(self.fs)

                # Update z by epsilon.
                print(param_name)
                if param_name == 'us':
                    u_idx = np.random.randint(0, 4)
                    us[u_idx] += self.epsilon
                    params['us'] = us
                    _param_idx = param_idx + u_idx
                elif param_name == 'rs':
                    r_idx = np.random.randint(0, n_rs)
                    rs[r_idx] += self.epsilon
                    params['rs'] = rs
                    _param_idx = param_idx + r_idx
                else:
                    params[param_name] = params[param_name] + self.epsilon
                    _param_idx = param_idx

                # Get gradient.
                df_dy_a = np.copy(self.fs_grad[:, _param_idx])

                # Compute fluxes at new z.
                self._build_test_data_structures(n_dp=100, n_us=4, n_rs=n_rs)
                bindings.orbit(params['t0'], params['period'], params['a'],
                               params['inc'], params['e'], params['w'],
                               self.times, self.ds, self.nus,
                               self.ds_grad, self.nus_grad,
                               require_gradients=True)
                bindings.light_curve(1, params['us'], params['rs'],
                                     self.ds, self.nus, self.fs,
                                     self.ds_grad, self.nus_grad,
                                     self.fs_grad, 50, 500,
                                     require_gradients=True)
                fs_b = np.copy(self.fs)

                # Check algebraic gradients match numerical.
                res_iter = zip(fs_a, fs_b, df_dy_a)
                for res_idx, (d_a, d_b, grad) in enumerate(res_iter):
                    delta_d = d_b - d_a
                    residual = np.abs(d_b - (grad * self.epsilon + d_a))
                    tol = max(np.abs(delta_d * 1.e-2), 1.e-13)
                    self.assertLess(
                        residual, tol, msg='df/d{} failed no.{}.'.format(
                            param_name, res_idx))


if __name__ == '__main__':
    unittest.main()
