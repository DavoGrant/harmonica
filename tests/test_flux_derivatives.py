import unittest
import numpy as np
from jax import grad

from harmonica.jax import harmonica_transit


class TestFlux(unittest.TestCase):
    """ Test flux computations. """

    def __init__(self, *args, **kwargs):
        super(TestFlux, self).__init__(*args, **kwargs)

        # Make reproducible.
        np.random.seed(3)

        # Differential element and gradient error tolerance.
        self.epsilon = 1.e-8
        self.grad_tol = 1.e-5

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

    def test_flux_derivative_quad_ld(self):
        """ Test flux derivative for quadratic limb-darkening. """
        z_idxs = [1, 2, 3, 4, 5, 6, 8, 10]
        z_names = ['t0', 'period', 'a', 'inc', 'e', 'w', 'us', 'rs']
        for param_idx, param_name in zip(z_idxs, z_names):

            # Randomly generate trial light curves.
            for i in range(10):

                # Binomial probability of circular orbit.
                circular_bool = np.random.binomial(1, 0.5)
                if not circular_bool or param_name == 'e':
                    ecc = np.random.uniform(0., 0.9)
                else:
                    ecc = 0.

                # Uniform distributions of limb-darkening coeffs.
                us = np.random.uniform(0.01, 0.9, 2)

                # Uniform distributions of transmission string coeffs.
                n_rs = 2 * (np.random.randint(3, 9) // 2) + 1
                a0 = np.random.uniform(0.05, 1.5)
                rs = np.append([a0], a0 * np.random.uniform(
                    -0.001, 0.001, n_rs - 1))

                # Build parameter set.
                params = {'t0': np.random.uniform(2., 8.),
                          'period': np.random.uniform(5., 100.),
                          'a': np.random.uniform(5., 10.),
                          'inc': np.random.uniform(80., 90.) * np.pi / 180.,
                          'e': ecc,
                          'w': np.random.uniform(0., 2. * np.pi),
                          'us': us,
                          'rs': rs}

                # Compute fluxes.
                self._build_test_data_structures(n_dp=100, start=2.5, stop=7.5)
                args = [
                    self.times,
                    params['t0'],
                    params['period'],
                    params['a'],
                    params['inc'],
                    params['e'],
                    params['w'],
                    'quadratic']
                for u in params['us']:
                    args.append(u)
                for r in params['rs']:
                    args.append(r)
                fs_a = harmonica_transit(*args)

                # Update one parameter by epsilon.
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
                algebraic_gradients = []
                for j in args[0]:
                    point_args = args
                    point_args[0] = j
                    algebraic_gradients.append(grad(
                        harmonica_transit, argnums=_param_idx)(*point_args))

                # Compute fluxes with updated parameter set.
                self._build_test_data_structures(n_dp=100, start=2.5, stop=7.5)
                args = [
                    self.times,
                    params['t0'],
                    params['period'],
                    params['a'],
                    params['inc'],
                    params['e'],
                    params['w'],
                    'quadratic']
                for u in params['us']:
                    args.append(u)
                for r in params['rs']:
                    args.append(r)
                fs_b = harmonica_transit(*args)

                # Check algebraic gradients match finite difference.
                res_iter = zip(fs_a, fs_b, algebraic_gradients)
                for res_idx, (f_a, f_b, algebraic_grad) in enumerate(res_iter):
                    finite_diff_grad = (f_b - f_a) / self.epsilon
                    grad_err = np.abs(finite_diff_grad - algebraic_grad)
                    self.assertLess(
                        grad_err, self.grad_tol,
                        msg='df/d{} failed no.{}.'.format(param_name, res_idx))

    def test_flux_derivative_nonlinear_ld(self):
        """ Test flux derivative for non-linear limb-darkening. """
        z_idxs = [1, 2, 3, 4, 5, 6, 8, 12]
        z_names = ['t0', 'period', 'a', 'inc', 'e', 'w', 'us', 'rs']
        for param_idx, param_name in zip(z_idxs, z_names):

            # Randomly generate trial light curves.
            for i in range(10):

                # Binomial probability of circular orbit.
                circular_bool = np.random.binomial(1, 0.5)
                if not circular_bool or param_name == 'e':
                    ecc = np.random.uniform(0., 0.9)
                else:
                    ecc = 0.

                # Uniform distributions of limb-darkening coeffs.
                us = np.random.uniform(-0.9, 0.9, 4)

                # Uniform distributions of transmission string coeffs.
                n_rs = 2 * (np.random.randint(3, 9) // 2) + 1
                a0 = np.random.uniform(0.05, 1.5)
                rs = np.append([a0], a0 * np.random.uniform(
                    -0.001, 0.001, n_rs - 1))

                # Build parameter set.
                params = {'t0': np.random.uniform(2., 8.),
                          'period': np.random.uniform(5., 100.),
                          'a': np.random.uniform(5., 10.),
                          'inc': np.random.uniform(80., 90.) * np.pi / 180.,
                          'e': ecc,
                          'w': np.random.uniform(0., 2. * np.pi),
                          'us': us,
                          'rs': rs}

                # Compute fluxes.
                self._build_test_data_structures(n_dp=100, start=2.5, stop=7.5)
                args = [
                    self.times,
                    params['t0'],
                    params['period'],
                    params['a'],
                    params['inc'],
                    params['e'],
                    params['w'],
                    'non-linear']
                for u in params['us']:
                    args.append(u)
                for r in params['rs']:
                    args.append(r)
                fs_a = harmonica_transit(*args)

                # Update one parameter by epsilon.
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
                algebraic_gradients = []
                for j in args[0]:
                    point_args = args
                    point_args[0] = j
                    algebraic_gradients.append(grad(
                        harmonica_transit, argnums=_param_idx)(*point_args))

                # Compute fluxes with updated parameter set.
                self._build_test_data_structures(n_dp=100, start=2.5, stop=7.5)
                args = [
                    self.times,
                    params['t0'],
                    params['period'],
                    params['a'],
                    params['inc'],
                    params['e'],
                    params['w'],
                    'non-linear']
                for u in params['us']:
                    args.append(u)
                for r in params['rs']:
                    args.append(r)
                fs_b = harmonica_transit(*args)

                # Check algebraic gradients match finite difference.
                res_iter = zip(fs_a, fs_b, algebraic_gradients)
                for res_idx, (f_a, f_b, algebraic_grad) in enumerate(res_iter):
                    finite_diff_grad = (f_b - f_a) / self.epsilon
                    grad_err = np.abs(finite_diff_grad - algebraic_grad)
                    self.assertLess(
                        grad_err, self.grad_tol,
                        msg='df/d{} failed no.{}.'.format(param_name, res_idx))


if __name__ == '__main__':
    unittest.main()
