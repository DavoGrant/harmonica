import unittest
import numpy as np
import jax.numpy as jnp
from jax import jit, jvp, grad

from harmonica.jax import harmonica_transit_quad_ld, \
    harmonica_transit_nonlinear_ld
from harmonica.jax.custom_primitives import jax_light_curve_quad_ld_prim, \
    jax_light_curve_nonlinear_ld_prim


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

    def test_custom_jax_primitive_quad_ld(self):
        """ Test custom jax primitive for quadratic limb-darkening. """
        n_dp = 1000
        self._build_test_data_structures(n_dp=n_dp)
        u1 = 0.1
        u2 = 0.5
        r = jnp.array([0.1, -0.003, 0.])
        args_struc = [self.t0, self.period, self.a, self.inc,
                      self.ecc_non_zero, self.omega, u1, u2]
        for rn in r:
            args_struc.append(rn)

        # Broadcast input params to same length as times.
        times, *args_struc = jnp.broadcast_arrays(self.times, *args_struc)

        # Check JIT.
        f, df_dz = jit(jax_light_curve_quad_ld_prim)(times, *args_struc)
        self.assertEqual(f.shape, self.times.shape)
        self.assertEqual(df_dz.ndim, 2)
        self.assertEqual(df_dz.shape[0], self.times.shape[0])
        self.assertEqual(df_dz.shape[1], 6 + 2 + 3)
        self.assertEqual(np.sum(np.isfinite(f)), n_dp)
        self.assertEqual(np.sum(np.isfinite(df_dz)), n_dp * (6 + 2 + 3))

        # Check JVP.
        der_jit = jit(lambda arg_values, arg_tangents: jvp(
            jax_light_curve_quad_ld_prim, arg_values, arg_tangents))
        input_tangents = tuple(jnp.ones(n_dp) for p in range(
            (len((times, *args_struc)))))
        (f, df_dz), (jacobian_vp, _) = der_jit(
            (self.times, *args_struc), input_tangents)
        self.assertEqual(f.shape, jacobian_vp.shape)
        self.assertEqual(np.sum(np.isfinite(jacobian_vp)), n_dp)

    def test_custom_jax_primitive_nonlinear_ld(self):
        """ Test custom jax primitive for non-linear limb-darkening. """
        n_dp = 1000
        self._build_test_data_structures(n_dp=n_dp)
        u1 = 0.33
        u2 = 0.96
        u3 = -0.68
        u4 = 0.17
        r = jnp.array([0.1, -0.003, 0.])
        args_struc = [self.t0, self.period, self.a, self.inc,
                      self.ecc_non_zero, self.omega, u1, u2, u3, u4]
        for rn in r:
            args_struc.append(rn)

        # Broadcast input params to same length as times.
        times, *args_struc = jnp.broadcast_arrays(self.times, *args_struc)

        # Check JIT.
        f, df_dz = jit(jax_light_curve_nonlinear_ld_prim)(times, *args_struc)
        self.assertEqual(f.shape, self.times.shape)
        self.assertEqual(df_dz.ndim, 2)
        self.assertEqual(df_dz.shape[0], self.times.shape[0])
        self.assertEqual(df_dz.shape[1], 6 + 4 + 3)
        self.assertEqual(np.sum(np.isfinite(f)), n_dp)
        self.assertEqual(np.sum(np.isfinite(df_dz)), n_dp * (6 + 4 + 3))

        # Check JVP.
        der_jit = jit(lambda arg_values, arg_tangents: jvp(
            jax_light_curve_nonlinear_ld_prim, arg_values, arg_tangents))
        input_tangents = tuple(jnp.ones(n_dp) for p in range(
            (len((times, *args_struc)))))
        (f, df_dz), (jacobian_vp, _) = der_jit(
            (self.times, *args_struc), input_tangents)
        self.assertEqual(f.shape, jacobian_vp.shape)
        self.assertEqual(np.sum(np.isfinite(jacobian_vp)), n_dp)

    def test_api_jax_quad_ld(self):
        """ Test jax api for quadratic limb-darkening. """
        n_dp = 1000
        self._build_test_data_structures(n_dp=n_dp)

        # Check circle.
        f = harmonica_transit_quad_ld(
            self.times, self.t0, self.period, self.a, self.inc)
        self.assertEqual(f.shape, self.times.shape)
        self.assertEqual(np.sum(np.isfinite(f)), n_dp)

        # Check n_rs = 3.
        f = harmonica_transit_quad_ld(
            self.times, self.t0, self.period, self.a, self.inc,
            self.ecc_zero, self.omega, u1=0.1, u2=0.5,
            r=jnp.array([0.1, -0.003, 0.]))
        self.assertEqual(f.shape, self.times.shape)
        self.assertEqual(np.sum(np.isfinite(f)), n_dp)

        # Check n_rs = 7.
        f = harmonica_transit_quad_ld(
            self.times, self.t0, self.period, self.a, self.inc,
            self.ecc_zero, self.omega, u1=0.1, u2=0.5,
            r=jnp.array([0.1, -0.003, 0., 0., 0., 0., 0.001]))
        self.assertEqual(f.shape, self.times.shape)
        self.assertEqual(np.sum(np.isfinite(f)), n_dp)

    def test_api_jax_nonlinear_ld(self):
        """ Test jax api for non-linear limb-darkening. """
        n_dp = 1000
        self._build_test_data_structures(n_dp=n_dp)

        # Check circle.
        f = harmonica_transit_nonlinear_ld(
            self.times, self.t0, self.period, self.a, self.inc)
        self.assertEqual(f.shape, self.times.shape)
        self.assertEqual(np.sum(np.isfinite(f)), n_dp)

        # Check n_rs = 3.
        f = harmonica_transit_nonlinear_ld(
            self.times, self.t0, self.period, self.a, self.inc,
            self.ecc_zero, self.omega, u1=0.1, u2=0.5, u3=-0.1, u4=0.,
            r=jnp.array([0.1, -0.003, 0.]))
        self.assertEqual(f.shape, self.times.shape)
        self.assertEqual(np.sum(np.isfinite(f)), n_dp)

        # Check n_rs = 7.
        f = harmonica_transit_nonlinear_ld(
            self.times, self.t0, self.period, self.a, self.inc,
            self.ecc_zero, self.omega, u1=0.1, u2=0.5, u3=-0.1, u4=0.,
            r=jnp.array([0.1, -0.003, 0., 0., 0., 0., 0.001]))
        self.assertEqual(f.shape, self.times.shape)
        self.assertEqual(np.sum(np.isfinite(f)), n_dp)

    def test_flux_derivative_quad_ld(self):
        """ Test flux derivative for quadratic limb-darkening. """
        z_idxs = [1, 2, 3, 4, 5, 6, 7, 9]
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
                u1 = np.random.uniform(0., 1.)
                u2 = np.random.uniform(-1, 1.)
                us = np.array([u1, u2])

                # Uniform distributions of transmission string coeffs.
                n_rs = 2 * (np.random.randint(3, 9) // 2) + 1
                a0 = np.random.uniform(0.05, 1.5)
                rs = np.append([a0], a0 * np.random.uniform(
                    -0.01, 0.01, n_rs - 1))

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
                self._build_test_data_structures(n_dp=100)
                fs_a = harmonica_transit_quad_ld(
                    self.times, params['t0'], params['period'], params['a'],
                    params['inc'], params['e'], params['w'], params['us'][0],
                    params['us'][1], params['rs'])

                # Update one parameter by epsilon.
                updated_params = params
                if param_name == 'us':
                    u_idx = np.random.randint(0, 2)
                    us[u_idx] += self.epsilon
                    updated_params['us'] = us
                    _param_idx = param_idx + u_idx
                elif param_name == 'rs':
                    r_idx = np.random.randint(0, n_rs)
                    rs[r_idx] += self.epsilon
                    updated_params['rs'] = rs
                    _param_idx = param_idx + r_idx
                else:
                    updated_params[param_name] = updated_params[param_name] \
                                                 + self.epsilon
                    _param_idx = param_idx

                # Get gradients.
                algebraic_gradients = []
                for j in range(self.times.shape[0]):
                    param_lst = [params['t0'], params['period'],  params['a'],
                                 params['inc'], params['e'], params['w'],
                                 params['us'][0], params['us'][1]]
                    for rn in params['rs']:
                        param_lst.append(rn)
                    scalar_prim = lambda t, *p: \
                        jax_light_curve_quad_ld_prim(t, *p)[0]
                    algebraic_gradients.append(
                        grad(scalar_prim, argnums=_param_idx)(
                            self.times[j], *param_lst))

                # Compute fluxes with updated parameter set.
                self._build_test_data_structures(n_dp=100)
                fs_b = harmonica_transit_quad_ld(
                    self.times, updated_params['t0'],
                    updated_params['period'], updated_params['a'],
                    updated_params['inc'], updated_params['e'],
                    updated_params['w'], updated_params['us'][0],
                    updated_params['us'][1], updated_params['rs'])

                # Check algebraic gradients match finite difference.
                res_iter = zip(fs_a, fs_b, algebraic_gradients)
                for res_idx, (f_a, f_b, algebraic_grad) in enumerate(res_iter):
                    finite_diff_grad = (f_b - f_a) / self.epsilon
                    grad_err = np.abs(finite_diff_grad - algebraic_grad)

                    self.assertLess(
                        grad_err, self.grad_tol,
                        msg='df/d{} failed lc {} dp {}.'.format(
                            param_name, i, res_idx))

    def test_flux_derivative_nonlinear_ld(self):
        """ Test flux derivative for non-linear limb-darkening. """
        z_idxs = [1, 2, 3, 4, 5, 6, 7, 11]
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
                us = np.random.uniform(-1., 1., 4)

                # Uniform distributions of transmission string coeffs.
                n_rs = 2 * (np.random.randint(3, 9) // 2) + 1
                a0 = np.random.uniform(0.05, 1.5)
                rs = np.append([a0], a0 * np.random.uniform(
                    -0.01, 0.01, n_rs - 1))

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
                self._build_test_data_structures(n_dp=100)
                fs_a = harmonica_transit_nonlinear_ld(
                    self.times, params['t0'], params['period'], params['a'],
                    params['inc'], params['e'], params['w'], params['us'][0],
                    params['us'][1], params['us'][2], params['us'][3],
                    params['rs'])

                # Update one parameter by epsilon.
                updated_params = params
                if param_name == 'us':
                    u_idx = np.random.randint(0, 4)
                    us[u_idx] += self.epsilon
                    updated_params['us'] = us
                    _param_idx = param_idx + u_idx
                elif param_name == 'rs':
                    r_idx = np.random.randint(0, n_rs)
                    rs[r_idx] += self.epsilon
                    updated_params['rs'] = rs
                    _param_idx = param_idx + r_idx
                else:
                    updated_params[param_name] = updated_params[param_name] \
                                                 + self.epsilon
                    _param_idx = param_idx

                # Get gradients.
                algebraic_gradients = []
                for j in range(self.times.shape[0]):
                    param_lst = [params['t0'], params['period'],  params['a'],
                                 params['inc'], params['e'], params['w'],
                                 params['us'][0], params['us'][1],
                                 params['us'][2], params['us'][3]]
                    for rn in params['rs']:
                        param_lst.append(rn)
                    scalar_prim = lambda t, *p: \
                        jax_light_curve_nonlinear_ld_prim(t, *p)[0]
                    algebraic_gradients.append(
                        grad(scalar_prim, argnums=_param_idx)(
                            self.times[j], *param_lst))

                # Compute fluxes with updated parameter set.
                self._build_test_data_structures(n_dp=100)
                fs_b = harmonica_transit_nonlinear_ld(
                    self.times, updated_params['t0'],
                    updated_params['period'], updated_params['a'],
                    updated_params['inc'], updated_params['e'],
                    updated_params['w'], updated_params['us'][0],
                    updated_params['us'][1], updated_params['us'][2],
                    updated_params['us'][3], updated_params['rs'])

                # Check algebraic gradients match finite difference.
                res_iter = zip(fs_a, fs_b, algebraic_gradients)
                for res_idx, (f_a, f_b, algebraic_grad) in enumerate(res_iter):
                    finite_diff_grad = (f_b - f_a) / self.epsilon
                    grad_err = np.abs(finite_diff_grad - algebraic_grad)

                    self.assertLess(
                        grad_err, self.grad_tol,
                        msg='df/d{} failed lc {} dp {}.'.format(
                            param_name, i, res_idx))


if __name__ == '__main__':
    unittest.main()
