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

    def test_flux_data_structures(self):
        """ Test flux data structures. """
        limb_dark_law = 0
        us = np.array([0.40, 0.29], dtype=np.float64, order='C')
        rs = np.array([0.1, 0.002, 0.001, -0.003, 0.004], dtype=np.float64, order='C')
        ds = np.ascontiguousarray(np.linspace(2., 0., 1000), dtype=np.float64)
        nus = np.ascontiguousarray([0.01] * 1000, dtype=np.float64)
        fs = np.empty(ds.shape, dtype=np.float64, order='C')
        n_od = ds.shape + (6,)
        ds_grad = np.empty(n_od, dtype=np.float64, order='C')
        nus_grad = np.empty(n_od, dtype=np.float64, order='C')
        n_lcd = ds.shape + (6 + len(us) + len(rs),)
        fs_grad = np.empty(n_lcd, dtype=np.float64, order='C')

        bindings.light_curve(limb_dark_law, us, rs, ds, nus, fs,
                             ds_grad, nus_grad, fs_grad,
                             require_gradients=False)

    def test_get_flux_derivatives_working(self):
        """ Test bla. """
        times = np.linspace(4.57, 5, 1000)
        limb_dark_law = 0
        us = np.array([0.40, 0.29], dtype=np.float64, order='C')
        # todo - check correct derivatives
        rs = np.array([0.1, 0.001, 0.001, 0.001, 0.001], dtype=np.float64, order='C')

        ds = np.empty(times.shape, dtype=np.float64, order='C')
        nus = np.empty(times.shape, dtype=np.float64, order='C')
        fs = np.empty(times.shape, dtype=np.float64, order='C')

        n_od = ds.shape + (6,)
        ds_grad = np.empty(n_od, dtype=np.float64, order='C')
        nus_grad = np.empty(n_od, dtype=np.float64, order='C')
        n_lcd = ds.shape + (6 + len(us) + len(rs),)
        fs_grad = np.ones(n_lcd, dtype=np.float64)

        bindings.orbit(5., 10., 7., 88. / 180. * np.pi, 0.05, 0.,
                       times, ds, nus,
                       ds_grad, nus_grad, require_gradients=True)
        import time
        s = time.time()
        for i in range(100):
            bindings.light_curve(limb_dark_law, us, rs, ds, nus, fs,
                                 ds_grad, nus_grad, fs_grad, 20, 50,
                                 require_gradients=False)
        print(fs_grad)
        print((time.time() - s) / 100)

    def test_flux_derivative_df_dz(self):
        """ Test orbit derivative eccentric dnu_dz, z={t0, p, a, i, e, w}. """
        times = np.array([5.23])
        limb_dark_law = 0
        r0_a = 0.001
        r_idx = 4
        us = np.array([0.1, 0.5], dtype=np.float64, order='C')
        rs = np.array([0.1, -0.003, 0.003, -0.003, 0.003, -0.003, 0.003, -0.003, 0.003], dtype=np.float64, order='C')
        rs[r_idx] = r0_a

        ds = np.empty(times.shape, dtype=np.float64, order='C')
        nus = np.empty(times.shape, dtype=np.float64, order='C')
        fs = np.empty(times.shape, dtype=np.float64, order='C')

        n_od = ds.shape + (6,)
        ds_grad = np.empty(n_od, dtype=np.float64, order='C')
        nus_grad = np.empty(n_od, dtype=np.float64, order='C')
        n_lcd = ds.shape + (6 + len(us) + len(rs),)
        fs_grad = np.empty(n_lcd, dtype=np.float64)

        bindings.orbit(5., 10., 7., 88. / 180. * np.pi, 0.05, 1.,
                       times, ds, nus,
                       ds_grad, nus_grad, require_gradients=True)
        bindings.light_curve(limb_dark_law, us, rs, ds, nus, fs,
                             ds_grad, nus_grad, fs_grad, 50, 500,
                             require_gradients=True)

        fs_a = np.copy(fs)
        df_dt0_a = np.copy(fs_grad[:, 8 + r_idx])

        r0_b = r0_a + self.epsilon
        rs[r_idx] = r0_b

        bindings.orbit(5., 10., 7., 88. / 180. * np.pi, 0.05, 1.,
                       times, ds, nus,
                       ds_grad, nus_grad, require_gradients=True)
        bindings.light_curve(limb_dark_law, us, rs, ds, nus, fs,
                             ds_grad, nus_grad, fs_grad, 50, 500,
                             require_gradients=True)
        fs_b = np.copy(fs)

        import matplotlib.pyplot as plt

        def grad_arrow(x_draw, x, y, grad):
            c = y - grad * x
            return grad * x_draw + c

        plt.scatter(r0_a, fs_a, label='$u_1$')
        plt.scatter(r0_b, fs_b, label='$u_1 + \delta$: $\delta={}$'.format(self.epsilon))
        x_arrow = np.linspace(r0_a, r0_b, 2)
        plt.plot(x_arrow, grad_arrow(x_arrow, r0_a, fs_a, df_dt0_a),
                 label='Gradient: $\\frac{dF}{d u_1}$')
        print(fs_a[0], fs_b[0], ((df_dt0_a * self.epsilon + fs_a) - fs_b)[0])

        plt.legend()
        plt.xlabel('$u_1$')
        plt.ylabel('$F$')
        plt.tight_layout()
        plt.show()

