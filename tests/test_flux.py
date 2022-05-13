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

        # Todo: test egde cases, eg circle, d=0 a=1, rp>1 etc.
        # Todo: promote tp pre-compute anthing idep loc, eg c conv c.

    def test_flux_data_structures(self):
        """ Test flux data structures. """
        limb_dark_law = 0
        us = np.array([0.40, 0.29], dtype=np.float64, order='C')
        rs = np.array([0.1, 0.002, 0.001], dtype=np.float64, order='C')
        ds = np.ascontiguousarray(np.linspace(2., 0.1, 1), dtype=np.float64)
        nus = np.ascontiguousarray(np.linspace(0.01, np.pi/2, len(ds)), dtype=np.float64)
        fs = np.empty(ds.shape, dtype=np.float64, order='C')
        n_od = ds.shape + (6,)
        ds_grad = np.empty(n_od, dtype=np.float64, order='C')
        nus_grad = np.empty(n_od, dtype=np.float64, order='C')
        n_lcd = ds.shape + (6 + len(us) + len(rs),)
        fs_grad = np.empty(n_lcd, dtype=np.float64, order='C')

        bindings.light_curve(limb_dark_law, us, rs, ds, nus, fs,
                             ds_grad, nus_grad, fs_grad,
                             require_gradients=False)
