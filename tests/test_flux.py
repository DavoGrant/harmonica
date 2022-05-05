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

        # Build test input data structures.
        self.ds = np.empty(self.times.shape, dtype=np.float64, order='C')
        self.nus = np.empty(self.times.shape, dtype=np.float64, order='C')
        n_od = self.times.shape + (6,)
        self.ds_grad = np.empty(n_od, dtype=np.float64, order='C')
        self.nus_grad = np.empty(n_od, dtype=np.float64, order='C')

    def test_flux_data_structures(self):
        """ Test flux data structures. """
        return
