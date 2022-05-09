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

    def test_flux_data_structures(self):
        """ Test flux data structures. """
        bindings.light_curve(require_gradients=False)
