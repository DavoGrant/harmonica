import numpy as np

from .core import example


class HarmonicLimbMap(object):
    """
    Harmonic limb mapping class.

    Compute light curves for exoplanet transmission mapping through
    parameterising the planet shape as a Fourier series.

    Parameters
    ----------
    x : (N,) array_like
        A 1-D array of real values.
    y : (...,N,...) array_like
        A N-D array of real values. The length of `y` along the interpolation
        axis must be equal to the length of `x`.

    Attributes
    ----------
    fill_value

    Methods
    -------
    __call__

    Notes
    -----
    Calling `interp1d` with NaNs present in input values results in
    undefined behaviour.

    Input values `x` and `y` must be convertible to `float` values like
    `int` or `float`.

    Examples
    --------


    """

    def __init__(self, verbose=False):
        self.verbose = verbose

    def __repr__(self):
        return '<Harmonic limb mapper>'

    def method(self):
        a = example.add(1, 2)
        print(a)
        return
