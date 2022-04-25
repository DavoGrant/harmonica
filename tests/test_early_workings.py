import numpy as np
from harmonica import HarmonicLimbMap


# Test instantiate harmonic limb mapper.
hlm = HarmonicLimbMap()
print(hlm)

# Test orbit.
hlm.set_orbit(t0=5., period=10., a=7., inc=85. * np.pi / 180.,
              ecc=0.1, omega=0. * np.pi / 180)
hlm.get_transit_light_curve(times=np.linspace(0., 10., 10000))
