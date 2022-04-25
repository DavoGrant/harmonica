import numpy as np
from harmonica import HarmonicaTransit


# Test instantiate harmonic limb mapper.
ht = HarmonicaTransit(times=np.linspace(0, 10, 1000),
                      require_gradients=False)
print(ht)

# Test orbit.
ht.set_orbit(t0=5., period=10., a=7., inc=88. * np.pi / 180.,
             ecc=0.05, omega=41. * np.pi / 180)
ht.get_transit_light_curve()
