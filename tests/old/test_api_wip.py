import time
import batman
import numpy as np
import matplotlib.pyplot as plt
from harmonica import HarmonicaTransit


# Harmonica transit light curve.
s = time.time()
ts = np.linspace(4.5, 5.5, 1000)
ht = HarmonicaTransit(times=ts, require_gradients=False)
for i in range(1000):
    ht.set_orbit(t0=5., period=10., a=7., inc=88. * np.pi / 180.,
                 ecc=0., omega=41. * np.pi / 180)
    ht.set_stellar_limb_darkening(np.array([0.40, 0.29]), limb_dark_law='quadratic')
    ht.set_planet_transmission_string(np.array([0.1]))
    fs = ht.get_transit_light_curve()
print((time.time() - s) / 1000)

# Batman transit light curve.
params = batman.TransitParams()
params.t0 = 5.
params.per = 10.
params.rp = 0.1
params.a = 7.
params.inc = 88.
params.ecc = 0.
params.w = 41.
params.u = [0.40, 0.29]
params.limb_dark = "quadratic"
m = batman.TransitModel(params, ts, max_err=0.01)
fs_batman = m.light_curve(params)

plt.plot(ts, (fs - fs_batman) * 1e6)
plt.show()
