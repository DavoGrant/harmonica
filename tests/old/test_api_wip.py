import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import nquad
from harmonica import HarmonicaTransit


# Harmonica transit light curve.
s = time.time()
ts = np.linspace(4.5, 5.5, 50)
ht = HarmonicaTransit(times=ts, n_l=100, require_gradients=False)
ht.set_orbit(t0=5., period=10., a=7., inc=88. * np.pi / 180.,
             ecc=0., omega=0. * np.pi / 180)
ht.set_stellar_limb_darkening(np.array([0.40, 0.29]), limb_dark_law='quadratic')
ht.set_planet_transmission_string(np.array([0.1, -0.001, 0.001]))
fs = ht.get_transit_light_curve()
print((time.time() - s) / 1)


# Numerical transit light curve.
def planet_radius(a_ns, b_ns, _theta):
    r_p = a_ns[0]
    for n, (a_n, b_n) in enumerate(zip(a_ns[1:], b_ns)):
        r_p += a_n * np.cos((n + 1) * _theta)
        r_p += b_n * np.sin((n + 1) * _theta)
    return r_p


def numerical_delta(r_prime, theta, _as, _bs, _d, _nu):
    _r = (r_prime**2 + _d**2 - 2. * r_prime * _d * np.cos(theta - _nu))**0.5
    if _r > 1.:
        return 0.
    _z = (1. - _r**2)**0.5
    _i0 = 1. / ((1. - 0.40 / 3. - 0.29 / 6.) * np.pi)
    intensity = 1 - 0.40 * (1 - _z) - 0.29 * (1 - _z)**2
    return _i0 * intensity * r_prime


def bounds_r_prime(theta, _as, _bs, _d, _nu):
    return [0., planet_radius(_as, _bs, theta)]


def bounds_theta(_as, _bs, _d, _nu):
    return [-np.pi, np.pi]


fs_numerical = []
fs_numerical_err = []
for d, nu in zip(ht.ds, ht.nus):
    # noinspection PyTupleAssignmentBalance
    delta, abserr = nquad(
        numerical_delta, [bounds_r_prime, bounds_theta],
        args=([0.1, -0.001], [0.001], d, nu),
        opts={'limit': 100, 'epsabs': 1e-10, 'epsrel': 1e-10})
    print(delta, abserr)
    fs_numerical.append(1. - delta)
    fs_numerical_err.append(abserr)

fs_numerical = np.array(fs_numerical)
fs_numerical_err = np.array(fs_numerical_err)

plt.plot(ts, (fs - fs_numerical) * 1e6, lw=1.5, c='#000000')
plt.plot(ts, fs_numerical_err * 1e6, lw=1.5, c='#ba7a12')
plt.plot(ts, -fs_numerical_err * 1e6, lw=1.5, c='#ba7a12')
plt.tight_layout()
plt.show()
