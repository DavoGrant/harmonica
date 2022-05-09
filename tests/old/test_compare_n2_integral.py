import numpy as np
from scipy.integrate import nquad


# Config.
a_s = [0.1, 0.002]
b_s = [0.001]
c_s = [a_s[0]]
for a_n, b_n in zip(a_s[1:], b_s):
    c_s.append((a_n - 1j * b_n) / 2)
    c_s.insert(0, (a_n + 1j * b_n) / 2)
c_s = np.array(c_s)
N_c = int((len(c_s) - 1) / 2)
N_c_s = np.arange(-N_c, N_c + 1, 1)
theta_0 = 2.654604059665821 - 2. * np.pi
theta_1 = 0.46466780317831047
theta_2 = 2.654604059665821
d = 0.95
nu = -np.pi/2

epsabs = 1e-15
epsrel = 1e-15

cosine_array = np.cos(nu) * np.array([0.5, 0, 0.5]) \
               + np.sin(nu) * np.array([0.5j, 0, -0.5j])
sine_array = np.cos(nu) * np.array([0.5j, 0, -0.5j]) \
             - np.sin(nu) * np.array([0.5, 0, 0.5])


def r_p(theta):
    _r_p = 0.
    for i, n, in enumerate(N_c_s):
        _r_p += c_s[i] * np.exp(1j * n * theta)
    return np.real(_r_p)


def s_numerical(r, theta):
    return (1 - d**2 - r**2 + 2 * d * r * np.cos(theta - nu)) * r


def lim_r_planet(theta):
    return [0., r_p(theta)]


def lim_theta_planet():
    return [theta_0, theta_1]


def lim_r_star(theta):
    return [0., d * np.cos(theta - nu)
                + (d**2 * np.cos(theta - nu)**2 - d**2 + 1)**0.5]


def lim_theta_star():
    return [theta_1, theta_2]


res_planet, err_planet = nquad(s_numerical, [lim_r_planet, lim_theta_planet],
                               opts={'epsabs': epsabs, 'epsrel': epsrel})
res_star, err_star = nquad(s_numerical, [lim_r_star, lim_theta_star],
                           opts={'epsabs': epsabs, 'epsrel': epsrel})
res_numerical = res_planet + res_star
print(res_numerical, (err_planet**2 + err_star**2)**0.5)


def add_arrays_centre_aligned(aa, bb):
    a_shape = aa.shape[0]
    b_shape = bb.shape[0]
    if a_shape == b_shape:
        x_n = aa + bb
    elif a_shape > b_shape:
        x_n = aa + np.pad(bb, (a_shape - b_shape) // 2,
                          'constant', constant_values=0.)
    else:
        x_n = np.pad(aa, (b_shape - a_shape) // 2,
                     'constant', constant_values=0.) + bb
    return x_n


def compute_q_s():
    a = np.convolve(c_s, c_s)
    b = -d * np.convolve(cosine_array, c_s)
    c = -d * np.convolve(sine_array, 1j * N_c_s * c_s)
    rhs = add_arrays_centre_aligned(add_arrays_centre_aligned(a, b), c)

    f = 2. - d**2
    g = -np.convolve(c_s, c_s)
    h = 2. * d * np.convolve(cosine_array, c_s)
    lhs = add_arrays_centre_aligned(add_arrays_centre_aligned(np.array([f]), g), h)

    q_s = 1. / 4 * np.convolve(lhs, rhs)
    return q_s


# import matplotlib.pyplot as plt
#
#
# s0 = []
# for tt in np.linspace(0, 2 * np.pi, 100):
#     rpt = 0.
#     dd_rpt = 0.
#     for i, n, in enumerate(N_c_s):
#         rpt += c_s[i] * np.exp(1j * n * tt)
#         dd_rpt += 1j * n * c_s[i] * np.exp(1j * n * tt)
#     bla = 1. / 4 * (1 + (1 - d**2 - rpt**2 + 2. * d * rpt * np.cos(tt - nu))) \
#           * (rpt**2 - d * np.cos(tt - nu) * rpt - d * np.sin(tt - nu) * dd_rpt)
#     s0.append(bla)
#
# q_s = compute_q_s()
# ss = []
# for tt in np.linspace(0, 2 * np.pi, 100):
#     ccv = 0.
#     N_q = int((len(q_s) - 1) / 2)
#     N_q_s = np.arange(-N_q, N_q + 1, 1)
#     for i, n, in enumerate(N_q_s):
#         ccv += q_s[i] * np.exp(1j * n * tt)
#     ss.append(ccv)
#
# plt.plot(np.linspace(0, 2 * np.pi, 100), s0)
# plt.plot(np.linspace(0, 2 * np.pi, 100), ss)
# print(np.max(np.array(ss) - np.array(s0)))
# plt.show()


def s_analytical():
    # Planet limb piece.
    q_s = compute_q_s()
    _s = 0.
    N_q = int((len(q_s) - 1) / 2)
    N_q_s = np.arange(-N_q, N_q + 1, 1)
    for i, n, in enumerate(N_q_s):
        if n == 0:
            _s += q_s[i] * (theta_1 - theta_0)
        else:
            _s += q_s[i] / (1j * n) * (np.exp(1j * n * theta_1)
                                       - np.exp(1j * n * theta_0))

    # Stellar limb piece.
    phi_1 = np.arctan2(-r_p(theta_1) * np.sin(theta_1 - nu),
                       -r_p(theta_1) * np.cos(theta_1 - nu) + d)
    phi_2 = np.arctan2(-r_p(theta_2) * np.sin(theta_2 - nu),
                       -r_p(theta_2) * np.cos(theta_2 - nu) + d)
    _p = 1. / 4 * (phi_2 - phi_1)

    return np.real(_s) + _p


res_ana = s_analytical()
print(res_ana)

print(res_numerical - res_ana)

