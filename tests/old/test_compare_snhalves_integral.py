import numpy as np
from scipy.integrate import nquad
from scipy.special import roots_legendre


# Config.
n_vals = np.array([0.5, 1., 1.5])
p_vals = np.array([0.3, 0.4, 0.2])
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
    sn_half = p_vals[0] * (1 - d**2 - r**2 + 2 * d * r
                           * np.cos(theta - nu))**(n_vals[0] / 2.) * r
    sn_one = p_vals[1] * (1 - d**2 - r**2 + 2 * d * r
                          * np.cos(theta - nu))**(n_vals[1] / 2.) * r
    sn_three_half = p_vals[2] * (1 - d**2 - r**2 + 2 * d * r
                                 * np.cos(theta - nu))**(n_vals[2] / 2.) * r

    return sn_half + sn_one + sn_three_half


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


def dr_p_dt(theta):
    _r_p = 0.
    for i, n, in enumerate(N_c_s):
        _r_p += 1j * n * c_s[i] * np.exp(1j * n * theta)
    return np.real(_r_p)


def z_p(theta):
    return (1 - d**2 - r_p(theta)**2 + 2 * d * r_p(theta) * np.cos(theta - nu))**0.5


def zeta(_z_p, n_val):
    return (1 - _z_p**(n_val + 2)) / ((n_val + 2) * (1 - _z_p**2))


def eta(theta):
    return r_p(theta)**2 - d * np.cos(theta - nu) * r_p(theta) \
           - d * np.sin(theta - nu) * dr_p_dt(theta)


def line_integral(theta):
    zn_half = p_vals[0] * zeta(z_p(theta), n_val=n_vals[0])
    zn_one = p_vals[1] * zeta(z_p(theta), n_val=n_vals[1])
    zn_three_half = p_vals[2] * zeta(z_p(theta), n_val=n_vals[2])

    return (zn_half + zn_one + zn_three_half) * eta(theta)


def s_approximate():
    # Planet limb piece.
    roots, weights = roots_legendre(10)
    t = (theta_1 - theta_0) / 2. * (roots + 1.) + theta_0
    _s = (theta_1 - theta_0) / 2. * line_integral(t).dot(weights)

    # Stellar limb piece.
    phi_1 = np.arctan2(-r_p(theta_1) * np.sin(theta_1 - nu),
                       -r_p(theta_1) * np.cos(theta_1 - nu) + d)
    phi_2 = np.arctan2(-r_p(theta_2) * np.sin(theta_2 - nu),
                       -r_p(theta_2) * np.cos(theta_2 - nu) + d)
    _pn_half = p_vals[0] * 1. / (n_vals[0] + 2.) * (phi_2 - phi_1)
    _pn_one = p_vals[1] * 1. / (n_vals[1] + 2.) * (phi_2 - phi_1)
    _pn_three_half = p_vals[2] * 1. / (n_vals[2] + 2.) * (phi_2 - phi_1)
    _p = _pn_half + _pn_one + _pn_three_half

    return np.real(_s) + _p


res_ana = s_approximate()
print(res_ana)

print(res_numerical - res_ana)

