import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from harmonica import HarmonicaTransit


# Config.
t0 = 5.
period = 10.
a = 7.
inc = 88. * np.pi / 180.
us = np.array([1., 0.40, 0.29])
B = np.array([[1., -1., -1.],
              [0., 1., 2.],
              [0., 0., -1.]])
ps = np.matmul(B, us)
I_0 = 1. / ((1. - us[1] / 3. - us[2] / 6.) * np.pi)
_as = np.array([0.1, 0.001, 0.001])
_bs = np.array([0.001, 0.001])
rs = [_as[0]]
for i in range(1, len(_as)):
    rs.append(_as[i])
    rs.append(_bs[i - 1])
rs = np.array(rs)
N_c = int((len(rs) - 1) / 2)
N_c_s = np.arange(-N_c, N_c + 1, 1)
ts = np.linspace(4.7, 5.3, 1000)
epsilon = 1.e-7


def generate_complex_fourier_coeffs():
    _cs = [_as[0]]
    for _a, _b in zip(_as[1:], _bs):
        _cs.append((_a - 1j * _b) / 2)
        _cs.insert(0, (_a + 1j * _b) / 2)
    return _cs


def r_p(_theta):
    _r_p = _as[0]
    for n, (_a, _b) in enumerate(zip(_as[1:], _bs)):
        _r_p += _a * np.cos((n + 1) * _theta)
        _r_p += _b * np.sin((n + 1) * _theta)
    return _r_p


def stellar_radius(_theta, _d, _nu, plus_solution=1):
    dcos = _d * np.cos(_theta - _nu)
    sqrt_term = (dcos**2 - _d**2 + 1.)**0.5
    if plus_solution == 1:
        return dcos + sqrt_term
    else:
        return dcos - sqrt_term


def a_p_values(p, _cs, _d, _nu):
    a_p = 0.
    if 0 <= p < N_c - 1:
        for n in range(-N_c, -N_c + p + 1):
            a_p += _cs[n + N_c] * _cs[(p - n - 2 * N_c) + N_c]
    elif N_c - 1 <= p < N_c + 1:
        a_p += -_d * np.exp(1j * _nu) * _cs[(p - 2 * N_c + 1) + N_c]
        for n in range(-N_c, -N_c + p + 1):
            a_p += _cs[n + N_c] * _cs[(p - n - 2 * N_c) + N_c]
    elif N_c + 1 <= p < 2 * N_c:
        a_p += -_d * np.exp(1j * _nu) * _cs[(p - 2 * N_c + 1) + N_c]
        a_p += -_d * np.exp(-1j * _nu) * _cs[(p - 2 * N_c - 1) + N_c]
        for n in range(-N_c, -N_c + p + 1):
            a_p += _cs[n + N_c] * _cs[(p - n - 2 * N_c) + N_c]
    elif p == 2 * N_c:
        a_p += -_d * np.exp(1j * _nu) * _cs[(p - 2 * N_c + 1) + N_c]
        a_p += -_d * np.exp(-1j * _nu) * _cs[(p - 2 * N_c - 1) + N_c]
        a_p += _d**2 - 1.
        for n in range(-N_c, N_c + 1):
            a_p += _cs[n + N_c] * _cs[(p - n - 2 * N_c) + N_c]
    elif 2 * N_c + 1 <= p < 3 * N_c:
        a_p += -_d * np.exp(1j * _nu) * _cs[(p - 2 * N_c + 1) + N_c]
        a_p += -_d * np.exp(-1j * _nu) * _cs[(p - 2 * N_c - 1) + N_c]
        for n in range(-3 * N_c + p, N_c + 1):
            a_p += _cs[n + N_c] * _cs[(p - n - 2 * N_c) + N_c]
    elif 3 * N_c - 0 <= p < 3 * N_c + 2:
        a_p += -_d * np.exp(-1j * _nu) * _cs[(p - 2 * N_c - 1) + N_c]
        for n in range(-3 * N_c + p, N_c + 1):
            a_p += _cs[n + N_c] * _cs[(p - n - 2 * N_c) + N_c]
    elif 3 * N_c + 2 <= p < 4 * N_c + 1:
        for n in range(-3 * N_c + p, N_c + 1):
            a_p += _cs[n + N_c] * _cs[(p - n - 2 * N_c) + N_c]

    return a_p


def c_jk_values(j, k, _cs, _d, _nu):
    if k == 4 * N_c:
        return -a_p_values(j - 1, _cs, _d, _nu) / a_p_values(4 * N_c, _cs, _d, _nu)
    else:
        if j == k + 1:
            return 1.
        else:
            return 0.


def companion_matrix_verbose(_cs, _d, _nu):
    assert N_c >= 1
    C = np.zeros((4 * N_c, 4 * N_c), dtype=np.cdouble)
    for (j, k), value in np.ndenumerate(C):
        C[j, k] = c_jk_values(j + 1, k + 1, _cs, _d, _nu)
    return C


def derivative_intersection_eqn(_theta, _d, _nu, plus_solution=1):
    r_dash = np.zeros(_theta.shape)
    for n, (_a, _b) in enumerate(zip(_as[1:], _bs)):
        r_dash += -_a * (n + 1) * np.sin((n + 1) * _theta)
        r_dash += _b * (n + 1) * np.cos((n + 1) * _theta)
    dsin = _d * np.sin(_theta - _nu)
    dcos = _d * np.cos(_theta - _nu)
    sqrt_term = (dcos**2 - d**2 + 1.)**0.5
    r_dash += dsin
    if plus_solution == 1:
        return r_dash + (dsin * dcos) / sqrt_term
    else:
        return r_dash - (dsin * dcos) / sqrt_term


def find_intersections(_cs, _d, _nu):
    # Get intersections.
    c_matrix = companion_matrix_verbose(_cs, _d, _nu)
    eigenvalues = np.linalg.eigvals(c_matrix)
    intersections = [np.angle(r) for r in eigenvalues
                     if 1. - epsilon < np.abs(r) < 1. + epsilon]

    # Characterise intersections.
    intersection_types = []
    if len(intersections) == 0:
        rp_nu = r_p(_nu)
        if d <= 1.:
            if rp_nu < 1. + d:
                intersections = [_nu - np.pi, _nu + np.pi]
                intersection_types = [0]
            elif rp_nu > 1. + d:
                intersections = [-np.pi, np.pi]
                intersection_types = [1]
        else:
            if rp_nu < 1. + d:
                intersections = []
                intersection_types = [2]
            elif rp_nu > 1. + d:
                intersections = [-np.pi, np.pi]
                intersection_types = [1]

    else:
        intersections = sorted(intersections)
        intersections.append(intersections[0] + 2 * np.pi)
        intersections = np.array(intersections)
        for j in range(len(intersections) - 1):

            # Check circle association of j and j+1.
            if d <= 1.:
                T_theta_j = 1
                T_theta_j_plus_1 = 1
            else:
                residuals_T_theta_j = \
                    r_p(intersections[j]) - stellar_radius(
                        intersections[j], _d, _nu, plus_solution=1)
                residuals_T_theta_j_plus_1 = \
                    r_p(intersections[j]) - stellar_radius(
                        intersections[j + 1], _d, _nu, plus_solution=1)
                if residuals_T_theta_j < epsilon:
                    T_theta_j = 1
                else:
                    T_theta_j = 0
                if residuals_T_theta_j_plus_1 < epsilon:
                    T_theta_j_plus_1 = 1
                else:
                    T_theta_j_plus_1 = 0

            # Check derivative of j and j+1.
            derivatives_T_theta_j = derivative_intersection_eqn(
                intersections[j], _d, _nu, plus_solution=T_theta_j)
            derivatives_T_theta_j_plus_1 = derivative_intersection_eqn(
                intersections[j + 1], _d, _nu, plus_solution=T_theta_j_plus_1)
            if derivatives_T_theta_j > 0.:
                dT_theta_j = 1
            else:
                dT_theta_j = 0
            if derivatives_T_theta_j_plus_1 > 0.:
                dT_theta_j_plus_1 = 1
            else:
                dT_theta_j_plus_1 = 0

            # Decide type from table.
            if T_theta_j == 1 and T_theta_j_plus_1 == 1 \
                    and dT_theta_j == 0 and dT_theta_j_plus_1 == 1:
                intersection_types.append(0)
            elif T_theta_j == 1 and T_theta_j_plus_1 == 1 \
                    and dT_theta_j == 1 and dT_theta_j_plus_1 == 0:
                intersection_types.append(1)
            elif T_theta_j == 0 and T_theta_j_plus_1 == 0 \
                    and dT_theta_j == 0 and dT_theta_j_plus_1 == 1:
                intersection_types.append(1)
            elif T_theta_j == 0 and T_theta_j_plus_1 == 0 \
                    and dT_theta_j == 1 and dT_theta_j_plus_1 == 0:
                intersection_types.append(0)
            elif T_theta_j == 1 and T_theta_j_plus_1 == 0 \
                    and dT_theta_j == 0 and dT_theta_j_plus_1 == 0:
                intersection_types.append(0)
            elif T_theta_j == 1 and T_theta_j_plus_1 == 0 \
                    and dT_theta_j == 1 and dT_theta_j_plus_1 == 1:
                intersection_types.append(1)
            elif T_theta_j == 0 and T_theta_j_plus_1 == 1 \
                    and dT_theta_j == 0 and dT_theta_j_plus_1 == 0:
                intersection_types.append(1)
            elif T_theta_j == 0 and T_theta_j_plus_1 == 1 \
                    and dT_theta_j == 1 and dT_theta_j_plus_1 == 1:
                intersection_types.append(0)

    return intersections, intersection_types


def drp_dtheta(_cs, _theta):
    _drp_dtheta = 0.
    for k, n, in enumerate(N_c_s):
        _drp_dtheta += 1j * n * _cs[k] * np.exp(1j * n * _theta)
    return np.real(_drp_dtheta)


def z_p(_cs, _theta, _d, _nu):
    return (1. - d**2 - r_p(_theta)**2 + 2 * d * r_p(_theta)
            * np.cos(_theta - _nu))**0.5


def zeta(_z_p, n_val):
    return (1 - _z_p**(n_val + 2)) / ((n_val + 2) * (1 - _z_p**2))


def eta(_theta, _cs, _d, _nu):
    return r_p(_theta)**2 - _d * np.cos(_theta - _nu) * r_p(_theta) \
           - d * np.sin(_theta - _nu) * drp_dtheta(_cs, _theta)


def line_integral_planet(_theta, _cs, _d, _nu, n_val):
    return zeta(z_p(_cs, _theta, _d, _nu), n_val) * eta(_theta, _cs, _d, _nu)


def line_integral_star(_theta, _cs, _d, _nu, n_val):
    return zeta(z_p(_cs, _theta, _d, _nu), n_val) * eta(_theta, _cs, _d, _nu)


def compute_numerical_flux(_cs, _d, _nu):

    # Find planet-stellar limb intersections.
    intersections, intersection_types = find_intersections(_cs, _d, _nu)

    # Iterate pairs of thetas computing line integral.
    alpha = 0.
    alpha_numerical_err = 0.
    for j in range(len(intersection_types)):
        if intersection_types[j] == 0:
            s0, s0_err = quad(
                line_integral_planet, intersections[j], intersections[j + 1],
                args=(_cs, _d, _nu, 0),
                epsabs=1.e-15, epsrel=1.e-15, limit=500)
            s1, s1_err = quad(
                line_integral_planet, intersections[j], intersections[j + 1],
                args=(_cs, _d, _nu, 1),
                epsabs=1.e-15, epsrel=1.e-15, limit=500)
            s2, s2_err = quad(
                line_integral_planet, intersections[j], intersections[j + 1],
                args=(_cs, _d, _nu, 2),
                epsabs=1.e-15, epsrel=1.e-15, limit=500)
            alpha += I_0 * (s0 * ps[0] + s1 * ps[1] + s2 * ps[2])
            alpha_numerical_err += I_0 * ((s0_err * ps[0])**2
                                          + (s1_err * ps[1])**2
                                          + (s2_err * ps[2])**2)**0.5
        elif intersection_types[j] == 1:
            phi_j = np.arctan2(
                -r_p(intersections[j]) * np.sin(intersections[j] - _nu),
                -r_p(intersections[j]) * np.cos(intersections[j] - _nu) + _d)
            phi_j_plus_1 = np.arctan2(
                -r_p(intersections[j + 1]) * np.sin(intersections[j + 1] - _nu),
                -r_p(intersections[j + 1]) * np.cos(intersections[j + 1] - _nu) + _d)
            s0 = 1. / (0. + 2) * (phi_j_plus_1 - phi_j)
            s1 = 1. / (1. + 2) * (phi_j_plus_1 - phi_j)
            s2 = 1. / (2. + 2) * (phi_j_plus_1 - phi_j)
            alpha += I_0 * (s0 * ps[0] + s1 * ps[1] + s2 * ps[2])
        else:
            break

    return 1. - alpha, alpha_numerical_err


# Harmonica transit light curve.
ht = HarmonicaTransit(times=ts, require_gradients=False)
ht.set_orbit(t0, period, a, inc)
ht.set_stellar_limb_darkening(us[1:], limb_dark_law='quadratic')
ht.set_planet_transmission_string(rs)
fs = ht.get_precision_estimate()

# Numerical light curve.
fs_numerical = []
fs_numerical_err = []
cs = generate_complex_fourier_coeffs()
for d, nu in zip(ht.ds, ht.nus):
    f_num, f_num_err = compute_numerical_flux(cs, d, nu)
    fs_numerical.append(f_num)
    fs_numerical_err.append(f_num_err)

fs_numerical = np.array(fs_numerical)
fs_numerical_err = np.array(fs_numerical_err)

fig = plt.figure(figsize=(8, 6))
ax1 = plt.subplot(1, 1, 1)
ax1.plot(ht.ds, np.abs(fs - fs_numerical), c='#000000')
ax1.plot(ht.ds, fs_numerical_err, c='#bc5090')
ax1.set_yscale('log')
ax1.set_xlabel('Time / days')
ax1.set_ylabel('Error')
plt.tight_layout()
plt.show()
