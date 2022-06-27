import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.special import roots_legendre


def generate_complex_fourier_coeffs(_as, _bs):
    _cs = [_as[0]]
    for _a, _b in zip(_as[1:], _bs):
        _cs.append((_a - 1j * _b) / 2)
        _cs.insert(0, (_a + 1j * _b) / 2)
    return _cs


# Config.
np.random.seed(123)
u1 = 0.4
u2 = 0.3
N_c = None
N_c_s = None
d = 0.98
nu = 0.1
N_l = 100
epsilon = 1.e-7
a_s = None
b_s = None


def r_p(_theta):
    _r_p = a_s[0]
    for n, (_a, _b) in enumerate(zip(a_s[1:], b_s)):
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


def da_p_values_dcn(p, _cs, _d, _nu, _n):
    a_p = 0.
    if 0 <= p < N_c - 1:
        for n in range(-N_c, -N_c + p + 1):
            if _n == n:
                a_p += _cs[(p - n - 2 * N_c) + N_c]
            if _n == p - n - 2 * N_c:
                a_p += _cs[n + N_c]
    elif N_c - 1 <= p < N_c + 1:
        for n in range(-N_c, -N_c + p + 1):
            if _n == n:
                a_p += _cs[(p - n - 2 * N_c) + N_c]
            if _n == p - n - 2 * N_c:
                a_p += _cs[n + N_c]
        if _n == p - 2 * N_c + 1:
            a_p += -d * np.exp(1j * _nu)
    elif N_c + 1 <= p < 2 * N_c:
        for n in range(-N_c, -N_c + p + 1):
            if _n == n:
                a_p += _cs[(p - n - 2 * N_c) + N_c]
            if _n == p - n - 2 * N_c:
                a_p += _cs[n + N_c]
        if _n == p - 2 * N_c + 1:
            a_p += -d * np.exp(1j * _nu)
        if _n == p - 2 * N_c - 1:
            a_p += -d * np.exp(-1j * _nu)
    elif p == 2 * N_c:
        for n in range(-N_c, N_c + 1):
            if _n == n:
                a_p += _cs[(p - n - 2 * N_c) + N_c]
            if _n == p - n - 2 * N_c:
                a_p += _cs[n + N_c]
        if _n == p - 2 * N_c + 1:
            a_p += -d * np.exp(1j * _nu)
        if _n == p - 2 * N_c - 1:
            a_p += -d * np.exp(-1j * _nu)
    elif 2 * N_c + 1 <= p < 3 * N_c:
        for n in range(-3 * N_c + p, N_c + 1):
            if _n == n:
                a_p += _cs[(p - n - 2 * N_c) + N_c]
            if _n == p - n - 2 * N_c:
                a_p += _cs[n + N_c]
        if _n == p - 2 * N_c + 1:
            a_p += -d * np.exp(1j * _nu)
        if _n == p - 2 * N_c - 1:
            a_p += -d * np.exp(-1j * _nu)
    elif 3 * N_c - 0 <= p < 3 * N_c + 2:
        for n in range(-3 * N_c + p, N_c + 1):
            if _n == n:
                a_p += _cs[(p - n - 2 * N_c) + N_c]
            if _n == p - n - 2 * N_c:
                a_p += _cs[n + N_c]
        if _n == p - 2 * N_c - 1:
            a_p += -d * np.exp(-1j * _nu)
    elif 3 * N_c + 2 <= p < 4 * N_c + 1:
        for n in range(-3 * N_c + p, N_c + 1):
            if _n == n:
                a_p += _cs[(p - n - 2 * N_c) + N_c]
            if _n == p - n - 2 * N_c:
                a_p += _cs[n + N_c]

    return a_p


def c_jk_values(j, k, _cs, _d, _nu):
    if k == 4 * N_c:
        return -a_p_values(j - 1, _cs, _d, _nu) / a_p_values(4 * N_c, _cs, _d, _nu)
    else:
        if j == k + 1:
            return 1.
        else:
            return 0.


def dc_jk_values_dcn(j, k, _cs, _d, _nu, _n):
    if k == 4 * N_c:
        return (a_p_values(j - 1, _cs, _d, _nu) * da_p_values_dcn(4 * N_c, _cs, _d, _nu, _n)
                - da_p_values_dcn(j - 1, _cs, _d, _nu, _n) * a_p_values(4 * N_c, _cs, _d, _nu)) \
               / a_p_values(4 * N_c, _cs, _d, _nu)**2
    else:
        if j == k + 1:
            return 0.
        else:
            return 0.


def companion_matrix_verbose(_cs, _d, _nu):
    assert N_c >= 1
    C = np.zeros((4 * N_c, 4 * N_c), dtype=np.cdouble)
    for (j, k), value in np.ndenumerate(C):
        C[j, k] = c_jk_values(j + 1, k + 1, _cs, _d, _nu)
    return C


def dcompanion_matrix_dcn(_cs, _d, _nu, _n):
    assert N_c >= 1
    dC_dcn = np.zeros((4 * N_c, 4 * N_c), dtype=np.cdouble)
    for (j, k), value in np.ndenumerate(dC_dcn):
        dC_dcn[j, k] = dc_jk_values_dcn(j + 1, k + 1, _cs, _d, _nu, _n)
    return dC_dcn


def derivative_intersection_eqn(_theta, _d, _nu, plus_solution=1):
    r_dash = np.zeros(_theta.shape)
    for n, (_a, _b) in enumerate(zip(a_s[1:], b_s)):
        r_dash += -_a * (n + 1) * np.sin((n + 1) * _theta)
        r_dash += _b * (n + 1) * np.cos((n + 1) * _theta)
    dsin = _d * np.sin(_theta - _nu)
    dcos = _d * np.cos(_theta - _nu)
    sqrt_term = (dcos**2 - _d**2 + 1.)**0.5
    r_dash += dsin
    if plus_solution == 1:
        return r_dash + (dsin * dcos) / sqrt_term
    else:
        return r_dash - (dsin * dcos) / sqrt_term


def find_intersections(_cs, _d, _nu):
    # Get intersections.
    c_matrix = companion_matrix_verbose(_cs, _d, _nu)
    eigenvalues, eigenvectors = np.linalg.eig(c_matrix)

    dtheta_dcnss = []
    for n in N_c_s:
        dC_dcn_total = dcompanion_matrix_dcn(_cs, _d, _nu, n)
        dlambda_dcn_total = np.diag(np.linalg.inv(eigenvectors) @ dC_dcn_total @ eigenvectors)

        intersections_not_sorted = []
        dtheta_dcns = []
        for idx, e_val in enumerate(eigenvalues):
            if 1. - epsilon < np.abs(e_val) < 1. + epsilon:
                intersections_not_sorted.append(np.angle(e_val))
                dtheta_dcns.append(-1j / e_val * dlambda_dcn_total[idx])

        dtheta_dcnss.append(dtheta_dcns)

    # Characterise intersections.
    # planet=0, entire planet=1, star=2, entire star=3, beyond=4
    intersection_types = []
    if len(intersections_not_sorted) == 0:
        rp_nu = r_p(_nu)
        if _d <= 1.:
            if rp_nu < 1. + _d:
                intersections = [_nu - np.pi, _nu + np.pi]
                intersection_types = [1]
            elif rp_nu > 1. + _d:
                intersections = [-np.pi, np.pi]
                intersection_types = [3]
        else:
            if rp_nu < 1. + _d:
                intersections = []
                intersection_types = [4]
            elif rp_nu > 1. + _d:
                intersections = [-np.pi, np.pi]
                intersection_types = [3]

    else:
        intersections = sorted(intersections_not_sorted)
        intersections.append(intersections[0] + 2 * np.pi)
        intersections = np.array(intersections)
        for n in N_c_s:
            dtheta_dcnss[n] = [x for _, x in sorted(zip(intersections_not_sorted, dtheta_dcnss[n]))]
            dtheta_dcnss[n].append(dtheta_dcnss[n][0])
            dtheta_dcnss[n] = np.array(dtheta_dcnss[n])
        for j in range(len(intersections) - 1):

            # Check circle association of j and j+1.
            if _d <= 1.:
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
                intersection_types.append(2)
            elif T_theta_j == 0 and T_theta_j_plus_1 == 0 \
                    and dT_theta_j == 0 and dT_theta_j_plus_1 == 1:
                intersection_types.append(2)
            elif T_theta_j == 0 and T_theta_j_plus_1 == 0 \
                    and dT_theta_j == 1 and dT_theta_j_plus_1 == 0:
                intersection_types.append(0)
            elif T_theta_j == 1 and T_theta_j_plus_1 == 0 \
                    and dT_theta_j == 0 and dT_theta_j_plus_1 == 0:
                intersection_types.append(0)
            elif T_theta_j == 1 and T_theta_j_plus_1 == 0 \
                    and dT_theta_j == 1 and dT_theta_j_plus_1 == 1:
                intersection_types.append(2)
            elif T_theta_j == 0 and T_theta_j_plus_1 == 1 \
                    and dT_theta_j == 0 and dT_theta_j_plus_1 == 0:
                intersection_types.append(2)
            elif T_theta_j == 0 and T_theta_j_plus_1 == 1 \
                    and dT_theta_j == 1 and dT_theta_j_plus_1 == 1:
                intersection_types.append(0)

    return intersections, intersection_types, dtheta_dcnss


def drp_dtheta(_cs, _theta):
    _drp_dtheta = 0.
    for k, n, in enumerate(N_c_s):
        _drp_dtheta += 1j * n * _cs[k] * np.exp(1j * n * _theta)
    return np.real(_drp_dtheta)


def d2rp_dtheta2(_cs, _theta):
    _d2rp_dtheta2 = 0.
    for k, n, in enumerate(N_c_s):
        _d2rp_dtheta2 += (1j * n)**2 * _cs[k] * np.exp(1j * n * _theta)
    return np.real(_d2rp_dtheta2)


def z_p(_cs, _theta, _d, _nu):
    return (1. - _d**2 - r_p(_theta)**2 + 2 * _d * r_p(_theta)
            * np.cos(_theta - _nu))**0.5


def zeta(_z_p, n_val):
    return (1 - _z_p**(n_val + 2)) / ((n_val + 2) * (1 - _z_p**2))


def eta(_theta, _cs, _d, _nu):
    return r_p(_theta)**2 - _d * np.cos(_theta - _nu) * r_p(_theta) \
           - _d * np.sin(_theta - _nu) * drp_dtheta(_cs, _theta)


def line_integral_planet(_theta, _cs, _d, _nu, n_val):
    return zeta(z_p(_cs, _theta, _d, _nu), n_val) * eta(_theta, _cs, _d, _nu)


def F_func(cs):
    # Limb darkening.
    us = np.array([1., u1, u2])
    B = np.array([[1., -1., -1.],
                  [0., 1., 2.],
                  [0., 0., -1.]])
    ps = np.matmul(B, us)
    I_0 = 1. / ((1. - us[1] / 3. - us[2] / 6.) * np.pi)

    # Find planet-stellar limb intersections.
    intersections, intersection_types, _ = find_intersections(cs, d, nu)

    # Iterate pairs of thetas computing line integral.
    alpha = 0.
    for j in range(len(intersection_types)):
        if intersection_types[j] == 0 or intersection_types[j] == 1:
            s0, s0_err = quad(
                line_integral_planet, intersections[j], intersections[j + 1],
                args=(cs, d, nu, 0),
                epsabs=1.e-15, epsrel=1.e-15, limit=500)
            s1, s1_err = quad(
                line_integral_planet, intersections[j], intersections[j + 1],
                args=(cs, d, nu, 1),
                epsabs=1.e-15, epsrel=1.e-15, limit=500)
            s2, s2_err = quad(
                line_integral_planet, intersections[j], intersections[j + 1],
                args=(cs, d, nu, 2),
                epsabs=1.e-15, epsrel=1.e-15, limit=500)
            alpha += I_0 * (s0 * ps[0] + s1 * ps[1] + s2 * ps[2])
        elif intersection_types[j] == 2 or intersection_types[j] == 3:
            phi_j = np.arctan2(
                -r_p(intersections[j]) * np.sin(intersections[j] - nu),
                -r_p(intersections[j]) * np.cos(
                    intersections[j] - nu) + d)
            phi_j_plus_1 = np.arctan2(
                -r_p(intersections[j + 1]) * np.sin(
                    intersections[j + 1] - nu),
                -r_p(intersections[j + 1]) * np.cos(
                    intersections[j + 1] - nu) + d)
            s0 = 1. / (0. + 2) * (phi_j_plus_1 - phi_j)
            s1 = 1. / (1. + 2) * (phi_j_plus_1 - phi_j)
            s2 = 1. / (2. + 2) * (phi_j_plus_1 - phi_j)
            alpha += I_0 * (s0 * ps[0] + s1 * ps[1] + s2 * ps[2])
        else:
            pass

    return 1. - alpha


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


def ds0_dq0_dq0_dcs(cs):
    intersections, intersection_types, _ = find_intersections(cs, d, nu)

    N_q_0 = int(2. / 4. * (max(4 * N_c + 1, 2 * N_c + 3) - 1))
    beta_cos = np.cos(nu) * np.array([0.5, 0, 0.5]) \
               + np.sin(nu) * np.array([0.5j, 0, -0.5j])
    beta_sin = np.cos(nu) * np.array([0.5j, 0, -0.5j]) \
               - np.sin(nu) * np.array([0.5, 0, 0.5])

    _ds0_dq0_dq0_dcs = []
    for n in N_c_s:
        el_vector = np.zeros(N_c * 2 + 1)
        el_vector[n + N_c] = 1

        dq0_dcn = 1. / 2 * add_arrays_centre_aligned(
            2. * np.convolve(el_vector, cs),
            np.convolve(-d * el_vector, beta_cos)
            + np.convolve(-d * 1j * n * el_vector, beta_sin))

        _ds0_dq0_dq0_dcn = 0.
        for j in range(len(intersection_types)):
            if intersection_types[j] == 0 or intersection_types[j] == 1:
                N_q_s = np.arange(-N_q_0, N_q_0 + 1, 1)
                for m in N_q_s:
                    if m == 0:
                        _ds0_dq0_dq0_dcn += (intersections[j + 1] - intersections[j]) \
                                           * dq0_dcn[m + N_q_0]
                    else:
                        _ds0_dq0_dq0_dcn += 1. / (1j * m) * (
                                np.exp(1j * m * intersections[j + 1])
                                - np.exp(1j * m * intersections[j])) * dq0_dcn[m + N_q_0]
            elif intersection_types[j] == 2 or intersection_types[j] == 3:
                pass
            else:
                pass

        _ds0_dq0_dq0_dcs.append(_ds0_dq0_dq0_dcn)

    return np.array(_ds0_dq0_dq0_dcs)


def ds2_dq2_dq2_dcs(cs):
    intersections, intersection_types, _ = find_intersections(cs, d, nu)

    N_q_2 = int(4 / 4 * (max(4 * N_c + 1, 2 * N_c + 3) - 1))
    beta_cos = np.cos(nu) * np.array([0.5, 0, 0.5]) \
               + np.sin(nu) * np.array([0.5j, 0, -0.5j])
    beta_sin = np.cos(nu) * np.array([0.5j, 0, -0.5j]) \
               - np.sin(nu) * np.array([0.5, 0, 0.5])

    _ds0_dq0_dq0_dcs = []
    for n in N_c_s:
        el_vector = np.zeros(N_c * 2 + 1)
        el_vector[n + N_c] = 1
        Delta_cs = 1j * N_c_s * cs

        q2_lhs = add_arrays_centre_aligned(
            add_arrays_centre_aligned(
                np.array([2. - d ** 2]), -np.convolve(cs, cs)),
            2. * d * np.convolve(beta_cos, cs))
        dq2_dcn_lhs = add_arrays_centre_aligned(
            -2. * np.convolve(el_vector, cs),
            np.convolve(2 * d * el_vector, beta_cos))
        q2_rhs = add_arrays_centre_aligned(
            np.convolve(cs, cs),
            d * np.convolve(-beta_cos, cs) + d * np.convolve(-beta_sin, Delta_cs))
        dq2_dcn_rhs = add_arrays_centre_aligned(
            2. * np.convolve(el_vector, cs),
            np.convolve(-d * el_vector, beta_cos)
            + np.convolve(-d * 1j * n * el_vector, beta_sin))

        dq2_dcn = 1. / 4. * add_arrays_centre_aligned(
            np.convolve(dq2_dcn_lhs, q2_rhs), np.convolve(q2_lhs, dq2_dcn_rhs))

        _ds2_dq2_dq2_dcn = 0.
        for j in range(len(intersection_types)):
            if intersection_types[j] == 0 or intersection_types[j] == 1:
                N_q_s = np.arange(-N_q_2, N_q_2 + 1, 1)
                for m in N_q_s:
                    if m == 0:
                        _ds2_dq2_dq2_dcn += (intersections[j + 1] - intersections[j]) \
                                           * dq2_dcn[m + N_q_2]
                    else:
                        _ds2_dq2_dq2_dcn += 1. / (1j * m) * (
                                np.exp(1j * m * intersections[j + 1])
                                - np.exp(1j * m * intersections[j])) * dq2_dcn[m + N_q_2]
            elif intersection_types[j] == 2 or intersection_types[j] == 3:
                pass
            else:
                pass

        _ds0_dq0_dq0_dcs.append(_ds2_dq2_dq2_dcn)

    return np.array(_ds0_dq0_dq0_dcs)


def ds0_dtheta_j_dtheta_j_dcs(cs):
    intersections, intersection_types, dtheta_dcss = find_intersections(cs, d, nu)

    N_q_0 = int(2 / 4 * (max(4 * N_c + 1, 2 * N_c + 3) - 1))
    beta_cos = np.cos(nu) * np.array([0.5, 0, 0.5]) \
               + np.sin(nu) * np.array([0.5j, 0, -0.5j])
    beta_sin = np.cos(nu) * np.array([0.5j, 0, -0.5j]) \
               - np.sin(nu) * np.array([0.5, 0, 0.5])

    Delta_cs = 1j * N_c_s * cs
    q0 = 1. / 2. * add_arrays_centre_aligned(
        np.convolve(cs, cs),
        np.convolve(-d * beta_cos, cs) + np.convolve(-d * beta_sin, Delta_cs))

    _ds0_dtheta_j_dtheta_j_dcs = []
    for n in N_c_s:

        _ds0_dtheta_j_dtheta_j_dcn = 0.
        for j in range(len(intersection_types)):
            if intersection_types[j] == 0:
                N_q_s = np.arange(-N_q_0, N_q_0 + 1, 1)
                _ds0_dtheta_j = 0.
                for m in N_q_s:
                    _ds0_dtheta_j += -q0[m + N_q_0] * np.exp(1j * m * intersections[j])
                _ds0_dtheta_j_dtheta_j_dcn += _ds0_dtheta_j * dtheta_dcss[n + N_c][j]
            elif intersection_types[j] == 2:
                pass
            elif intersection_types[j] == 3:
                pass
            elif intersection_types[j] == 4:
                pass
            else:
                pass

        _ds0_dtheta_j_dtheta_j_dcs.append(_ds0_dtheta_j_dtheta_j_dcn)

    return np.array(_ds0_dtheta_j_dtheta_j_dcs)


def ds2_dtheta_j_dtheta_j_dcs(cs):
    intersections, intersection_types, dtheta_dcss = find_intersections(cs, d, nu)

    N_q_2 = int(4 / 4 * (max(4 * N_c + 1, 2 * N_c + 3) - 1))
    beta_cos = np.cos(nu) * np.array([0.5, 0, 0.5]) \
               + np.sin(nu) * np.array([0.5j, 0, -0.5j])
    beta_sin = np.cos(nu) * np.array([0.5j, 0, -0.5j]) \
               - np.sin(nu) * np.array([0.5, 0, 0.5])

    Delta_cs = 1j * N_c_s * cs

    q2_lhs = add_arrays_centre_aligned(
        add_arrays_centre_aligned(
            np.array([2. - d ** 2]), -np.convolve(cs, cs)),
        2. * d * np.convolve(beta_cos, cs))
    q2_rhs = add_arrays_centre_aligned(
        np.convolve(cs, cs),
        d * np.convolve(-beta_cos, cs) + d * np.convolve(-beta_sin, Delta_cs))

    q2 = 1. / 4. * np.convolve(q2_lhs, q2_rhs)

    _ds2_dtheta_j_dtheta_j_dcs = []
    for n in N_c_s:

        _ds2_dtheta_j_dtheta_j_dcn = 0.
        for j in range(len(intersection_types)):
            if intersection_types[j] == 0:
                N_q_s = np.arange(-N_q_2, N_q_2 + 1, 1)
                _ds2_dtheta_j = 0.
                for m in N_q_s:
                    _ds2_dtheta_j += -q2[m + N_q_2] * np.exp(1j * m * intersections[j])
                _ds2_dtheta_j_dtheta_j_dcn += _ds2_dtheta_j * dtheta_dcss[n + N_c][j]
            elif intersection_types[j] == 2:
                pass
            elif intersection_types[j] == 3:
                pass
            elif intersection_types[j] == 4:
                pass
            else:
                pass

        _ds2_dtheta_j_dtheta_j_dcs.append(_ds2_dtheta_j_dtheta_j_dcn)

    return np.array(_ds2_dtheta_j_dtheta_j_dcs)


def ds1_dtheta_j_dtheta_j_dcs(cs):
    intersections, intersection_types, dtheta_dcss = find_intersections(cs, d, nu)

    _ds1_dtheta_j_dtheta_j_dcs = []
    for n in N_c_s:

        _ds1_dtheta_j_dtheta_j_dcn = 0.
        for j in range(len(intersection_types)):
            if intersection_types[j] == 0:
                half_theta_range = (intersections[j + 1] - intersections[j]) / 2.
                roots, weights = roots_legendre(N_l)
                t = half_theta_range * (roots + 1.) + intersections[j]
                _ds1_dtheta_j = 0.
                for k in range(N_l):
                    _ds1_dtheta_j += -0.5 * zeta(z_p(cs, t[k], d, nu), 1) \
                                     * eta(t[k], cs, d, nu) * weights[k]

                _ds1_dtheta_j_dtheta_j_dcn += _ds1_dtheta_j * dtheta_dcss[n + N_c][j]
            elif intersection_types[j] == 2:
                pass
            elif intersection_types[j] == 3:
                pass
            elif intersection_types[j] == 4:
                pass
            else:
                pass

        _ds1_dtheta_j_dtheta_j_dcs.append(_ds1_dtheta_j_dtheta_j_dcn)

    return np.array(_ds1_dtheta_j_dtheta_j_dcs)


def ds0_dtheta_j_plus_1_dtheta_j_plus_1_dcs(cs):
    intersections, intersection_types, dtheta_dcss = find_intersections(cs, d, nu)

    N_q_0 = int(2 / 4 * (max(4 * N_c + 1, 2 * N_c + 3) - 1))
    beta_cos = np.cos(nu) * np.array([0.5, 0, 0.5]) \
               + np.sin(nu) * np.array([0.5j, 0, -0.5j])
    beta_sin = np.cos(nu) * np.array([0.5j, 0, -0.5j]) \
               - np.sin(nu) * np.array([0.5, 0, 0.5])

    Delta_cs = 1j * N_c_s * cs
    q0 = 1. / 2. * add_arrays_centre_aligned(
        np.convolve(cs, cs),
        np.convolve(-d * beta_cos, cs) + np.convolve(-d * beta_sin, Delta_cs))

    _ds0_dtheta_j_dtheta_j_plus_1_dcs = []
    for n in N_c_s:

        _ds0_dtheta_j_dtheta_j_plus_1_dcn = 0.
        for j in range(len(intersection_types)):
            if intersection_types[j] == 0:
                N_q_s = np.arange(-N_q_0, N_q_0 + 1, 1)
                _ds0_dtheta_j_plus_1 = 0.
                for m in N_q_s:
                    _ds0_dtheta_j_plus_1 += q0[m + N_q_0] * np.exp(1j * m * intersections[j + 1])
                _ds0_dtheta_j_dtheta_j_plus_1_dcn += _ds0_dtheta_j_plus_1 * dtheta_dcss[n + N_c][j + 1]
            elif intersection_types[j] == 1:
                pass
            elif intersection_types[j] == 2:
                pass
            elif intersection_types[j] == 3:
                pass
            else:
                pass

        _ds0_dtheta_j_dtheta_j_plus_1_dcs.append(_ds0_dtheta_j_dtheta_j_plus_1_dcn)

    return np.array(_ds0_dtheta_j_dtheta_j_plus_1_dcs)


def ds2_dtheta_j_plus_1_dtheta_j_plus_1_dcs(cs):
    intersections, intersection_types, dtheta_dcss = find_intersections(cs, d, nu)

    N_q_2 = int(4 / 4 * (max(4 * N_c + 1, 2 * N_c + 3) - 1))
    beta_cos = np.cos(nu) * np.array([0.5, 0, 0.5]) \
               + np.sin(nu) * np.array([0.5j, 0, -0.5j])
    beta_sin = np.cos(nu) * np.array([0.5j, 0, -0.5j]) \
               - np.sin(nu) * np.array([0.5, 0, 0.5])

    Delta_cs = 1j * N_c_s * cs

    q2_lhs = add_arrays_centre_aligned(
        add_arrays_centre_aligned(
            np.array([2. - d ** 2]), -np.convolve(cs, cs)),
        2. * d * np.convolve(beta_cos, cs))
    q2_rhs = add_arrays_centre_aligned(
        np.convolve(cs, cs),
        d * np.convolve(-beta_cos, cs) + d * np.convolve(-beta_sin, Delta_cs))

    q2 = 1. / 4. * np.convolve(q2_lhs, q2_rhs)

    _ds2_dtheta_j_dtheta_j_plus_1_dcs = []
    for n in N_c_s:

        _ds2_dtheta_j_dtheta_j_plus_1_dcn = 0.
        for j in range(len(intersection_types)):
            if intersection_types[j] == 0:
                N_q_s = np.arange(-N_q_2, N_q_2 + 1, 1)
                _ds2_dtheta_j_plus_1 = 0.
                for m in N_q_s:
                    _ds2_dtheta_j_plus_1 += q2[m + N_q_2] * np.exp(1j * m * intersections[j + 1])
                _ds2_dtheta_j_dtheta_j_plus_1_dcn += _ds2_dtheta_j_plus_1 * dtheta_dcss[n + N_c][j + 1]
            elif intersection_types[j] == 1:
                pass
            elif intersection_types[j] == 2:
                pass
            elif intersection_types[j] == 3:
                pass
            else:
                pass

        _ds2_dtheta_j_dtheta_j_plus_1_dcs.append(_ds2_dtheta_j_dtheta_j_plus_1_dcn)

    return np.array(_ds2_dtheta_j_dtheta_j_plus_1_dcs)


def ds1_dtheta_j_plus_1_dtheta_j_plus_1_dcs(cs):
    intersections, intersection_types, dtheta_dcss = find_intersections(cs, d, nu)

    _ds1_dtheta_j_plus_1_dtheta_j_plus_1_dcs = []
    for n in N_c_s:

        _ds1_dtheta_j_plus_1_dtheta_j_plus_1_dcn = 0.
        for j in range(len(intersection_types)):
            if intersection_types[j] == 0:
                half_theta_range = (intersections[j + 1] - intersections[j]) / 2.
                roots, weights = roots_legendre(N_l)
                t = half_theta_range * (roots + 1.) + intersections[j]
                _ds1_dtheta_j_plus_1 = 0.
                for k in range(N_l):
                    _ds1_dtheta_j_plus_1 += 0.5 * zeta(z_p(cs, t[k], d, nu), 1) \
                                            * eta(t[k], cs, d, nu) * weights[k]

                _ds1_dtheta_j_plus_1_dtheta_j_plus_1_dcn += _ds1_dtheta_j_plus_1 \
                                                           * dtheta_dcss[n + N_c][j + 1]
            elif intersection_types[j] == 2:
                pass
            elif intersection_types[j] == 3:
                pass
            elif intersection_types[j] == 4:
                pass
            else:
                pass

        _ds1_dtheta_j_plus_1_dtheta_j_plus_1_dcs.append(_ds1_dtheta_j_plus_1_dtheta_j_plus_1_dcn)

    return np.array(_ds1_dtheta_j_plus_1_dtheta_j_plus_1_dcs)


def ds0_dphi_j_dphi_j_drp_drp_dcs(cs):
    intersections, intersection_types, _ = find_intersections(cs, d, nu)

    _ds0_dphi_j_dphi_j_drp_drp_dcs = []
    for n in N_c_s:

        _ds0_dphi_j_dphi_j_drp_drp_dcn = 0.
        for j in range(len(intersection_types)):
            if intersection_types[j] == 0:
                pass
            elif intersection_types[j] == 1:
                pass
            elif intersection_types[j] == 2:
                _rp_j = r_p(intersections[j])
                _ds0_dphi_j = -1. / 2.
                _dphi_j_drp = (- d * np.sin(intersections[j] - nu)) \
                             / (-2 * d * _rp_j * np.cos(intersections[j] - nu)
                                + d**2 + _rp_j**2)
                _drp_dcn = np.exp(1j * n * intersections[j])
                _ds0_dphi_j_dphi_j_drp_drp_dcn += \
                    _ds0_dphi_j * _dphi_j_drp * _drp_dcn
            elif intersection_types[j] == 3:
                pass
            else:
                pass

        _ds0_dphi_j_dphi_j_drp_drp_dcs.append(_ds0_dphi_j_dphi_j_drp_drp_dcn)

    return np.array(_ds0_dphi_j_dphi_j_drp_drp_dcs)


def ds2_dphi_j_dphi_j_drp_drp_dcs(cs):
    intersections, intersection_types, _ = find_intersections(cs, d, nu)

    _ds2_dphi_j_dphi_j_drp_drp_dcs = []
    for n in N_c_s:

        _ds2_dphi_j_dphi_j_drp_drp_dcn = 0.
        for j in range(len(intersection_types)):
            if intersection_types[j] == 0:
                pass
            elif intersection_types[j] == 1:
                pass
            elif intersection_types[j] == 2:
                _rp_j = r_p(intersections[j])
                _ds2_dphi_j = -1. / 4.
                _dphi_j_drp = (- d * np.sin(intersections[j] - nu)) \
                             / (-2 * d * _rp_j * np.cos(intersections[j] - nu)
                                + d**2 + _rp_j**2)
                _drp_dcn = np.exp(1j * n * intersections[j])
                _ds2_dphi_j_dphi_j_drp_drp_dcn += \
                    _ds2_dphi_j * _dphi_j_drp * _drp_dcn
            elif intersection_types[j] == 3:
                pass
            else:
                pass

        _ds2_dphi_j_dphi_j_drp_drp_dcs.append(_ds2_dphi_j_dphi_j_drp_drp_dcn)

    return np.array(_ds2_dphi_j_dphi_j_drp_drp_dcs)


def ds1_dphi_j_dphi_j_drp_drp_dcs(cs):
    intersections, intersection_types, _ = find_intersections(cs, d, nu)

    _ds1_dphi_j_dphi_j_drp_drp_dcs = []
    for n in N_c_s:

        _ds1_dphi_j_dphi_j_drp_drp_dcn = 0.
        for j in range(len(intersection_types)):
            if intersection_types[j] == 0:
                pass
            elif intersection_types[j] == 1:
                pass
            elif intersection_types[j] == 2:
                _rp_j = r_p(intersections[j])
                _ds1_dphi_j = -1. / 3.
                _dphi_j_drp = (- d * np.sin(intersections[j] - nu)) \
                             / (-2 * d * _rp_j * np.cos(intersections[j] - nu)
                                + d**2 + _rp_j**2)
                _drp_dcn = np.exp(1j * n * intersections[j])
                _ds1_dphi_j_dphi_j_drp_drp_dcn += \
                    _ds1_dphi_j * _dphi_j_drp * _drp_dcn
            elif intersection_types[j] == 3:
                pass
            else:
                pass

        _ds1_dphi_j_dphi_j_drp_drp_dcs.append(_ds1_dphi_j_dphi_j_drp_drp_dcn)

    return np.array(_ds1_dphi_j_dphi_j_drp_drp_dcs)


def ds0_dphi_j_plus_1_dphi_j_plus_1_drp_drp_dcs(cs):
    intersections, intersection_types, _ = find_intersections(cs, d, nu)

    _ds0_dphi_j_plus_1_dphi_j_plus_1_drp_drp_dcs = []
    for n in N_c_s:

        _ds0_dphi_j_plus_1_dphi_j_plus_1_drp_drp_dcn = 0.
        for j in range(len(intersection_types)):
            if intersection_types[j] == 0:
                pass
            elif intersection_types[j] == 1:
                pass
            elif intersection_types[j] == 2:
                _rp_j_plus_1 = r_p(intersections[j + 1])
                _ds0_dphi_j_plus_1 = 1. / 2.
                _dphi_j_plus_1_drp = (- d * np.sin(intersections[j + 1] - nu)) \
                             / (-2 * d * _rp_j_plus_1 * np.cos(intersections[j + 1] - nu)
                                + d**2 + _rp_j_plus_1**2)
                _drp_dcn = np.exp(1j * n * intersections[j + 1])
                _ds0_dphi_j_plus_1_dphi_j_plus_1_drp_drp_dcn += \
                    _ds0_dphi_j_plus_1 * _dphi_j_plus_1_drp * _drp_dcn
            elif intersection_types[j] == 3:
                pass
            else:
                pass

        _ds0_dphi_j_plus_1_dphi_j_plus_1_drp_drp_dcs.append(
            _ds0_dphi_j_plus_1_dphi_j_plus_1_drp_drp_dcn)

    return np.array(_ds0_dphi_j_plus_1_dphi_j_plus_1_drp_drp_dcs)


def ds2_dphi_j_plus_1_dphi_j_plus_1_drp_drp_dcs(cs):
    intersections, intersection_types, _ = find_intersections(cs, d, nu)

    _ds2_dphi_j_plus_1_dphi_j_plus_1_drp_drp_dcs = []
    for n in N_c_s:

        _ds2_dphi_j_plus_1_dphi_j_plus_1_drp_drp_dcn = 0.
        for j in range(len(intersection_types)):
            if intersection_types[j] == 0:
                pass
            elif intersection_types[j] == 1:
                pass
            elif intersection_types[j] == 2:
                _rp_j_plus_1 = r_p(intersections[j + 1])
                _ds2_dphi_j_plus_1 = 1. / 4.
                _dphi_j_plus_1_drp = (- d * np.sin(intersections[j + 1] - nu)) \
                             / (-2 * d * _rp_j_plus_1 * np.cos(intersections[j + 1] - nu)
                                + d**2 + _rp_j_plus_1**2)
                _drp_dcn = np.exp(1j * n * intersections[j + 1])
                _ds2_dphi_j_plus_1_dphi_j_plus_1_drp_drp_dcn += \
                    _ds2_dphi_j_plus_1 * _dphi_j_plus_1_drp * _drp_dcn
            elif intersection_types[j] == 3:
                pass
            else:
                pass

        _ds2_dphi_j_plus_1_dphi_j_plus_1_drp_drp_dcs.append(
            _ds2_dphi_j_plus_1_dphi_j_plus_1_drp_drp_dcn)

    return np.array(_ds2_dphi_j_plus_1_dphi_j_plus_1_drp_drp_dcs)


def ds1_dphi_j_plus_1_dphi_j_plus_1_drp_drp_dcs(cs):
    intersections, intersection_types, _ = find_intersections(cs, d, nu)

    _ds1_dphi_j_plus_1_dphi_j_plus_1_drp_drp_dcs = []
    for n in N_c_s:

        _ds1_dphi_j_plus_1_dphi_j_plus_1_drp_drp_dcn = 0.
        for j in range(len(intersection_types)):
            if intersection_types[j] == 0:
                pass
            elif intersection_types[j] == 1:
                pass
            elif intersection_types[j] == 2:
                _rp_j_plus_1 = r_p(intersections[j + 1])
                _ds1_dphi_j_plus_1 = 1. / 3.
                _dphi_j_plus_1_drp = (- d * np.sin(intersections[j + 1] - nu)) \
                             / (-2 * d * _rp_j_plus_1 * np.cos(intersections[j + 1] - nu)
                                + d**2 + _rp_j_plus_1**2)
                _drp_dcn = np.exp(1j * n * intersections[j + 1])
                _ds1_dphi_j_plus_1_dphi_j_plus_1_drp_drp_dcn += \
                    _ds1_dphi_j_plus_1 * _dphi_j_plus_1_drp * _drp_dcn
            elif intersection_types[j] == 3:
                pass
            else:
                pass

        _ds1_dphi_j_plus_1_dphi_j_plus_1_drp_drp_dcs.append(
            _ds1_dphi_j_plus_1_dphi_j_plus_1_drp_drp_dcn)

    return np.array(_ds1_dphi_j_plus_1_dphi_j_plus_1_drp_drp_dcs)


def ds0_dphi_j_dphi_j_dtheta_j_dtheta_j_dcs(cs):
    intersections, intersection_types, dtheta_dcss = find_intersections(cs, d, nu)

    _ds0_dphi_j_dphi_j_dtheta_j_dtheta_j_dcs = []
    for n in N_c_s:

        _ds0_dphi_j_dphi_j_dtheta_j_dtheta_j_dcn = 0.
        for j in range(len(intersection_types)):
            if intersection_types[j] == 0:
                pass
            elif intersection_types[j] == 1:
                pass
            elif intersection_types[j] == 2:
                _rp_j = r_p(intersections[j])
                _ds0_dphi_j = -1. / 2.
                _dphi_j_dtheta_j = (_rp_j**2 - d * _rp_j * np.cos(intersections[j] - nu)) \
                             / (-2 * d * _rp_j * np.cos(intersections[j] - nu)
                                + d**2 + _rp_j**2)
                _dtheta_j_dcn = dtheta_dcss[n + N_c][j]
                _ds0_dphi_j_dphi_j_dtheta_j_dtheta_j_dcn += \
                    _ds0_dphi_j * _dphi_j_dtheta_j * _dtheta_j_dcn
            elif intersection_types[j] == 3:
                pass
            else:
                pass

        _ds0_dphi_j_dphi_j_dtheta_j_dtheta_j_dcs.append(
            _ds0_dphi_j_dphi_j_dtheta_j_dtheta_j_dcn)

    return np.array(_ds0_dphi_j_dphi_j_dtheta_j_dtheta_j_dcs)


def ds2_dphi_j_dphi_j_dtheta_j_dtheta_j_dcs(cs):
    intersections, intersection_types, dtheta_dcss = find_intersections(cs, d, nu)

    _ds2_dphi_j_dphi_j_dtheta_j_dtheta_j_dcs = []
    for n in N_c_s:

        _ds2_dphi_j_dphi_j_dtheta_j_dtheta_j_dcn = 0.
        for j in range(len(intersection_types)):
            if intersection_types[j] == 0:
                pass
            elif intersection_types[j] == 1:
                pass
            elif intersection_types[j] == 2:
                _rp_j = r_p(intersections[j])
                _ds2_dphi_j = -1. / 4.
                _dphi_j_dtheta_j = (_rp_j**2 - d * _rp_j * np.cos(intersections[j] - nu)) \
                             / (-2 * d * _rp_j * np.cos(intersections[j] - nu)
                                + d**2 + _rp_j**2)
                _dtheta_j_dcn = dtheta_dcss[n + N_c][j]
                _ds2_dphi_j_dphi_j_dtheta_j_dtheta_j_dcn += \
                    _ds2_dphi_j * _dphi_j_dtheta_j * _dtheta_j_dcn
            elif intersection_types[j] == 3:
                pass
            else:
                pass

        _ds2_dphi_j_dphi_j_dtheta_j_dtheta_j_dcs.append(
            _ds2_dphi_j_dphi_j_dtheta_j_dtheta_j_dcn)

    return np.array(_ds2_dphi_j_dphi_j_dtheta_j_dtheta_j_dcs)


def ds1_dphi_j_dphi_j_dtheta_j_dtheta_j_dcs(cs):
    intersections, intersection_types, dtheta_dcss = find_intersections(cs, d, nu)

    _ds1_dphi_j_dphi_j_dtheta_j_dtheta_j_dcs = []
    for n in N_c_s:

        _ds1_dphi_j_dphi_j_dtheta_j_dtheta_j_dcn = 0.
        for j in range(len(intersection_types)):
            if intersection_types[j] == 0:
                pass
            elif intersection_types[j] == 1:
                pass
            elif intersection_types[j] == 2:
                _rp_j = r_p(intersections[j])
                _ds1_dphi_j = -1. / 3.
                _dphi_j_dtheta_j = (_rp_j**2 - d * _rp_j * np.cos(intersections[j] - nu)) \
                             / (-2 * d * _rp_j * np.cos(intersections[j] - nu)
                                + d**2 + _rp_j**2)
                _dtheta_j_dcn = dtheta_dcss[n + N_c][j]
                _ds1_dphi_j_dphi_j_dtheta_j_dtheta_j_dcn += \
                    _ds1_dphi_j * _dphi_j_dtheta_j * _dtheta_j_dcn
            elif intersection_types[j] == 3:
                pass
            else:
                pass

        _ds1_dphi_j_dphi_j_dtheta_j_dtheta_j_dcs.append(
            _ds1_dphi_j_dphi_j_dtheta_j_dtheta_j_dcn)

    return np.array(_ds1_dphi_j_dphi_j_dtheta_j_dtheta_j_dcs)


def ds0_dphi_j_plus_1_dphi_j_plus_1_dtheta_j_plus_1_dtheta_j_plus_1_dcs(cs):
    intersections, intersection_types, dtheta_dcss = find_intersections(cs, d, nu)

    _ds0_dphi_j_plus_1_dphi_j_plus_1_dtheta_j_plus_1_dtheta_j_plus_1_dcs = []
    for n in N_c_s:

        _ds0_dphi_j_plus_1_dphi_j_plus_1_dtheta_j_plus_1_dtheta_j_plus_1_dcn = 0.
        for j in range(len(intersection_types)):
            if intersection_types[j] == 0:
                pass
            elif intersection_types[j] == 1:
                pass
            elif intersection_types[j] == 2:
                _rp_j_plus_1 = r_p(intersections[j + 1])
                _ds0_dphi_j_plus_1 = 1. / 2.
                _dphi_j_plus_1_dtheta_j_plus_1 = \
                    (_rp_j_plus_1**2 - d * _rp_j_plus_1 * np.cos(intersections[j + 1] - nu)) \
                    / (-2 * d * _rp_j_plus_1 * np.cos(intersections[j + 1] - nu)
                       + d**2 + _rp_j_plus_1**2)
                _dtheta_j_plus_1_dcn = dtheta_dcss[n + N_c][j + 1]
                _ds0_dphi_j_plus_1_dphi_j_plus_1_dtheta_j_plus_1_dtheta_j_plus_1_dcn += \
                    _ds0_dphi_j_plus_1 * _dphi_j_plus_1_dtheta_j_plus_1 * _dtheta_j_plus_1_dcn
            elif intersection_types[j] == 3:
                pass
            else:
                pass

        _ds0_dphi_j_plus_1_dphi_j_plus_1_dtheta_j_plus_1_dtheta_j_plus_1_dcs.append(
            _ds0_dphi_j_plus_1_dphi_j_plus_1_dtheta_j_plus_1_dtheta_j_plus_1_dcn)

    return np.array(_ds0_dphi_j_plus_1_dphi_j_plus_1_dtheta_j_plus_1_dtheta_j_plus_1_dcs)


def ds2_dphi_j_plus_1_dphi_j_plus_1_dtheta_j_plus_1_dtheta_j_plus_1_dcs(cs):
    intersections, intersection_types, dtheta_dcss = find_intersections(cs, d, nu)

    _ds2_dphi_j_plus_1_dphi_j_plus_1_dtheta_j_plus_1_dtheta_j_plus_1_dcs = []
    for n in N_c_s:

        _ds2_dphi_j_plus_1_dphi_j_plus_1_dtheta_j_plus_1_dtheta_j_plus_1_dcn = 0.
        for j in range(len(intersection_types)):
            if intersection_types[j] == 0:
                pass
            elif intersection_types[j] == 1:
                pass
            elif intersection_types[j] == 2:
                _rp_j_plus_1 = r_p(intersections[j + 1])
                _ds2_dphi_j_plus_1 = 1. / 4.
                _dphi_j_plus_1_dtheta_j_plus_1 = \
                    (_rp_j_plus_1**2 - d * _rp_j_plus_1 * np.cos(intersections[j + 1] - nu)) \
                    / (-2 * d * _rp_j_plus_1 * np.cos(intersections[j + 1] - nu)
                       + d**2 + _rp_j_plus_1**2)
                _dtheta_j_plus_1_dcn = dtheta_dcss[n + N_c][j + 1]
                _ds2_dphi_j_plus_1_dphi_j_plus_1_dtheta_j_plus_1_dtheta_j_plus_1_dcn += \
                    _ds2_dphi_j_plus_1 * _dphi_j_plus_1_dtheta_j_plus_1 * _dtheta_j_plus_1_dcn
            elif intersection_types[j] == 3:
                pass
            else:
                pass

        _ds2_dphi_j_plus_1_dphi_j_plus_1_dtheta_j_plus_1_dtheta_j_plus_1_dcs.append(
            _ds2_dphi_j_plus_1_dphi_j_plus_1_dtheta_j_plus_1_dtheta_j_plus_1_dcn)

    return np.array(_ds2_dphi_j_plus_1_dphi_j_plus_1_dtheta_j_plus_1_dtheta_j_plus_1_dcs)


def ds1_dphi_j_plus_1_dphi_j_plus_1_dtheta_j_plus_1_dtheta_j_plus_1_dcs(cs):
    intersections, intersection_types, dtheta_dcss = find_intersections(cs, d, nu)

    _ds1_dphi_j_plus_1_dphi_j_plus_1_dtheta_j_plus_1_dtheta_j_plus_1_dcs = []
    for n in N_c_s:

        _ds1_dphi_j_plus_1_dphi_j_plus_1_dtheta_j_plus_1_dtheta_j_plus_1_dcn = 0.
        for j in range(len(intersection_types)):
            if intersection_types[j] == 0:
                pass
            elif intersection_types[j] == 1:
                pass
            elif intersection_types[j] == 2:
                _rp_j_plus_1 = r_p(intersections[j + 1])
                _ds1_dphi_j_plus_1 = 1. / 3.
                _dphi_j_plus_1_dtheta_j_plus_1 = \
                    (_rp_j_plus_1**2 - d * _rp_j_plus_1 * np.cos(intersections[j + 1] - nu)) \
                    / (-2 * d * _rp_j_plus_1 * np.cos(intersections[j + 1] - nu)
                       + d**2 + _rp_j_plus_1**2)
                _dtheta_j_plus_1_dcn = dtheta_dcss[n + N_c][j + 1]
                _ds1_dphi_j_plus_1_dphi_j_plus_1_dtheta_j_plus_1_dtheta_j_plus_1_dcn += \
                    _ds1_dphi_j_plus_1 * _dphi_j_plus_1_dtheta_j_plus_1 * _dtheta_j_plus_1_dcn
            elif intersection_types[j] == 3:
                pass
            else:
                pass

        _ds1_dphi_j_plus_1_dphi_j_plus_1_dtheta_j_plus_1_dtheta_j_plus_1_dcs.append(
            _ds1_dphi_j_plus_1_dphi_j_plus_1_dtheta_j_plus_1_dtheta_j_plus_1_dcn)

    return np.array(_ds1_dphi_j_plus_1_dphi_j_plus_1_dtheta_j_plus_1_dtheta_j_plus_1_dcs)


def ds0_dphi_j_dphi_j_drp_drp_dtheta_j_dtheta_j_dcs(cs):
    intersections, intersection_types, dtheta_dcss = find_intersections(cs, d, nu)

    _ds0_dphi_j_dphi_j_drp_drp_dtheta_j_dtheta_j_dcs = []
    for n in N_c_s:

        _ds0_dphi_j_dphi_j_drp_drp_dtheta_j_dtheta_j_dcn = 0.
        for j in range(len(intersection_types)):
            if intersection_types[j] == 0:
                pass
            elif intersection_types[j] == 1:
                pass
            elif intersection_types[j] == 2:
                _rp_j = r_p(intersections[j])
                _ds0_dphi_j = -1. / 2.
                _dphi_j_drp = (- d * np.sin(intersections[j] - nu)) \
                             / (-2 * d * _rp_j * np.cos(intersections[j] - nu)
                                + d**2 + _rp_j**2)
                _drp_dtheta_j = drp_dtheta(cs, intersections[j])
                _dtheta_j_dcn = dtheta_dcss[n + N_c][j]
                _ds0_dphi_j_dphi_j_drp_drp_dtheta_j_dtheta_j_dcn += \
                    _ds0_dphi_j * _dphi_j_drp * _drp_dtheta_j * _dtheta_j_dcn
            elif intersection_types[j] == 3:
                pass
            else:
                pass

        _ds0_dphi_j_dphi_j_drp_drp_dtheta_j_dtheta_j_dcs.append(
            _ds0_dphi_j_dphi_j_drp_drp_dtheta_j_dtheta_j_dcn)

    return np.array(_ds0_dphi_j_dphi_j_drp_drp_dtheta_j_dtheta_j_dcs)


def ds2_dphi_j_dphi_j_drp_drp_dtheta_j_dtheta_j_dcs(cs):
    intersections, intersection_types, dtheta_dcss = find_intersections(cs, d, nu)

    _ds2_dphi_j_dphi_j_drp_drp_dtheta_j_dtheta_j_dcs = []
    for n in N_c_s:

        _ds2_dphi_j_dphi_j_drp_drp_dtheta_j_dtheta_j_dcn = 0.
        for j in range(len(intersection_types)):
            if intersection_types[j] == 0:
                pass
            elif intersection_types[j] == 1:
                pass
            elif intersection_types[j] == 2:
                _rp_j = r_p(intersections[j])
                _ds2_dphi_j = -1. / 4.
                _dphi_j_drp = (- d * np.sin(intersections[j] - nu)) \
                             / (-2 * d * _rp_j * np.cos(intersections[j] - nu)
                                + d**2 + _rp_j**2)
                _drp_dtheta_j = drp_dtheta(cs, intersections[j])
                _dtheta_j_dcn = dtheta_dcss[n + N_c][j]
                _ds2_dphi_j_dphi_j_drp_drp_dtheta_j_dtheta_j_dcn += \
                    _ds2_dphi_j * _dphi_j_drp * _drp_dtheta_j * _dtheta_j_dcn
            elif intersection_types[j] == 3:
                pass
            else:
                pass

        _ds2_dphi_j_dphi_j_drp_drp_dtheta_j_dtheta_j_dcs.append(
            _ds2_dphi_j_dphi_j_drp_drp_dtheta_j_dtheta_j_dcn)

    return np.array(_ds2_dphi_j_dphi_j_drp_drp_dtheta_j_dtheta_j_dcs)


def ds1_dphi_j_dphi_j_drp_drp_dtheta_j_dtheta_j_dcs(cs):
    intersections, intersection_types, dtheta_dcss = find_intersections(cs, d, nu)

    _ds1_dphi_j_dphi_j_drp_drp_dtheta_j_dtheta_j_dcs = []
    for n in N_c_s:

        _ds1_dphi_j_dphi_j_drp_drp_dtheta_j_dtheta_j_dcn = 0.
        for j in range(len(intersection_types)):
            if intersection_types[j] == 0:
                pass
            elif intersection_types[j] == 1:
                pass
            elif intersection_types[j] == 2:
                _rp_j = r_p(intersections[j])
                _ds1_dphi_j = -1. / 3.
                _dphi_j_drp = (- d * np.sin(intersections[j] - nu)) \
                             / (-2 * d * _rp_j * np.cos(intersections[j] - nu)
                                + d**2 + _rp_j**2)
                _drp_dtheta_j = drp_dtheta(cs, intersections[j])
                _dtheta_j_dcn = dtheta_dcss[n + N_c][j]
                _ds1_dphi_j_dphi_j_drp_drp_dtheta_j_dtheta_j_dcn += \
                    _ds1_dphi_j * _dphi_j_drp * _drp_dtheta_j * _dtheta_j_dcn
            elif intersection_types[j] == 3:
                pass
            else:
                pass

        _ds1_dphi_j_dphi_j_drp_drp_dtheta_j_dtheta_j_dcs.append(
            _ds1_dphi_j_dphi_j_drp_drp_dtheta_j_dtheta_j_dcn)

    return np.array(_ds1_dphi_j_dphi_j_drp_drp_dtheta_j_dtheta_j_dcs)


def ds0_dphi_j_plus_1_dphi_j_plus_1_drp_drp_dtheta_j_plus_1_dtheta_j_plus_1_dcs(cs):
    intersections, intersection_types, dtheta_dcss = find_intersections(cs, d, nu)

    _ds0_dphi_j_plus_1_dphi_j_plus_1_drp_drp_dtheta_j_plus_1_dtheta_j_plus_1_dcs = []
    for n in N_c_s:

        _ds0_dphi_j_plus_1_dphi_j_plus_1_drp_drp_dtheta_j_plus_1_dtheta_j_plus_1_dcn = 0.
        for j in range(len(intersection_types)):
            if intersection_types[j] == 0:
                pass
            elif intersection_types[j] == 1:
                pass
            elif intersection_types[j] == 2:
                _rp_j_plus_1 = r_p(intersections[j + 1])
                _ds0_dphi_j_plus_1 = 1. / 2.
                _dphi_j_plus_1_drp = (- d * np.sin(intersections[j + 1] - nu)) \
                             / (-2 * d * _rp_j_plus_1 * np.cos(intersections[j + 1] - nu)
                                + d**2 + _rp_j_plus_1**2)
                _drp_dtheta_j_plus_1 = drp_dtheta(cs, intersections[j + 1])
                _dtheta_j_plus_1_dcn = dtheta_dcss[n + N_c][j + 1]
                _ds0_dphi_j_plus_1_dphi_j_plus_1_drp_drp_dtheta_j_plus_1_dtheta_j_plus_1_dcn += \
                    _ds0_dphi_j_plus_1 * _dphi_j_plus_1_drp * _drp_dtheta_j_plus_1 \
                    * _dtheta_j_plus_1_dcn
            elif intersection_types[j] == 3:
                pass
            else:
                pass

        _ds0_dphi_j_plus_1_dphi_j_plus_1_drp_drp_dtheta_j_plus_1_dtheta_j_plus_1_dcs.append(
            _ds0_dphi_j_plus_1_dphi_j_plus_1_drp_drp_dtheta_j_plus_1_dtheta_j_plus_1_dcn)

    return np.array(_ds0_dphi_j_plus_1_dphi_j_plus_1_drp_drp_dtheta_j_plus_1_dtheta_j_plus_1_dcs)


def ds2_dphi_j_plus_1_dphi_j_plus_1_drp_drp_dtheta_j_plus_1_dtheta_j_plus_1_dcs(cs):
    intersections, intersection_types, dtheta_dcss = find_intersections(cs, d, nu)

    _ds2_dphi_j_plus_1_dphi_j_plus_1_drp_drp_dtheta_j_plus_1_dtheta_j_plus_1_dcs = []
    for n in N_c_s:

        _ds2_dphi_j_plus_1_dphi_j_plus_1_drp_drp_dtheta_j_plus_1_dtheta_j_plus_1_dcn = 0.
        for j in range(len(intersection_types)):
            if intersection_types[j] == 0:
                pass
            elif intersection_types[j] == 1:
                pass
            elif intersection_types[j] == 2:
                _rp_j_plus_1 = r_p(intersections[j + 1])
                _ds2_dphi_j_plus_1 = 1. / 4.
                _dphi_j_plus_1_drp = (- d * np.sin(intersections[j + 1] - nu)) \
                             / (-2 * d * _rp_j_plus_1 * np.cos(intersections[j + 1] - nu)
                                + d**2 + _rp_j_plus_1**2)
                _drp_dtheta_j_plus_1 = drp_dtheta(cs, intersections[j + 1])
                _dtheta_j_plus_1_dcn = dtheta_dcss[n + N_c][j + 1]
                _ds2_dphi_j_plus_1_dphi_j_plus_1_drp_drp_dtheta_j_plus_1_dtheta_j_plus_1_dcn += \
                    _ds2_dphi_j_plus_1 * _dphi_j_plus_1_drp * _drp_dtheta_j_plus_1 \
                    * _dtheta_j_plus_1_dcn
            elif intersection_types[j] == 3:
                pass
            else:
                pass

        _ds2_dphi_j_plus_1_dphi_j_plus_1_drp_drp_dtheta_j_plus_1_dtheta_j_plus_1_dcs.append(
            _ds2_dphi_j_plus_1_dphi_j_plus_1_drp_drp_dtheta_j_plus_1_dtheta_j_plus_1_dcn)

    return np.array(_ds2_dphi_j_plus_1_dphi_j_plus_1_drp_drp_dtheta_j_plus_1_dtheta_j_plus_1_dcs)


def ds1_dphi_j_plus_1_dphi_j_plus_1_drp_drp_dtheta_j_plus_1_dtheta_j_plus_1_dcs(cs):
    intersections, intersection_types, dtheta_dcss = find_intersections(cs, d, nu)

    _ds1_dphi_j_plus_1_dphi_j_plus_1_drp_drp_dtheta_j_plus_1_dtheta_j_plus_1_dcs = []
    for n in N_c_s:

        _ds1_dphi_j_plus_1_dphi_j_plus_1_drp_drp_dtheta_j_plus_1_dtheta_j_plus_1_dcn = 0.
        for j in range(len(intersection_types)):
            if intersection_types[j] == 0:
                pass
            elif intersection_types[j] == 1:
                pass
            elif intersection_types[j] == 2:
                _rp_j_plus_1 = r_p(intersections[j + 1])
                _ds1_dphi_j_plus_1 = 1. / 3.
                _dphi_j_plus_1_drp = (- d * np.sin(intersections[j + 1] - nu)) \
                             / (-2 * d * _rp_j_plus_1 * np.cos(intersections[j + 1] - nu)
                                + d**2 + _rp_j_plus_1**2)
                _drp_dtheta_j_plus_1 = drp_dtheta(cs, intersections[j + 1])
                _dtheta_j_plus_1_dcn = dtheta_dcss[n + N_c][j + 1]
                _ds1_dphi_j_plus_1_dphi_j_plus_1_drp_drp_dtheta_j_plus_1_dtheta_j_plus_1_dcn += \
                    _ds1_dphi_j_plus_1 * _dphi_j_plus_1_drp * _drp_dtheta_j_plus_1 \
                    * _dtheta_j_plus_1_dcn
            elif intersection_types[j] == 3:
                pass
            else:
                pass

        _ds1_dphi_j_plus_1_dphi_j_plus_1_drp_drp_dtheta_j_plus_1_dtheta_j_plus_1_dcs.append(
            _ds1_dphi_j_plus_1_dphi_j_plus_1_drp_drp_dtheta_j_plus_1_dtheta_j_plus_1_dcn)

    return np.array(_ds1_dphi_j_plus_1_dphi_j_plus_1_drp_drp_dtheta_j_plus_1_dtheta_j_plus_1_dcs)


def ds1_dzeta_dzeta_dz_dz_dr_dr_dt_dt_dtheta_j_dtheta_j_dcs(cs):
    intersections, intersection_types, dtheta_dcss = find_intersections(cs, d, nu)

    _ds1_dzeta_dzeta_dz_dz_dr_dr_dt_dt_dtheta_j_dtheta_j_dcs = []
    for n in N_c_s:

        _ds1_dzeta_dzeta_dz_dz_dr_dr_dt_dt_dtheta_j_dtheta_j_dcn = 0.
        for j in range(len(intersection_types)):
            if intersection_types[j] == 0:
                chain_j = 0.
                half_theta_range = (intersections[j + 1] - intersections[j]) / 2.
                roots, weights = roots_legendre(N_l)
                t = half_theta_range * (roots + 1.) + intersections[j]
                for k in range(N_l):
                    _ds1_dzeta = eta(t[k], cs, d, nu) * weights[k]
                    _dzeta_dz = 1. / 3 - 1. / (3 * (z_p(cs, t[k], d, nu) + 1)**2)
                    _dz_dr = (d * np.cos(t[k] - nu) - r_p(t[k])) / z_p(cs, t[k], d, nu)
                    _dr_dt = drp_dtheta(cs, t[k])
                    _dt_dtheta_j = -(roots[k] + 1.) / 2 + 1

                    chain_j += _ds1_dzeta * _dzeta_dz * _dz_dr * _dr_dt \
                               * _dt_dtheta_j * dtheta_dcss[n + N_c][j]

                chain_j *= half_theta_range
                _ds1_dzeta_dzeta_dz_dz_dr_dr_dt_dt_dtheta_j_dtheta_j_dcn += chain_j

            elif intersection_types[j] == 1:
                pass
            elif intersection_types[j] == 2:
                pass
            elif intersection_types[j] == 3:
                pass
            else:
                pass

        _ds1_dzeta_dzeta_dz_dz_dr_dr_dt_dt_dtheta_j_dtheta_j_dcs.append(
            _ds1_dzeta_dzeta_dz_dz_dr_dr_dt_dt_dtheta_j_dtheta_j_dcn)

    return np.array(_ds1_dzeta_dzeta_dz_dz_dr_dr_dt_dt_dtheta_j_dtheta_j_dcs)


def ds1_dzeta_dzeta_dz_dz_dr_dr_dt_dt_dtheta_j_plus_1_dtheta_j_plus_1_dcs(cs):
    intersections, intersection_types, dtheta_dcss = find_intersections(cs, d, nu)

    _ds1_dzeta_dzeta_dz_dz_dr_dr_dt_dt_dtheta_j_plus_1_dtheta_j_plus_1_dcs = []
    for n in N_c_s:

        _ds1_dzeta_dzeta_dz_dz_dr_dr_dt_dt_dtheta_j_plus_1_dtheta_j_plus_1_dcn = 0.
        for j in range(len(intersection_types)):
            if intersection_types[j] == 0:
                chain_j_plus_1 = 0.
                half_theta_range = (intersections[j + 1] - intersections[j]) / 2.
                roots, weights = roots_legendre(N_l)
                t = half_theta_range * (roots + 1.) + intersections[j]
                for k in range(N_l):
                    _ds1_dzeta = eta(t[k], cs, d, nu) * weights[k]
                    _dzeta_dz = 1. / 3 - 1. / (3 * (z_p(cs, t[k], d, nu) + 1)**2)
                    _dz_dr = (d * np.cos(t[k] - nu) - r_p(t[k])) / z_p(cs, t[k], d, nu)
                    _dr_dt = drp_dtheta(cs, t[k])
                    _dt_dtheta_j_plus_1 = (roots[k] + 1.) / 2

                    chain_j_plus_1 += _ds1_dzeta * _dzeta_dz * _dz_dr * _dr_dt \
                                      * _dt_dtheta_j_plus_1 * dtheta_dcss[n + N_c][j + 1]

                chain_j_plus_1 *= half_theta_range
                _ds1_dzeta_dzeta_dz_dz_dr_dr_dt_dt_dtheta_j_plus_1_dtheta_j_plus_1_dcn += chain_j_plus_1

            elif intersection_types[j] == 1:
                pass
            elif intersection_types[j] == 2:
                pass
            elif intersection_types[j] == 3:
                pass
            else:
                pass

        _ds1_dzeta_dzeta_dz_dz_dr_dr_dt_dt_dtheta_j_plus_1_dtheta_j_plus_1_dcs.append(
            _ds1_dzeta_dzeta_dz_dz_dr_dr_dt_dt_dtheta_j_plus_1_dtheta_j_plus_1_dcn)

    return np.array(_ds1_dzeta_dzeta_dz_dz_dr_dr_dt_dt_dtheta_j_plus_1_dtheta_j_plus_1_dcs)


def ds1_dzeta_dzeta_dz_dz_dt_dt_dtheta_j_dtheta_j_dcs(cs):
    intersections, intersection_types, dtheta_dcss = find_intersections(cs, d, nu)

    _ds1_dzeta_dzeta_dz_dz_dt_dt_dtheta_j_dtheta_j_dcs = []
    for n in N_c_s:

        _ds1_dzeta_dzeta_dz_dz_dt_dt_dtheta_j_dtheta_j_dcn = 0.
        for j in range(len(intersection_types)):
            if intersection_types[j] == 0:
                chain_j = 0.
                half_theta_range = (intersections[j + 1] - intersections[j]) / 2.
                roots, weights = roots_legendre(N_l)
                t = half_theta_range * (roots + 1.) + intersections[j]
                for k in range(N_l):
                    _ds1_dzeta = eta(t[k], cs, d, nu) * weights[k]
                    _dzeta_dz = 1. / 3 - 1. / (3 * (z_p(cs, t[k], d, nu) + 1)**2)
                    _dz_dt = (-d * r_p(t[k]) * np.sin(t[k] - nu)) / z_p(cs, t[k], d, nu)
                    _dt_dtheta_j = -(roots[k] + 1.) / 2 + 1

                    chain_j += _ds1_dzeta * _dzeta_dz * _dz_dt \
                               * _dt_dtheta_j * dtheta_dcss[n + N_c][j]

                chain_j *= half_theta_range
                _ds1_dzeta_dzeta_dz_dz_dt_dt_dtheta_j_dtheta_j_dcn += chain_j

            elif intersection_types[j] == 1:
                pass
            elif intersection_types[j] == 2:
                pass
            elif intersection_types[j] == 3:
                pass
            else:
                pass

        _ds1_dzeta_dzeta_dz_dz_dt_dt_dtheta_j_dtheta_j_dcs.append(
            _ds1_dzeta_dzeta_dz_dz_dt_dt_dtheta_j_dtheta_j_dcn)

    return np.array(_ds1_dzeta_dzeta_dz_dz_dt_dt_dtheta_j_dtheta_j_dcs)


def ds1_dzeta_dzeta_dz_dz_dt_dt_dtheta_j_plus_1_dtheta_j_plus_1_dcs(cs):
    intersections, intersection_types, dtheta_dcss = find_intersections(cs, d, nu)

    _ds1_dzeta_dzeta_dz_dz_dt_dt_dtheta_j_plus_1_dtheta_j_plus_1_dcs = []
    for n in N_c_s:

        _ds1_dzeta_dzeta_dz_dz_dt_dt_dtheta_j_plus_1_dtheta_j_plus_1_dcn = 0.
        for j in range(len(intersection_types)):
            if intersection_types[j] == 0:
                chain_j_plus_1 = 0.
                half_theta_range = (intersections[j + 1] - intersections[j]) / 2.
                roots, weights = roots_legendre(N_l)
                t = half_theta_range * (roots + 1.) + intersections[j]
                for k in range(N_l):
                    _ds1_dzeta = eta(t[k], cs, d, nu) * weights[k]
                    _dzeta_dz = 1. / 3 - 1. / (3 * (z_p(cs, t[k], d, nu) + 1)**2)
                    _dz_dt = (-d * r_p(t[k]) * np.sin(t[k] - nu)) / z_p(cs, t[k], d, nu)
                    _dt_dtheta_j_plus_1 = (roots[k] + 1.) / 2

                    chain_j_plus_1 += _ds1_dzeta * _dzeta_dz * _dz_dt \
                                      * _dt_dtheta_j_plus_1 * dtheta_dcss[n + N_c][j + 1]

                chain_j_plus_1 *= half_theta_range
                _ds1_dzeta_dzeta_dz_dz_dt_dt_dtheta_j_plus_1_dtheta_j_plus_1_dcn += chain_j_plus_1

            elif intersection_types[j] == 1:
                pass
            elif intersection_types[j] == 2:
                pass
            elif intersection_types[j] == 3:
                pass
            else:
                pass

        _ds1_dzeta_dzeta_dz_dz_dt_dt_dtheta_j_plus_1_dtheta_j_plus_1_dcs.append(
            _ds1_dzeta_dzeta_dz_dz_dt_dt_dtheta_j_plus_1_dtheta_j_plus_1_dcn)

    return np.array(_ds1_dzeta_dzeta_dz_dz_dt_dt_dtheta_j_plus_1_dtheta_j_plus_1_dcs)


def ds1_dzeta_dzeta_dz_dz_dr_dr_dcs(cs):
    intersections, intersection_types, _ = find_intersections(cs, d, nu)

    _ds1_dzeta_dzeta_dz_dz_dr_dr_dcs = []
    for n in N_c_s:

        _ds1_dzeta_dzeta_dz_dz_dr_dr_dcn = 0.
        for j in range(len(intersection_types)):
            if intersection_types[j] == 0 or intersection_types[j] == 1:
                chain_j = 0.
                half_theta_range = (intersections[j + 1] - intersections[j]) / 2.
                roots, weights = roots_legendre(N_l)
                t = half_theta_range * (roots + 1.) + intersections[j]
                for k in range(N_l):
                    _ds1_dzeta = eta(t[k], cs, d, nu) * weights[k]
                    _dzeta_dz = 1. / 3 - 1. / (3 * (z_p(cs, t[k], d, nu) + 1)**2)
                    _dz_dr = (d * np.cos(t[k] - nu) - r_p(t[k])) / z_p(cs, t[k], d, nu)
                    _dr_dcn = np.exp(1j * n * t[k])

                    chain_j += _ds1_dzeta * _dzeta_dz * _dz_dr * _dr_dcn

                chain_j *= half_theta_range
                _ds1_dzeta_dzeta_dz_dz_dr_dr_dcn += chain_j

            elif intersection_types[j] == 2 or intersection_types[j] == 3:
                pass
            else:
                pass

        _ds1_dzeta_dzeta_dz_dz_dr_dr_dcs.append(
            _ds1_dzeta_dzeta_dz_dz_dr_dr_dcn)

    return np.array(_ds1_dzeta_dzeta_dz_dz_dr_dr_dcs)


def ds1_deta_deta_dr_dr_dt_dt_dtheta_j_dtheta_j_dcs(cs):
    intersections, intersection_types, dtheta_dcss = find_intersections(cs, d, nu)

    _ds1_deta_deta_dr_dr_dt_dt_dtheta_j_dtheta_j_dcs = []
    for n in N_c_s:

        _ds1_deta_deta_dr_dr_dt_dt_dtheta_j_dtheta_j_dcn = 0.
        for j in range(len(intersection_types)):
            if intersection_types[j] == 0:
                chain_j = 0.
                half_theta_range = (intersections[j + 1] - intersections[j]) / 2.
                roots, weights = roots_legendre(N_l)
                t = half_theta_range * (roots + 1.) + intersections[j]
                for k in range(N_l):
                    _ds1_deta = zeta(z_p(cs, t[k], d, nu), 1) * weights[k]
                    _deta_dr = 2. * r_p(t[k]) - d * np.cos(t[k] - nu)
                    _dr_dt = drp_dtheta(cs, t[k])
                    _dt_dtheta_j = -(roots[k] + 1.) / 2 + 1

                    chain_j += _ds1_deta * _deta_dr * _dr_dt \
                               * _dt_dtheta_j * dtheta_dcss[n + N_c][j]

                chain_j *= half_theta_range
                _ds1_deta_deta_dr_dr_dt_dt_dtheta_j_dtheta_j_dcn += chain_j

            elif intersection_types[j] == 1:
                pass
            elif intersection_types[j] == 2:
                pass
            elif intersection_types[j] == 3:
                pass
            else:
                pass

        _ds1_deta_deta_dr_dr_dt_dt_dtheta_j_dtheta_j_dcs.append(
            _ds1_deta_deta_dr_dr_dt_dt_dtheta_j_dtheta_j_dcn)

    return np.array(_ds1_deta_deta_dr_dr_dt_dt_dtheta_j_dtheta_j_dcs)


def ds1_deta_deta_dr_dr_dt_dt_dtheta_j_plus_1_dtheta_j_plus_1_dcs(cs):
    intersections, intersection_types, dtheta_dcss = find_intersections(cs, d, nu)

    _ds1_deta_deta_dr_dr_dt_dt_dtheta_j_plus_1_dtheta_j_plus_1_dcs =[]
    for n in N_c_s:

        _ds1_deta_deta_dr_dr_dt_dt_dtheta_j_plus_1_dtheta_j_plus_1_dcn = 0.
        for j in range(len(intersection_types)):
            if intersection_types[j] == 0:
                chain_j_plus_1 = 0.
                half_theta_range = (intersections[j + 1] - intersections[j]) / 2.
                roots, weights = roots_legendre(N_l)
                t = half_theta_range * (roots + 1.) + intersections[j]
                for k in range(N_l):
                    _ds1_deta = zeta(z_p(cs, t[k], d, nu), 1) * weights[k]
                    _deta_dr = 2. * r_p(t[k]) - d * np.cos(t[k] - nu)
                    _dr_dt = drp_dtheta(cs, t[k])
                    _dt_dtheta_j_plus_1 = (roots[k] + 1.) / 2

                    chain_j_plus_1 += _ds1_deta * _deta_dr * _dr_dt \
                                      * _dt_dtheta_j_plus_1 * dtheta_dcss[n + N_c][j + 1]

                chain_j_plus_1 *= half_theta_range
                _ds1_deta_deta_dr_dr_dt_dt_dtheta_j_plus_1_dtheta_j_plus_1_dcn += chain_j_plus_1

            elif intersection_types[j] == 1:
                pass
            elif intersection_types[j] == 2:
                pass
            elif intersection_types[j] == 3:
                pass
            else:
                pass

        _ds1_deta_deta_dr_dr_dt_dt_dtheta_j_plus_1_dtheta_j_plus_1_dcs.append(
            _ds1_deta_deta_dr_dr_dt_dt_dtheta_j_plus_1_dtheta_j_plus_1_dcn)

    return np.array(_ds1_deta_deta_dr_dr_dt_dt_dtheta_j_plus_1_dtheta_j_plus_1_dcs)


def ds1_deta_deta_drdash_drdash_dt_dt_dtheta_j_dtheta_j_dcs(cs):
    intersections, intersection_types, dtheta_dcss = find_intersections(cs, d, nu)

    _ds1_deta_deta_drdash_drdash_dt_dt_dtheta_j_dtheta_j_dcs = []
    for n in N_c_s:

        _ds1_deta_deta_drdash_drdash_dt_dt_dtheta_j_dtheta_j_dcn = 0.
        for j in range(len(intersection_types)):
            if intersection_types[j] == 0:
                chain_j = 0.
                half_theta_range = (intersections[j + 1] - intersections[j]) / 2.
                roots, weights = roots_legendre(N_l)
                t = half_theta_range * (roots + 1.) + intersections[j]
                for k in range(N_l):
                    _ds1_deta = zeta(z_p(cs, t[k], d, nu), 1) * weights[k]
                    _deta_drdash = -d * np.sin(t[k] - nu)
                    _drdash_dt = d2rp_dtheta2(cs, t[k])
                    _dt_dtheta_j = -(roots[k] + 1.) / 2 + 1

                    chain_j += _ds1_deta * _deta_drdash * _drdash_dt \
                               * _dt_dtheta_j * dtheta_dcss[n + N_c][j]

                chain_j *= half_theta_range
                _ds1_deta_deta_drdash_drdash_dt_dt_dtheta_j_dtheta_j_dcn += chain_j

            elif intersection_types[j] == 1:
                pass
            elif intersection_types[j] == 2:
                pass
            elif intersection_types[j] == 3:
                pass
            else:
                pass

        _ds1_deta_deta_drdash_drdash_dt_dt_dtheta_j_dtheta_j_dcs.append(
            _ds1_deta_deta_drdash_drdash_dt_dt_dtheta_j_dtheta_j_dcn)

    return np.array(_ds1_deta_deta_drdash_drdash_dt_dt_dtheta_j_dtheta_j_dcs)


def ds1_deta_deta_drdash_drdash_dt_dt_dtheta_j_plus_1_dtheta_j_plus_1_dcs(cs):
    intersections, intersection_types, dtheta_dcss = find_intersections(cs, d, nu)

    _ds1_deta_deta_drdash_drdash_dt_dt_dtheta_j_plus_1_dtheta_j_plus_1_dcs = []
    for n in N_c_s:

        _ds1_deta_deta_drdash_drdash_dt_dt_dtheta_j_plus_1_dtheta_j_plus_1_dcn = 0.
        for j in range(len(intersection_types)):
            if intersection_types[j] == 0:
                chain_j_plus_1 = 0.
                half_theta_range = (intersections[j + 1] - intersections[j]) / 2.
                roots, weights = roots_legendre(N_l)
                t = half_theta_range * (roots + 1.) + intersections[j]
                for k in range(N_l):
                    _ds1_deta = zeta(z_p(cs, t[k], d, nu), 1) * weights[k]
                    _deta_drdash = -d * np.sin(t[k] - nu)
                    _drdash_dt = d2rp_dtheta2(cs, t[k])
                    _dt_dtheta_j_plus_1 = (roots[k] + 1.) / 2

                    chain_j_plus_1 += _ds1_deta * _deta_drdash * _drdash_dt \
                                      * _dt_dtheta_j_plus_1 * dtheta_dcss[n + N_c][j + 1]

                chain_j_plus_1 *= half_theta_range
                _ds1_deta_deta_drdash_drdash_dt_dt_dtheta_j_plus_1_dtheta_j_plus_1_dcn += chain_j_plus_1

            elif intersection_types[j] == 1:
                pass
            elif intersection_types[j] == 2:
                pass
            elif intersection_types[j] == 3:
                pass
            else:
                pass

        _ds1_deta_deta_drdash_drdash_dt_dt_dtheta_j_plus_1_dtheta_j_plus_1_dcs.append(
            _ds1_deta_deta_drdash_drdash_dt_dt_dtheta_j_plus_1_dtheta_j_plus_1_dcn)

    return np.array(_ds1_deta_deta_drdash_drdash_dt_dt_dtheta_j_plus_1_dtheta_j_plus_1_dcs)


def ds1_deta_deta_dt_dt_dtheta_j_dtheta_j_dcs(cs):
    intersections, intersection_types, dtheta_dcss = find_intersections(cs, d, nu)

    _ds1_deta_deta_dt_dt_dtheta_j_dtheta_j_dcs = []
    for n in N_c_s:

        _ds1_deta_deta_dt_dt_dtheta_j_dtheta_j_dcn = 0.
        for j in range(len(intersection_types)):
            if intersection_types[j] == 0:
                chain_j = 0.
                half_theta_range = (intersections[j + 1] - intersections[j]) / 2.
                roots, weights = roots_legendre(N_l)
                t = half_theta_range * (roots + 1.) + intersections[j]
                for k in range(N_l):
                    _ds1_deta = zeta(z_p(cs, t[k], d, nu), 1) * weights[k]
                    _deta_dt = d * np.sin(t[k] - nu) * r_p(t[k]) \
                               - d * np.cos(t[k] - nu) * drp_dtheta(cs, t[k])
                    _dt_dtheta_j = -(roots[k] + 1.) / 2 + 1

                    chain_j += _ds1_deta * _deta_dt * _dt_dtheta_j * dtheta_dcss[n + N_c][j]

                chain_j *= half_theta_range
                _ds1_deta_deta_dt_dt_dtheta_j_dtheta_j_dcn += chain_j

            elif intersection_types[j] == 1:
                pass
            elif intersection_types[j] == 2:
                pass
            elif intersection_types[j] == 3:
                pass
            else:
                pass

        _ds1_deta_deta_dt_dt_dtheta_j_dtheta_j_dcs.append(
            _ds1_deta_deta_dt_dt_dtheta_j_dtheta_j_dcn)

    return np.array(_ds1_deta_deta_dt_dt_dtheta_j_dtheta_j_dcs)


def ds1_deta_deta_dt_dt_dtheta_j_plus_1_dtheta_j_plus_1_dcs(cs):
    intersections, intersection_types, dtheta_dcss = find_intersections(cs, d, nu)

    _ds1_deta_deta_dt_dt_dtheta_j_plus_1_dtheta_j_plus_1_dcs = []
    for n in N_c_s:

        _ds1_deta_deta_dt_dt_dtheta_j_plus_1_dtheta_j_plus_1_dcn = 0.
        for j in range(len(intersection_types)):
            if intersection_types[j] == 0:
                chain_j_plus_1 = 0.
                half_theta_range = (intersections[j + 1] - intersections[j]) / 2.
                roots, weights = roots_legendre(N_l)
                t = half_theta_range * (roots + 1.) + intersections[j]
                for k in range(N_l):
                    _ds1_deta = zeta(z_p(cs, t[k], d, nu), 1) * weights[k]
                    _deta_dt = d * np.sin(t[k] - nu) * r_p(t[k]) \
                               - d * np.cos(t[k] - nu) * drp_dtheta(cs, t[k])
                    _dt_dtheta_j_plus_1 = (roots[k] + 1.) / 2

                    chain_j_plus_1 += _ds1_deta * _deta_dt * _dt_dtheta_j_plus_1 \
                                      * dtheta_dcss[n + N_c][j + 1]

                chain_j_plus_1 *= half_theta_range
                _ds1_deta_deta_dt_dt_dtheta_j_plus_1_dtheta_j_plus_1_dcn += chain_j_plus_1

            elif intersection_types[j] == 1:
                pass
            elif intersection_types[j] == 2:
                pass
            elif intersection_types[j] == 3:
                pass
            else:
                pass

        _ds1_deta_deta_dt_dt_dtheta_j_plus_1_dtheta_j_plus_1_dcs.append(
            _ds1_deta_deta_dt_dt_dtheta_j_plus_1_dtheta_j_plus_1_dcn)

    return np.array(_ds1_deta_deta_dt_dt_dtheta_j_plus_1_dtheta_j_plus_1_dcs)


def ds1_deta_deta_dr_dr_dcs(cs):
    intersections, intersection_types, _ = find_intersections(cs, d, nu)

    _ds1_deta_deta_dr_dr_dcs = []
    for n in N_c_s:

        _ds1_deta_deta_dr_dr_dcn = 0.
        for j in range(len(intersection_types)):
            if intersection_types[j] == 0 or intersection_types[j] == 1:
                chain_j = 0.
                half_theta_range = (intersections[j + 1] - intersections[j]) / 2.
                roots, weights = roots_legendre(N_l)
                t = half_theta_range * (roots + 1.) + intersections[j]
                for k in range(N_l):
                    _ds1_deta = zeta(z_p(cs, t[k], d, nu), 1) * weights[k]
                    _deta_dr = 2. * r_p(t[k]) - d * np.cos(t[k] - nu)
                    _dr_dcn = np.exp(1j * n * t[k])

                    chain_j += _ds1_deta * _deta_dr * _dr_dcn

                chain_j *= half_theta_range
                _ds1_deta_deta_dr_dr_dcn += chain_j

            elif intersection_types[j] == 2 or intersection_types[j] == 3:
                pass
            else:
                pass

        _ds1_deta_deta_dr_dr_dcs.append(_ds1_deta_deta_dr_dr_dcn)

    return np.array(_ds1_deta_deta_dr_dr_dcs)


def ds1_deta_deta_drdash_drdash_dcs(cs):
    intersections, intersection_types, _ = find_intersections(cs, d, nu)

    _ds1_deta_deta_drdash_drdash_dcs = []
    for n in N_c_s:

        _ds1_deta_deta_drdash_drdash_dcn = 0.
        for j in range(len(intersection_types)):
            if intersection_types[j] == 0 or intersection_types[j] == 1:
                chain_j = 0.
                half_theta_range = (intersections[j + 1] - intersections[j]) / 2.
                roots, weights = roots_legendre(N_l)
                t = half_theta_range * (roots + 1.) + intersections[j]
                for k in range(N_l):
                    _ds1_deta = zeta(z_p(cs, t[k], d, nu), 1) * weights[k]
                    _deta_drdash = -d * np.sin(t[k] - nu)
                    _drdash_dcn = 1j * n * np.exp(1j * n * t[k])

                    chain_j += _ds1_deta * _deta_drdash * _drdash_dcn

                chain_j *= half_theta_range
                _ds1_deta_deta_drdash_drdash_dcn += chain_j

            elif intersection_types[j] == 2 or intersection_types[j] == 3:
                pass
            else:
                pass

        _ds1_deta_deta_drdash_drdash_dcs.append(_ds1_deta_deta_drdash_drdash_dcn)

    return np.array(_ds1_deta_deta_drdash_drdash_dcs)


def dF_dabs_total(cs):
    us = np.array([1., u1, u2])
    I_0 = 1. / (np.pi * (1. - us[1] / 3. - us[2] / 6.))

    dF_dalpha = -1.

    # S0.
    dalpha_ds0 = I_0 * (1 - u1 - u2)
    _ds0_dq0_dq0_dcs = ds0_dq0_dq0_dcs(cs)
    _ds0_dtheta_j_dtheta_j_dcs = ds0_dtheta_j_dtheta_j_dcs(cs)
    _ds0_dtheta_j_plus_1_dtheta_j_plus_1_dcs = ds0_dtheta_j_plus_1_dtheta_j_plus_1_dcs(cs)
    _ds0_dphi_j_dphi_j_drp_drp_dcs = ds0_dphi_j_dphi_j_drp_drp_dcs(cs)
    _ds0_dphi_j_plus_1_dphi_j_plus_1_drp_drp_dcs = ds0_dphi_j_plus_1_dphi_j_plus_1_drp_drp_dcs(cs)
    _ds0_dphi_j_dphi_j_dtheta_j_dtheta_j_dcs = ds0_dphi_j_dphi_j_dtheta_j_dtheta_j_dcs(cs)
    _ds0_dphi_j_plus_1_dphi_j_plus_1_dtheta_j_plus_1_dtheta_j_plus_1_dcs = \
        ds0_dphi_j_plus_1_dphi_j_plus_1_dtheta_j_plus_1_dtheta_j_plus_1_dcs(cs)
    _ds0_dphi_j_dphi_j_drp_drp_dtheta_j_dtheta_j_dcs = \
        ds0_dphi_j_dphi_j_drp_drp_dtheta_j_dtheta_j_dcs(cs)
    _ds0_dphi_j_plus_1_dphi_j_plus_1_drp_drp_dtheta_j_plus_1_dtheta_j_plus_1_dcs = \
        ds0_dphi_j_plus_1_dphi_j_plus_1_drp_drp_dtheta_j_plus_1_dtheta_j_plus_1_dcs(cs)

    ds0_dcs_total = _ds0_dq0_dq0_dcs \
                    + _ds0_dtheta_j_dtheta_j_dcs \
                    + _ds0_dtheta_j_plus_1_dtheta_j_plus_1_dcs \
                    + _ds0_dphi_j_dphi_j_drp_drp_dcs \
                    + _ds0_dphi_j_plus_1_dphi_j_plus_1_drp_drp_dcs \
                    + _ds0_dphi_j_dphi_j_dtheta_j_dtheta_j_dcs \
                    + _ds0_dphi_j_plus_1_dphi_j_plus_1_dtheta_j_plus_1_dtheta_j_plus_1_dcs \
                    + _ds0_dphi_j_dphi_j_drp_drp_dtheta_j_dtheta_j_dcs \
                    + _ds0_dphi_j_plus_1_dphi_j_plus_1_drp_drp_dtheta_j_plus_1_dtheta_j_plus_1_dcs

    # S1.
    dalpha_ds1 = I_0 * (u1 + 2. * u2)
    _ds1_dzeta_dzeta_dz_dz_dr_dr_dt_dt_dtheta_j_dtheta_j_dcs = \
        ds1_dzeta_dzeta_dz_dz_dr_dr_dt_dt_dtheta_j_dtheta_j_dcs(cs)
    _ds1_dzeta_dzeta_dz_dz_dr_dr_dt_dt_dtheta_j_plus_1_dtheta_j_plus_1_dcs = \
        ds1_dzeta_dzeta_dz_dz_dr_dr_dt_dt_dtheta_j_plus_1_dtheta_j_plus_1_dcs(cs)
    _ds1_dzeta_dzeta_dz_dz_dt_dt_dtheta_j_dtheta_j_dcs = \
        ds1_dzeta_dzeta_dz_dz_dt_dt_dtheta_j_dtheta_j_dcs(cs)
    _ds1_dzeta_dzeta_dz_dz_dt_dt_dtheta_j_plus_1_dtheta_j_plus_1_dcs = \
        ds1_dzeta_dzeta_dz_dz_dt_dt_dtheta_j_plus_1_dtheta_j_plus_1_dcs(cs)
    _ds1_dzeta_dzeta_dz_dz_dr_dr_dcs = \
        ds1_dzeta_dzeta_dz_dz_dr_dr_dcs(cs)
    _ds1_deta_deta_dr_dr_dt_dt_dtheta_j_dtheta_j_dcs = \
        ds1_deta_deta_dr_dr_dt_dt_dtheta_j_dtheta_j_dcs(cs)
    _ds1_deta_deta_dr_dr_dt_dt_dtheta_j_plus_1_dtheta_j_plus_1_dcs = \
        ds1_deta_deta_dr_dr_dt_dt_dtheta_j_plus_1_dtheta_j_plus_1_dcs(cs)
    _ds1_deta_deta_drdash_drdash_dt_dt_dtheta_j_dtheta_j_dcs = \
        ds1_deta_deta_drdash_drdash_dt_dt_dtheta_j_dtheta_j_dcs(cs)
    _ds1_deta_deta_drdash_drdash_dt_dt_dtheta_j_plus_1_dtheta_j_plus_1_dcs = \
        ds1_deta_deta_drdash_drdash_dt_dt_dtheta_j_plus_1_dtheta_j_plus_1_dcs(cs)
    _ds1_deta_deta_dt_dt_dtheta_j_dtheta_j_dcs = \
        ds1_deta_deta_dt_dt_dtheta_j_dtheta_j_dcs(cs)
    _ds1_deta_deta_dt_dt_dtheta_j_plus_1_dtheta_j_plus_1_dcs = \
        ds1_deta_deta_dt_dt_dtheta_j_plus_1_dtheta_j_plus_1_dcs(cs)
    _ds1_deta_deta_dr_dr_dcs = ds1_deta_deta_dr_dr_dcs(cs)
    _ds1_deta_deta_drdash_drdash_dcs = ds1_deta_deta_drdash_drdash_dcs(cs)
    _ds1_dtheta_j_dtheta_j_dcs = ds1_dtheta_j_dtheta_j_dcs(cs)
    _ds1_dtheta_j_plus_1_dtheta_j_plus_1_dcs = \
        ds1_dtheta_j_plus_1_dtheta_j_plus_1_dcs(cs)
    _ds1_dphi_j_dphi_j_drp_drp_dcs = ds1_dphi_j_dphi_j_drp_drp_dcs(cs)
    _ds1_dphi_j_plus_1_dphi_j_plus_1_drp_drp_dcs = ds1_dphi_j_plus_1_dphi_j_plus_1_drp_drp_dcs(cs)
    _ds1_dphi_j_dphi_j_dtheta_j_dtheta_j_dcs = ds1_dphi_j_dphi_j_dtheta_j_dtheta_j_dcs(cs)
    _ds1_dphi_j_plus_1_dphi_j_plus_1_dtheta_j_plus_1_dtheta_j_plus_1_dcs = \
        ds1_dphi_j_plus_1_dphi_j_plus_1_dtheta_j_plus_1_dtheta_j_plus_1_dcs(cs)
    _ds1_dphi_j_dphi_j_drp_drp_dtheta_j_dtheta_j_dcs = \
        ds1_dphi_j_dphi_j_drp_drp_dtheta_j_dtheta_j_dcs(cs)
    _ds1_dphi_j_plus_1_dphi_j_plus_1_drp_drp_dtheta_j_plus_1_dtheta_j_plus_1_dcs = \
        ds1_dphi_j_plus_1_dphi_j_plus_1_drp_drp_dtheta_j_plus_1_dtheta_j_plus_1_dcs(cs)

    ds1_dcs_total = _ds1_dzeta_dzeta_dz_dz_dr_dr_dt_dt_dtheta_j_dtheta_j_dcs \
                    + _ds1_dzeta_dzeta_dz_dz_dr_dr_dt_dt_dtheta_j_plus_1_dtheta_j_plus_1_dcs \
                    + _ds1_dzeta_dzeta_dz_dz_dt_dt_dtheta_j_dtheta_j_dcs \
                    + _ds1_dzeta_dzeta_dz_dz_dt_dt_dtheta_j_plus_1_dtheta_j_plus_1_dcs \
                    + _ds1_dzeta_dzeta_dz_dz_dr_dr_dcs \
                    + _ds1_deta_deta_dr_dr_dt_dt_dtheta_j_dtheta_j_dcs \
                    + _ds1_deta_deta_dr_dr_dt_dt_dtheta_j_plus_1_dtheta_j_plus_1_dcs \
                    + _ds1_deta_deta_drdash_drdash_dt_dt_dtheta_j_dtheta_j_dcs \
                    + _ds1_deta_deta_drdash_drdash_dt_dt_dtheta_j_plus_1_dtheta_j_plus_1_dcs \
                    + _ds1_deta_deta_dt_dt_dtheta_j_dtheta_j_dcs \
                    + _ds1_deta_deta_dt_dt_dtheta_j_plus_1_dtheta_j_plus_1_dcs \
                    + _ds1_deta_deta_dr_dr_dcs \
                    + _ds1_deta_deta_drdash_drdash_dcs \
                    + _ds1_dtheta_j_dtheta_j_dcs \
                    + _ds1_dtheta_j_plus_1_dtheta_j_plus_1_dcs \
                    + _ds1_dphi_j_dphi_j_drp_drp_dcs \
                    + _ds1_dphi_j_plus_1_dphi_j_plus_1_drp_drp_dcs \
                    + _ds1_dphi_j_dphi_j_dtheta_j_dtheta_j_dcs \
                    + _ds1_dphi_j_plus_1_dphi_j_plus_1_dtheta_j_plus_1_dtheta_j_plus_1_dcs \
                    + _ds1_dphi_j_dphi_j_drp_drp_dtheta_j_dtheta_j_dcs \
                    + _ds1_dphi_j_plus_1_dphi_j_plus_1_drp_drp_dtheta_j_plus_1_dtheta_j_plus_1_dcs

    # S2.
    dalpha_ds2 = -I_0 * u2
    _ds2_dq2_dq2_dcs = ds2_dq2_dq2_dcs(cs)
    _ds2_dtheta_j_dtheta_j_dcs = ds2_dtheta_j_dtheta_j_dcs(cs)
    _ds2_dtheta_j_plus_1_dtheta_j_plus_1_dcs = ds2_dtheta_j_plus_1_dtheta_j_plus_1_dcs(cs)
    _ds2_dphi_j_dphi_j_drp_drp_dcs = ds2_dphi_j_dphi_j_drp_drp_dcs(cs)
    _ds2_dphi_j_plus_1_dphi_j_plus_1_drp_drp_dcs = ds2_dphi_j_plus_1_dphi_j_plus_1_drp_drp_dcs(cs)
    _ds2_dphi_j_dphi_j_dtheta_j_dtheta_j_dcs = ds2_dphi_j_dphi_j_dtheta_j_dtheta_j_dcs(cs)
    _ds2_dphi_j_plus_1_dphi_j_plus_1_dtheta_j_plus_1_dtheta_j_plus_1_dcs = \
        ds2_dphi_j_plus_1_dphi_j_plus_1_dtheta_j_plus_1_dtheta_j_plus_1_dcs(cs)
    _ds2_dphi_j_dphi_j_drp_drp_dtheta_j_dtheta_j_dcs = \
        ds2_dphi_j_dphi_j_drp_drp_dtheta_j_dtheta_j_dcs(cs)
    _ds2_dphi_j_plus_1_dphi_j_plus_1_drp_drp_dtheta_j_plus_1_dtheta_j_plus_1_dcs = \
        ds2_dphi_j_plus_1_dphi_j_plus_1_drp_drp_dtheta_j_plus_1_dtheta_j_plus_1_dcs(cs)

    ds2_dcs_total = _ds2_dq2_dq2_dcs \
                     + _ds2_dtheta_j_dtheta_j_dcs \
                     + _ds2_dtheta_j_plus_1_dtheta_j_plus_1_dcs \
                     + _ds2_dphi_j_dphi_j_drp_drp_dcs \
                     + _ds2_dphi_j_plus_1_dphi_j_plus_1_drp_drp_dcs \
                     + _ds2_dphi_j_dphi_j_dtheta_j_dtheta_j_dcs \
                     + _ds2_dphi_j_plus_1_dphi_j_plus_1_dtheta_j_plus_1_dtheta_j_plus_1_dcs \
                     + _ds2_dphi_j_dphi_j_drp_drp_dtheta_j_dtheta_j_dcs \
                     + _ds2_dphi_j_plus_1_dphi_j_plus_1_drp_drp_dtheta_j_plus_1_dtheta_j_plus_1_dcs

    _dF_dcs_total = dF_dalpha * (dalpha_ds0 * ds0_dcs_total
                                 + dalpha_ds1 * ds1_dcs_total
                                 + dalpha_ds2 * ds2_dcs_total)

    _dc_dabs = []
    for n in range(N_c + 1):

        if n == 0:
            _dc0_da0 = 1.
            _dF_da0_total = _dF_dcs_total[0 + N_c] * _dc0_da0
            _dc_dabs.append(_dF_da0_total)
        else:
            _dc_plus_n_dan = 1. / 2.
            _dc_minus_n_dan = 1. / 2.
            _dF_dan_total = _dF_dcs_total[n + N_c] * _dc_plus_n_dan + \
                            _dF_dcs_total[-n + N_c] * _dc_minus_n_dan
            _dc_dabs.append(_dF_dan_total)

            _dc_plus_n_dbn = -1j / 2.
            _dc_minus_n_dbn = 1j / 2.
            _dF_dbn_total = _dF_dcs_total[n + N_c] * _dc_plus_n_dbn \
                            + _dF_dcs_total[-n + N_c] * _dc_minus_n_dbn
            _dc_dabs.append(_dF_dbn_total)

    return np.array(np.real(_dc_dabs))


def grad_arrow(x_draw, x, y, grad):
    c = y - grad * x
    return grad * x_draw + c


while True:
    N_c = 1
    N_c_s = np.arange(-N_c, N_c + 1, 1)

    a0_a = np.random.uniform(0.05, 0.2)
    a_s = np.array([a0_a, -0.003, ])
    b_s = np.array([0.003])
    cs_a = generate_complex_fourier_coeffs(a_s, b_s)
    F_a = F_func(cs_a)

    delta = 1.e-7
    a0_b = a0_a + delta
    a_s = np.array([a0_b, -0.003])
    b_s = np.array([0.003])
    cs_b = generate_complex_fourier_coeffs(a_s, b_s)
    F_b = F_func(cs_b)

    a_s = np.array([a0_a, -0.003])
    b_s = np.array([0.003])
    cs_a = generate_complex_fourier_coeffs(a_s, b_s)
    a0_a_grad = dF_dabs_total(cs_a)[0]

    plt.scatter(a0_a, F_a, label='$a_0$')
    plt.scatter(a0_b, F_b, label='$a_0 + \delta$: $\delta={}$'.format(delta))
    x_arrow = np.linspace(a0_a, a0_b, 2)
    plt.plot(x_arrow, grad_arrow(x_arrow, a0_a, F_a, a0_a_grad),
             label='Gradient: $\\frac{dF}{da_0}$')
    print(F_a, F_b, (a0_a_grad * delta + F_a) - F_b)

    plt.legend()
    plt.xlabel('$a_0$')
    plt.ylabel('$F$')
    plt.tight_layout()
    plt.show()

    N_c = 1
    N_c_s = np.arange(-N_c, N_c + 1, 1)

    a1_a = np.random.uniform(-0.003, 0.003)
    a_s = np.array([0.1, a1_a])
    b_s = np.array([0.003])
    cs_a = generate_complex_fourier_coeffs(a_s, b_s)
    F_a = F_func(cs_a)

    delta = 1.e-7
    a1_b = a1_a + delta
    a_s = np.array([0.1, a1_b])
    b_s = np.array([0.003])
    cs_b = generate_complex_fourier_coeffs(a_s, b_s)
    F_b = F_func(cs_b)

    a_s = np.array([0.1, a1_a])
    b_s = np.array([0.003])
    cs_a = generate_complex_fourier_coeffs(a_s, b_s)
    a1_a_grad = dF_dabs_total(cs_a)[1]

    plt.scatter(a1_a, F_a, label='$a_1$')
    plt.scatter(a1_b, F_b, label='$a_1 + \delta$: $\delta={}$'.format(delta))
    x_arrow = np.linspace(a1_a, a1_b, 2)
    plt.plot(x_arrow, grad_arrow(x_arrow, a1_a, F_a, a1_a_grad),
             label='Gradient: $\\frac{dF}{da_1}$')
    print(F_a, F_b, (a1_a_grad * delta + F_a) - F_b)

    plt.legend()
    plt.xlabel('$a_1$')
    plt.ylabel('$F$')
    plt.tight_layout()
    plt.show()

    N_c = 1
    N_c_s = np.arange(-N_c, N_c + 1, 1)

    b1_a = np.random.uniform(-0.003, 0.003)
    a_s = np.array([0.1, -0.003])
    b_s = np.array([b1_a])
    cs_a = generate_complex_fourier_coeffs(a_s, b_s)
    F_a = F_func(cs_a)

    delta = 1.e-7
    b1_b = b1_a + delta
    a_s = np.array([0.1, -0.003])
    b_s = np.array([b1_b])
    cs_b = generate_complex_fourier_coeffs(a_s, b_s)
    F_b = F_func(cs_b)

    a_s = np.array([0.1, -0.003])
    b_s = np.array([b1_a])
    cs_a = generate_complex_fourier_coeffs(a_s, b_s)
    b1_a_grad = dF_dabs_total(cs_a)[2]

    plt.scatter(b1_a, F_a, label='$b_1$')
    plt.scatter(b1_b, F_b, label='$b_1 + \delta$: $\delta={}$'.format(delta))
    x_arrow = np.linspace(b1_a, b1_b, 2)
    plt.plot(x_arrow, grad_arrow(x_arrow, b1_a, F_a, b1_a_grad),
             label='Gradient: $\\frac{dF}{db_1}$')
    print(F_a, F_b, (b1_a_grad * delta + F_a) - F_b)

    plt.legend()
    plt.xlabel('$b_1$')
    plt.ylabel('$F$')
    plt.tight_layout()
    plt.show()
