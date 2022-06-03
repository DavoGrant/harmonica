import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad


def generate_complex_fourier_coeffs(_as, _bs):
    _cs = [_as[0]]
    for _a, _b in zip(_as[1:], _bs):
        _cs.append((_a - 1j * _b) / 2)
        _cs.insert(0, (_a + 1j * _b) / 2)
    return _cs


# Config.
u1 = 0.4
u2 = 0.3
a_s = np.array([0.1, -0.003, 0.001])
b_s = np.array([0.003, 0.001])
cs = generate_complex_fourier_coeffs(a_s, b_s)
N_c = int((len(cs) - 1) / 2)
N_c_s = np.arange(-N_c, N_c + 1, 1)
nu = 0.1
epsilon = 1.e-7


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


def da_p_values_dd(p, _cs, _d, _nu):
    a_p = 0.
    if 0 <= p < N_c - 1:
        pass
    elif N_c - 1 <= p < N_c + 1:
        a_p += -np.exp(1j * _nu) * _cs[(p - 2 * N_c + 1) + N_c]
    elif N_c + 1 <= p < 2 * N_c:
        a_p += -np.exp(1j * _nu) * _cs[(p - 2 * N_c + 1) + N_c]
        a_p += -np.exp(-1j * _nu) * _cs[(p - 2 * N_c - 1) + N_c]
    elif p == 2 * N_c:
        a_p += -np.exp(1j * _nu) * _cs[(p - 2 * N_c + 1) + N_c]
        a_p += -np.exp(-1j * _nu) * _cs[(p - 2 * N_c - 1) + N_c]
        a_p += 2 * _d
    elif 2 * N_c + 1 <= p < 3 * N_c:
        a_p += -np.exp(1j * _nu) * _cs[(p - 2 * N_c + 1) + N_c]
        a_p += -np.exp(-1j * _nu) * _cs[(p - 2 * N_c - 1) + N_c]
    elif 3 * N_c - 0 <= p < 3 * N_c + 2:
        a_p += -np.exp(-1j * _nu) * _cs[(p - 2 * N_c - 1) + N_c]
    elif 3 * N_c + 2 <= p < 4 * N_c + 1:
        pass

    return a_p


def c_jk_values(j, k, _cs, _d, _nu):
    if k == 4 * N_c:
        return -a_p_values(j - 1, _cs, _d, _nu) / a_p_values(4 * N_c, _cs, _d, _nu)
    else:
        if j == k + 1:
            return 1.
        else:
            return 0.


def dc_jk_values_dd(j, k, _cs, _d, _nu):
    if k == 4 * N_c:
        return (a_p_values(j - 1, _cs, _d, _nu) * da_p_values_dd(4 * N_c, _cs, _d, _nu)
                - da_p_values_dd(j - 1, _cs, _d, _nu) * a_p_values(4 * N_c, _cs, _d, _nu)) \
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


def dcompanion_matrix_dd(_cs, _d, _nu):
    assert N_c >= 1
    dC_dd = np.zeros((4 * N_c, 4 * N_c), dtype=np.cdouble)
    for (j, k), value in np.ndenumerate(dC_dd):
        dC_dd[j, k] = dc_jk_values_dd(j + 1, k + 1, _cs, _d, _nu)
    return dC_dd


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

    dC_dd_total = dcompanion_matrix_dd(cs, _d, nu)
    dlambda_dd_total = np.diag(np.linalg.inv(eigenvectors) @ dC_dd_total @ eigenvectors)

    intersections_not_sorted = []
    dtheta_dds = []
    for idx, e_val in enumerate(eigenvalues):
        if 1. - epsilon < np.abs(e_val) < 1. + epsilon:
            intersections_not_sorted.append(np.angle(e_val))
            dtheta_dds.append(np.real(-1j / e_val * dlambda_dd_total[idx]))

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
        dtheta_dds = [x for _, x in sorted(zip(intersections_not_sorted, dtheta_dds))]
        dtheta_dds.append(dtheta_dds[0])
        dtheta_dds = np.array(dtheta_dds)
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

    return intersections, intersection_types, dtheta_dds


def drp_dtheta(_cs, _theta):
    _drp_dtheta = 0.
    for k, n, in enumerate(N_c_s):
        _drp_dtheta += 1j * n * _cs[k] * np.exp(1j * n * _theta)
    return np.real(_drp_dtheta)


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


def line_integral_star(_theta, _cs, _d, _nu, n_val):
    return zeta(z_p(_cs, _theta, _d, _nu), n_val) * eta(_theta, _cs, _d, _nu)


def F_func(d):
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
            # s1, s1_err = quad(
            #     line_integral_planet, intersections[j], intersections[j + 1],
            #     args=(cs, d, nu, 1),
            #     epsabs=1.e-15, epsrel=1.e-15, limit=500)
            s1, s1_err = (1., 0.)
            # s2, s2_err = quad(
            #     line_integral_planet, intersections[j], intersections[j + 1],
            #     args=(cs, d, nu, 2),
            #     epsabs=1.e-15, epsrel=1.e-15, limit=500)
            s2, s2_err = (1., 0.)
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
            # s1 = 1. / (1. + 2) * (phi_j_plus_1 - phi_j)
            s1 = 1.
            # s2 = 1. / (2. + 2) * (phi_j_plus_1 - phi_j)
            s2 = 1.
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


def ds0_dq0_dq0_dd(d):
    intersections, intersection_types, _ = find_intersections(cs, d, nu)

    N_q = int(2. / 4. * (max(4 * N_c + 1, 2 * N_c + 3) - 1))
    beta_cos = np.cos(nu) * np.array([0.5, 0, 0.5]) \
               + np.sin(nu) * np.array([0.5j, 0, -0.5j])
    beta_sin = np.cos(nu) * np.array([0.5j, 0, -0.5j]) \
               - np.sin(nu) * np.array([0.5, 0, 0.5])

    zero_vec = np.zeros(2 * (N_c * 2 + 1) - 1)
    Delta_cs = 1j * N_c_s * cs
    dq0_dd = 1. / 2 * add_arrays_centre_aligned(
        zero_vec,
        np.convolve(-beta_cos, cs) + np.convolve(-beta_sin, Delta_cs))

    _ds0_dq0_dq0_dd = 0.
    for j in range(len(intersection_types)):
        # Todo: if type 1 return 0.
        if intersection_types[j] == 0 or intersection_types[j] == 1:
            N_q_s = np.arange(-N_q, N_q + 1, 1)
            for m in N_q_s:
                if m == 0:
                    _ds0_dq0_dq0_dd += (intersections[j + 1] - intersections[j]) \
                                       * dq0_dd[m + N_q]
                else:
                    _ds0_dq0_dq0_dd += 1. / (1j * m) * (
                            np.exp(1j * m * intersections[j + 1])
                            - np.exp(1j * m * intersections[j])) * dq0_dd[m + N_q]
        elif intersection_types[j] == 2 or intersection_types[j] == 3:
            pass
        else:
            pass

    return np.real(_ds0_dq0_dq0_dd)


def ds0_dtheta_j_dtheta_j_dd(d):
    intersections, intersection_types, dtheta_dds = find_intersections(cs, d, nu)

    N_q = int(2 / 4 * (max(4 * N_c + 1, 2 * N_c + 3) - 1))
    beta_cos = np.cos(nu) * np.array([0.5, 0, 0.5]) \
               + np.sin(nu) * np.array([0.5j, 0, -0.5j])
    beta_sin = np.cos(nu) * np.array([0.5j, 0, -0.5j]) \
               - np.sin(nu) * np.array([0.5, 0, 0.5])

    Delta_cs = 1j * N_c_s * cs
    q0 = 1. / 2. * add_arrays_centre_aligned(
        np.convolve(cs, cs),
        np.convolve(-d * beta_cos, cs) + np.convolve(-d * beta_sin, Delta_cs))

    _ds0_dtheta_j_dtheta_j_dd = 0.
    for j in range(len(intersection_types)):
        if intersection_types[j] == 0:
            N_q_s = np.arange(-N_q, N_q + 1, 1)
            _ds0_dtheta_j = 0.
            for m in N_q_s:
                _ds0_dtheta_j += -q0[m + N_q] * np.exp(1j * m * intersections[j])
            _ds0_dtheta_j_dtheta_j_dd += _ds0_dtheta_j * dtheta_dds[j]
        elif intersection_types[j] == 2:
            pass
        elif intersection_types[j] == 3:
            pass
        elif intersection_types[j] == 4:
            pass
        else:
            pass

    return np.real(_ds0_dtheta_j_dtheta_j_dd)


def ds0_dtheta_j_plus_1_dtheta_j_plus_1_dd(d):
    intersections, intersection_types, dtheta_dds = find_intersections(cs, d, nu)

    N_q = int(2 / 4 * (max(4 * N_c + 1, 2 * N_c + 3) - 1))
    beta_cos = np.cos(nu) * np.array([0.5, 0, 0.5]) \
               + np.sin(nu) * np.array([0.5j, 0, -0.5j])
    beta_sin = np.cos(nu) * np.array([0.5j, 0, -0.5j]) \
               - np.sin(nu) * np.array([0.5, 0, 0.5])

    Delta_cs = 1j * N_c_s * cs
    q0 = 1. / 2. * add_arrays_centre_aligned(
        np.convolve(cs, cs),
        np.convolve(-d * beta_cos, cs) + np.convolve(-d * beta_sin, Delta_cs))

    _ds0_dtheta_j_dtheta_j_plus_1_dd = 0.
    for j in range(len(intersection_types)):
        if intersection_types[j] == 0:
            N_q_s = np.arange(-N_q, N_q + 1, 1)
            _ds0_dtheta_j_plus_1 = 0.
            for m in N_q_s:
                _ds0_dtheta_j_plus_1 += q0[m + N_q] * np.exp(1j * m * intersections[j + 1])
            _ds0_dtheta_j_dtheta_j_plus_1_dd += _ds0_dtheta_j_plus_1 * dtheta_dds[j + 1]
        elif intersection_types[j] == 1:
            pass
        elif intersection_types[j] == 2:
            pass
        elif intersection_types[j] == 3:
            pass
        else:
            pass

    return np.real(_ds0_dtheta_j_dtheta_j_plus_1_dd)


def ds0_dphi_j_dphi_j_dd(d):
    intersections, intersection_types, _ = find_intersections(cs, d, nu)

    _ds0_dphi_j_dphi_j_dd = 0.
    for j in range(len(intersection_types)):
        if intersection_types[j] == 0:
            pass
        elif intersection_types[j] == 1:
            pass
        elif intersection_types[j] == 2:
            _rp_j = r_p(intersections[j])
            _ds0_dphi_j = -1. / 2.
            _dphi_j_dd = (_rp_j * np.sin(intersections[j] - nu)) \
                         / (d**2 - 2. * d * _rp_j * np.cos(intersections[j] - nu)
                            + _rp_j**2)
            _ds0_dphi_j_dphi_j_dd += _ds0_dphi_j * _dphi_j_dd
        elif intersection_types[j] == 3:
            pass
        else:
            pass

    return _ds0_dphi_j_dphi_j_dd


def ds0_dphi_j_plus_1_dphi_j_plus_1_dd(d):
    intersections, intersection_types, _ = find_intersections(cs, d, nu)

    _ds0_dphi_j_plus_1_dphi_j_plus_1_dd = 0.
    for j in range(len(intersection_types)):
        if intersection_types[j] == 0:
            pass
        elif intersection_types[j] == 1:
            pass
        elif intersection_types[j] == 2:
            _rp_j_plus_1 = r_p(intersections[j + 1])
            _ds0_dphi_j_plus_1 = 1. / 2.
            _dphi_j_plus_1_dd = \
                (_rp_j_plus_1 * np.sin(intersections[j + 1] - nu)) \
                / (-2 * d * _rp_j_plus_1 * np.cos(intersections[j + 1] - nu)
                   + d**2 + _rp_j_plus_1**2)
            _ds0_dphi_j_plus_1_dphi_j_plus_1_dd += _ds0_dphi_j_plus_1 * _dphi_j_plus_1_dd
        elif intersection_types[j] == 3:
            pass
        else:
            pass

    return _ds0_dphi_j_plus_1_dphi_j_plus_1_dd


def ds0_dphi_j_dphi_j_dtheta_j_dtheta_j_dd(d):
    intersections, intersection_types, dtheta_dds = find_intersections(cs, d, nu)

    _ds0_dphi_j_dphi_j_dtheta_j_dtheta_j_dd = 0.
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
            _dtheta_j_dd = dtheta_dds[j]
            _ds0_dphi_j_dphi_j_dtheta_j_dtheta_j_dd += _ds0_dphi_j * _dphi_j_dtheta_j \
                                                       * _dtheta_j_dd
        elif intersection_types[j] == 3:
            pass
        else:
            pass

    return _ds0_dphi_j_dphi_j_dtheta_j_dtheta_j_dd


def ds0_dphi_j_plus_1_dphi_j_plus_1_dtheta_j_plus_1_dtheta_j_plus_1_dd(d):
    intersections, intersection_types, dtheta_dds = find_intersections(cs, d, nu)

    _ds0_dphi_j_plus_1_dphi_j_plus_1_dtheta_j_plus_1_dtheta_j_plus_1_dd = 0.
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
            _dtheta_j_plus_1_dd = dtheta_dds[j + 1]
            _ds0_dphi_j_plus_1_dphi_j_plus_1_dtheta_j_plus_1_dtheta_j_plus_1_dd += \
                _ds0_dphi_j_plus_1 * _dphi_j_plus_1_dtheta_j_plus_1 * _dtheta_j_plus_1_dd
        elif intersection_types[j] == 3:
            pass
        else:
            pass

    return _ds0_dphi_j_plus_1_dphi_j_plus_1_dtheta_j_plus_1_dtheta_j_plus_1_dd


def ds0_dphi_j_dphi_j_drp_drp_dtheta_j_dtheta_j_dd(d):
    intersections, intersection_types, dtheta_dds = find_intersections(cs, d, nu)

    _ds0_dphi_j_dphi_j_drp_drp_dtheta_j_dtheta_j_dd = 0.
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
            _dtheta_j_dd = dtheta_dds[j]
            _ds0_dphi_j_dphi_j_drp_drp_dtheta_j_dtheta_j_dd += \
                _ds0_dphi_j * _dphi_j_drp * _drp_dtheta_j * _dtheta_j_dd
        elif intersection_types[j] == 3:
            pass
        else:
            pass

    return _ds0_dphi_j_dphi_j_drp_drp_dtheta_j_dtheta_j_dd


def ds0_dphi_j_plus_1_dphi_j_plus_1_drp_drp_dtheta_j_plus_1_dtheta_j_plus_1_dd(d):
    intersections, intersection_types, dtheta_dds = find_intersections(cs, d, nu)

    _ds0_dphi_j_plus_1_dphi_j_plus_1_drp_drp_dtheta_j_plus_1_dtheta_j_plus_1_dd = 0.
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
            _dtheta_j_plus_1_dd = dtheta_dds[j + 1]
            _ds0_dphi_j_plus_1_dphi_j_plus_1_drp_drp_dtheta_j_plus_1_dtheta_j_plus_1_dd += \
                _ds0_dphi_j_plus_1 * _dphi_j_plus_1_drp * _drp_dtheta_j_plus_1 \
                * _dtheta_j_plus_1_dd
        elif intersection_types[j] == 3:
            pass
        else:
            pass

    return _ds0_dphi_j_plus_1_dphi_j_plus_1_drp_drp_dtheta_j_plus_1_dtheta_j_plus_1_dd


def dF_dd_total(d):
    us = np.array([1., u1, u2])
    I_0 = 1. / (np.pi * (1. - us[1] / 3. - us[2] / 6.))

    dF_dalpha = -1.

    # S0.
    dalpha_ds0 = I_0 * (1 - u1 - u2)
    _ds0_dq0_dq0_dd = ds0_dq0_dq0_dd(d)
    _ds0_dtheta_j_dtheta_j_dd = ds0_dtheta_j_dtheta_j_dd(d)
    _ds0_dtheta_j_plus_1_dtheta_j_plus_1_dd = ds0_dtheta_j_plus_1_dtheta_j_plus_1_dd(d)
    _ds0_dphi_j_dphi_j_dd = ds0_dphi_j_dphi_j_dd(d)
    _ds0_dphi_j_dphi_j_plus_1_dd = ds0_dphi_j_plus_1_dphi_j_plus_1_dd(d)
    _ds0_dphi_j_dphi_j_dtheta_j_dtheta_j_dd = ds0_dphi_j_dphi_j_dtheta_j_dtheta_j_dd(d)
    _ds0_dphi_j_plus_1_dphi_j_plus_1_dtheta_j_plus_1_dtheta_j_plus_1_dd = \
        ds0_dphi_j_plus_1_dphi_j_plus_1_dtheta_j_plus_1_dtheta_j_plus_1_dd(d)
    _ds0_dphi_j_dphi_j_drp_drp_dtheta_j_dtheta_j_dd = ds0_dphi_j_dphi_j_drp_drp_dtheta_j_dtheta_j_dd(d)
    _ds0_dphi_j_plus_1_dphi_j_plus_1_drp_drp_dtheta_j_plus_1_dtheta_j_plus_1_dd = \
        ds0_dphi_j_plus_1_dphi_j_plus_1_drp_drp_dtheta_j_plus_1_dtheta_j_plus_1_dd(d)

    ds0_dd_total = _ds0_dq0_dq0_dd \
                   + _ds0_dtheta_j_dtheta_j_dd \
                   + _ds0_dtheta_j_plus_1_dtheta_j_plus_1_dd \
                   + _ds0_dphi_j_dphi_j_dd \
                   + _ds0_dphi_j_dphi_j_plus_1_dd \
                   + _ds0_dphi_j_dphi_j_dtheta_j_dtheta_j_dd \
                   + _ds0_dphi_j_plus_1_dphi_j_plus_1_dtheta_j_plus_1_dtheta_j_plus_1_dd \
                   + _ds0_dphi_j_dphi_j_drp_drp_dtheta_j_dtheta_j_dd \
                   + _ds0_dphi_j_plus_1_dphi_j_plus_1_drp_drp_dtheta_j_plus_1_dtheta_j_plus_1_dd

    # S1.
    dalpha_ds1 = I_0 * (u1 + 2. * u2)
    ds1_dd_total = 0.

    # S2.
    dalpha_ds2 = -I_0 * u2
    ds2_dd_total = 0.

    # Todo is gradient real?
    _dF_dd_total = dF_dalpha * (dalpha_ds0 * ds0_dd_total +
                                dalpha_ds0 * ds1_dd_total +
                                dalpha_ds0 * ds2_dd_total)

    return _dF_dd_total


def grad_arrow(x_draw, x, y, grad):
    c = y - grad * x
    return grad * x_draw + c


while True:
    d_a = np.random.uniform(0.85, 1.15)
    F_a = F_func(d_a)

    delta = 1.e-6
    d_b = d_a + delta
    F_b = F_func(d_b)

    d_a_grad = dF_dd_total(d_a)

    plt.scatter(d_a, F_a, label='$u_1$')
    plt.scatter(d_b, F_b, label='$u_1 + \delta$: $\delta={}$'.format(delta))
    x_arrow = np.linspace(d_a, d_b, 2)
    plt.plot(x_arrow, grad_arrow(x_arrow, d_a, F_a, d_a_grad),
             label='Gradient: $\\frac{dF}{d u_1}$')

    plt.legend()
    plt.xlabel('$d$')
    plt.ylabel('$F$')
    plt.tight_layout()
    plt.show()
