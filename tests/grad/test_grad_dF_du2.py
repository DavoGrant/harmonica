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
np.random.seed(123)
u1 = 0.3
a_s = np.array([0.1, -0.003])
b_s = np.array([0.003])
cs = generate_complex_fourier_coeffs(a_s, b_s)
N_c = int((len(cs) - 1) / 2)
N_c_s = np.arange(-N_c, N_c + 1, 1)
d = 0.9
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
    for n, (_a, _b) in enumerate(zip(a_s[1:], b_s)):
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


def F_func(u2):
    # Limb darkening.
    us = np.array([1., u1, u2])
    B = np.array([[1., -1., -1.],
                  [0., 1., 2.],
                  [0., 0., -1.]])
    ps = np.matmul(B, us)
    I_0 = 1. / ((1. - us[1] / 3. - us[2] / 6.) * np.pi)

    # Find planet-stellar limb intersections.
    intersections, intersection_types = find_intersections(cs, d, nu)

    # Iterate pairs of thetas computing line integral.
    alpha = 0.
    for j in range(len(intersection_types)):
        if intersection_types[j] == 0:
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
        elif intersection_types[j] == 1:
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
            break

    return 1. - alpha


def s_vals(u2):
    # Find planet-stellar limb intersections.
    intersections, intersection_types = find_intersections(cs, d, nu)

    # Iterate pairs of thetas computing line integral.
    s0 = 0.
    s1 = 0.
    s2 = 0.
    for j in range(len(intersection_types)):
        if intersection_types[j] == 0:
            s0_j, _ = quad(
                line_integral_planet, intersections[j], intersections[j + 1],
                args=(cs, d, nu, 0),
                epsabs=1.e-15, epsrel=1.e-15, limit=500)
            s1_j, _ = quad(
                line_integral_planet, intersections[j], intersections[j + 1],
                args=(cs, d, nu, 1),
                epsabs=1.e-15, epsrel=1.e-15, limit=500)
            s2_j, _ = quad(
                line_integral_planet, intersections[j], intersections[j + 1],
                args=(cs, d, nu, 2),
                epsabs=1.e-15, epsrel=1.e-15, limit=500)
            s0 += s0_j
            s1 += s1_j
            s2 += s2_j
        elif intersection_types[j] == 1:
            phi_j = np.arctan2(
                -r_p(intersections[j]) * np.sin(intersections[j] - nu),
                -r_p(intersections[j]) * np.cos(
                    intersections[j] - nu) + d)
            phi_j_plus_1 = np.arctan2(
                -r_p(intersections[j + 1]) * np.sin(
                    intersections[j + 1] - nu),
                -r_p(intersections[j + 1]) * np.cos(
                    intersections[j + 1] - nu) + d)
            s0 += 1. / (0. + 2) * (phi_j_plus_1 - phi_j)
            s1 += 1. / (1. + 2) * (phi_j_plus_1 - phi_j)
            s2 += 1. / (2. + 2) * (phi_j_plus_1 - phi_j)
        else:
            break

    return s0, s1, s2


def dF_du2_total(u2):
    us = np.array([1., u1, u2])
    B = np.array([[1., -1., -1.],
                  [0., 1., 2.],
                  [0., 0., -1.]])
    ps = np.matmul(B, us)
    I_0 = 1. / (np.pi * (1. - us[1] / 3. - us[2] / 6.))
    s0, s1, s2 = s_vals(u1)

    dF_dalpha = -1.
    dalpha_dI0 = s0 * ps[0] + s1 * ps[1] + s2 * ps[2]
    dI0_du2 = 1. / (6. * np.pi * (1. - us[1] / 3. - us[2] / 6.)**2)
    dalpha_du2 = I_0 * (-s0 + 2 * s1 - s2)

    _dF_du2_total = dF_dalpha * dalpha_dI0 * dI0_du2 + dF_dalpha * dalpha_du2

    return _dF_du2_total


def grad_arrow(x_draw, x, y, grad):
    c = y - grad * x
    return grad * x_draw + c


while True:
    u2_a = np.random.uniform(0.01, 0.5)
    F_a = F_func(u2_a)

    delta = 1.e-6
    u2_b = u2_a + delta
    F_b = F_func(u2_b)

    u2_a_grad = dF_du2_total(u2_a)

    plt.scatter(u2_a, F_a, label='$u_2$')
    plt.scatter(u2_b, F_b, label='$u_2 + \delta$: $\delta={}$'.format(delta))
    x_arrow = np.linspace(u2_a, u2_b, 2)
    plt.plot(x_arrow, grad_arrow(x_arrow, u2_a, F_a, u2_a_grad),
             label='Gradient: $\\frac{dF}{d u_2}$')

    plt.legend()
    plt.xlabel('$u_2$')
    plt.ylabel('$F$')
    plt.tight_layout()
    plt.show()
