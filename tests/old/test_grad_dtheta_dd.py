import numpy as np
import matplotlib.pyplot as plt


def generate_complex_fourier_coeffs(_as, _bs):
    _cs = [_as[0]]
    for _a, _b in zip(_as[1:], _bs):
        _cs.append((_a - 1j * _b) / 2)
        _cs.insert(0, (_a + 1j * _b) / 2)
    return _cs


# Config.
a_s = np.array([0.1, -0.003, 0.003])
b_s = np.array([0.003, 0.003])
cs = generate_complex_fourier_coeffs(a_s, b_s)
N_c = int((len(cs) - 1) / 2)
N_c_s = np.arange(-N_c, N_c + 1, 1)
nu = 0.1
epsilon = 1.e-7


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


def theta_func(d):
    C = companion_matrix_verbose(cs, d, nu)
    eig_vals, eig_vectors = np.linalg.eig(C)

    n_real_intersections = 0
    intersections = []
    for eig in eig_vals:
        if 1. - epsilon < np.abs(eig) < 1. + epsilon:
            intersections.append(np.angle(eig))
            n_real_intersections += 1
        else:
            intersections.append(-np.inf)
    if n_real_intersections == 0:
        return 0.

    eig_idx = np.argmax(intersections)
    eig_val = eig_vals[eig_idx]
    eig_vector = eig_vectors[eig_idx]

    # return eig_val
    # return np.log(eig_val) * (0. - 1.j)
    return np.angle(eig_val)


def dtheta_dd_total(d):
    C = companion_matrix_verbose(cs, d, nu)
    eig_vals, eig_vectors = np.linalg.eig(C)

    n_real_intersections = 0
    intersections = []
    for eig in eig_vals:
        if 1. - epsilon < np.abs(eig) < 1. + epsilon:
            intersections.append(np.angle(eig))
            n_real_intersections += 1
        else:
            intersections.append(-np.inf)
    if n_real_intersections == 0:
        return 0.

    eig_idx = np.argmax(intersections)
    eig_val = eig_vals[eig_idx]
    eig_vector = eig_vectors[eig_idx]

    dC_dd_total = dcompanion_matrix_dd(cs, d, nu)
    dlambda_dd_total = np.diag(np.linalg.inv(eig_vectors) @ dC_dd_total @ eig_vectors)[eig_idx]

    print(-1j / eig_val * dlambda_dd_total)
    return np.real(-1j / eig_val * dlambda_dd_total)


def grad_arrow(x_draw, x, y, grad):
    c = y - grad * x
    return grad * x_draw + c


while True:
    d_a = np.random.uniform(0.8, 1.2)
    theta_a = theta_func(d_a)

    epsilon = 1.e-6
    d_b = d_a + epsilon
    theta_b = theta_func(d_b)

    theta_a_grad = dtheta_dd_total(d_a)

    plt.scatter(d_a, theta_a,
                label='$d$')
    plt.scatter(d_b, theta_b,
                label='$d + \\epsilon$: $\\epsilon={}$'.format(epsilon))
    x_arrow = np.linspace(d_a, d_b, 2)
    plt.plot(x_arrow, grad_arrow(
        x_arrow, d_a, theta_a, theta_a_grad),
        label='Gradient: $\\frac{d \\theta_1}{dd}$')

    plt.legend()
    plt.xlabel('d')
    plt.ylabel('$\\theta_1$')
    plt.tight_layout()
    plt.show()
