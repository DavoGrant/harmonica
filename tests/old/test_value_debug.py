import time
import numpy as np
import matplotlib.pyplot as plt


def planet_radius(a_ns, b_ns, theta):
    r_p = a_ns[0] * np.ones(theta.shape)
    for n, (a_n, b_n) in enumerate(zip(a_ns[1:], b_ns)):
        r_p += a_n * np.cos((n + 1) * theta)
        r_p += b_n * np.sin((n + 1) * theta)
    return r_p


def circle_radius(l, d, gamma):
    dcos = d * np.cos(gamma)
    sqrt_term = (dcos**2 - d**2 + l**2)**0.5
    return dcos + sqrt_term, dcos - sqrt_term


def a_p_values(c_ns, N, p, l, d, w):
    a_p = 0.
    if 0 <= p < N - 1:
        # NB. range + 1 to include upper range.
        for n in range(-N, -N + p + 1):
            # NB. map -N element to zero index etc.
            a_p += c_ns[n + N] * c_ns[(p - n - 2 * N) + N]
    elif N - 1 <= p < N + 1:
        a_p += -d * np.exp(1j * w) * c_ns[(p - 2 * N + 1) + N]
        for n in range(-N, -N + p + 1):
            a_p += c_ns[n + N] * c_ns[(p - n - 2 * N) + N]
    elif N + 1 <= p < 2 * N:
        a_p += -d * np.exp(1j * w) * c_ns[(p - 2 * N + 1) + N]
        a_p += -d * np.exp(-1j * w) * c_ns[(p - 2 * N - 1) + N]
        for n in range(-N, -N + p + 1):
            a_p += c_ns[n + N] * c_ns[(p - n - 2 * N) + N]
    elif p == 2 * N:
        a_p += -d * np.exp(1j * w) * c_ns[(p - 2 * N + 1) + N]
        a_p += -d * np.exp(-1j * w) * c_ns[(p - 2 * N - 1) + N]
        a_p += d ** 2 - l ** 2
        for n in range(-N, N + 1):
            a_p += c_ns[n + N] * c_ns[(p - n - 2 * N) + N]
    elif 2 * N + 1 <= p < 3 * N:
        a_p += -d * np.exp(1j * w) * c_ns[(p - 2 * N + 1) + N]
        a_p += -d * np.exp(-1j * w) * c_ns[(p - 2 * N - 1) + N]
        for n in range(-3 * N + p, N + 1):
            a_p += c_ns[n + N] * c_ns[(p - n - 2 * N) + N]
    elif 3 * N - 0 <= p < 3 * N + 2:
        a_p += -d * np.exp(-1j * w) * c_ns[(p - 2 * N - 1) + N]
        for n in range(-3 * N + p, N + 1):
            a_p += c_ns[n + N] * c_ns[(p - n - 2 * N) + N]
    elif 3 * N + 2 <= p < 4 * N + 1:
        for n in range(-3 * N + p, N + 1):
            a_p += c_ns[n + N] * c_ns[(p - n - 2 * N) + N]

    return a_p


def c_jk_values(c_ns, N, j, k, l, d, w):
    if k == 4 * N:
        return -a_p_values(c_ns, N, j - 1, l, d, w) / a_p_values(c_ns, N, 4 * N, l, d, w)
    else:
        if j == k + 1:
            return 1.
        else:
            return 0.


def companion_matrix_verbose(c_ns, l, d, w):
    # Detect scale.
    N = int((len(c_ns) - 1) / 2)
    assert N >= 1

    # Setup blank matrix.
    C = np.zeros((4 * N, 4 * N), dtype=np.cdouble)

    # Iterate elements.
    for (j, k), value in np.ndenumerate(C):
        # Fill elements.
        # NB. matrix values defined with index start at one.
        C[j, k] = c_jk_values(c_ns, N, j + 1, k + 1, l, d, w)

    return C


def generate_complex_fourier_coeffs(a_s, b_s):
    c_s = [a_s[0]]
    for a_n, b_n in zip(a_s[1:], b_s):
        c_s.append((a_n - 1j * b_n) / 2)
        c_s.insert(0, (a_n + 1j * b_n) / 2)
    return c_s


c_s = generate_complex_fourier_coeffs([0.1, 0.002, -0.003], [0.001, 0.004])
epsilon = 1e-7
c_matrix = companion_matrix_verbose(c_s, 1., 1.05, 0.01)
eigenvalues = np.linalg.eigvals(c_matrix)
theta_intercepts = [np.angle(r) for r in eigenvalues if 1. - epsilon < np.abs(r) < 1. + epsilon]
print(theta_intercepts)
