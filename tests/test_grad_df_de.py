import numpy as np
import matplotlib.pyplot as plt

from harmonica import bindings


def M0_func(e, w):
    alpha = ((1 - e) ** 0.5 * np.cos(w)) / ((1 + e) ** 0.5 * (1 + np.sin(w)))
    E0 = 2 * np.arctan(alpha)
    M0 = E0 - e * np.sin(E0)
    return M0


def M_func(e, w, p, t0, t):
    M0 = M0_func(e, w)
    n = 2 * np.pi / p
    tp = t0 - M0 / n
    M = (t - tp) * n
    return M


def sinf_func(e, w, p, t0, t):
    M = M_func(e, w, p, t0, t)
    sinf, cosf = bindings.test_solve_kepler(M, e)
    return sinf


def dE0_de_total(e, w):
    alpha = ((1 - e) ** 0.5 * np.cos(w)) / ((1 + e) ** 0.5 * (1 + np.sin(w)))

    dE0_dalpha_total = 2 / (alpha**2 + 1)
    dalpha_de_total = -np.cos(w) / ((1 + np.sin(w)) * (1 - e)**0.5 * (1 + e)**1.5)

    return dE0_dalpha_total * dalpha_de_total


def dM_de_total(e, w):
    alpha = ((1 - e) ** 0.5 * np.cos(w)) / ((1 + e) ** 0.5 * (1 + np.sin(w)))
    E0 = 2 * np.arctan(alpha)

    dM_dE0_partial = 1 - e * np.cos(E0)
    dM_de_partial = -np.sin(E0)
    de_de_total = 1.

    return dM_dE0_partial * dE0_de_total(e, w) + dM_de_partial * de_de_total


def df_de_total(e, w, p, t0, t):
    M = M_func(e, w, p, t0, t)
    sinf, cosf = bindings.test_solve_kepler(M, e)

    df_dM_partial = (1 + e * cosf)**2 / (1 - e**2)**1.5
    df_de_partial = ((2 + e * cosf) * sinf) / (1 - e**2)
    de_de_total = 1

    return df_dM_partial * dM_de_total(e, w) + df_de_partial * de_de_total


def dsinf_de_total(e, w, p, t0, t):
    M = M_func(e, w, p, t0, t)
    sinf, cosf = bindings.test_solve_kepler(M, e)

    dsinf_df_total = cosf

    return dsinf_df_total * df_de_total(e, w, p, t0, t)


def grad_arrow(x_draw, x, y, grad):
    c = y - grad * x
    return grad * x_draw + c


while True:
    ecc_a = np.random.uniform(0, 1)
    w_a = np.random.uniform(0, 2 * np.pi)
    p_a = np.random.uniform(0, 10)
    t0_a = np.random.uniform(0, 10)
    t_a = np.random.uniform(0, 10)
    sinf_a = sinf_func(ecc_a, w_a, p_a, t0_a, t_a)

    epsilon = 1e-6
    ecc_b = ecc_a + epsilon
    sinf_b = sinf_func(ecc_b, w_a, p_a, t0_a, t_a)

    sinf_a_grad = dsinf_de_total(ecc_a, w_a, p_a, t0_a, t_a)

    plt.scatter(ecc_a, sinf_a,
                label='$e_1$')
    plt.scatter(ecc_b, sinf_b,
                label='$e_1 + \\epsilon$: $\\epsilon={}$'.format(epsilon))
    x_arrow = np.linspace(ecc_a, ecc_b, 2)
    plt.plot(x_arrow, grad_arrow(x_arrow, ecc_a, sinf_a, sinf_a_grad),
             label='Gradient: $\\frac{d \sin{f}}{de}$')

    plt.legend()
    plt.xlabel('e')
    plt.ylabel('$\sin{f}$')
    plt.tight_layout()
    plt.show()
