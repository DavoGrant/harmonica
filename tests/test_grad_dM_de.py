import numpy as np
import matplotlib.pyplot as plt


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


def grad_arrow(x_draw, x, y, grad):
    c = y - grad * x
    return grad * x_draw + c


while True:
    ecc_a = np.random.uniform(0, 1)
    w_a = np.random.uniform(0, 2 * np.pi)
    p_a = np.random.uniform(0, 10)
    t0_a = np.random.uniform(0, 10)
    t_a = np.random.uniform(0, 10)
    M_a = M_func(ecc_a, w_a, p_a, t0_a, t_a)

    epsilon = 1e-6
    ecc_b = ecc_a + epsilon
    M_b = M_func(ecc_b, w_a, p_a, t0_a, t_a)

    M_a_grad = dM_de_total(ecc_a, w_a)

    plt.scatter(ecc_a, M_a, 
                label='$e_1$')
    plt.scatter(ecc_b, M_b, 
                label='$e_1 + \\epsilon$: $\\epsilon={}$'.format(epsilon))
    x_arrow = np.linspace(ecc_a, ecc_b, 10)
    plt.plot(x_arrow, grad_arrow(x_arrow, ecc_a, M_a, M_a_grad),
             label='Gradient: $\\frac{dM}{de}$')

    plt.legend()
    plt.xlabel('e')
    plt.xlabel('M')
    plt.tight_layout()
    plt.show()
