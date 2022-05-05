import numpy as np
import matplotlib.pyplot as plt

from harmonica import bindings


def d_func(i, a, p, t0, t):
    n = 2 * np.pi / p
    M = n * (t - t0) + np.pi / 2
    x = a * np.cos(M)
    y = a * np.sin(M) * np.cos(i)
    d = (x*x + y*y)**0.5
    return d


def dd_dp_total(i, a, p, t0, t):
    n = 2 * np.pi / p
    M = n * (t - t0) + np.pi / 2
    x = a * np.cos(M)
    y = a * np.sin(M) * np.cos(i)
    d = (x*x + y*y)**0.5

    dd_dx_partial = x / d
    dx_dM_partial = -a * np.sin(M)
    dM_dp_partial = (t - t0) * (-2 * np.pi / p**2)

    dd_dy_partial = y / d
    dy_dM_partial = a * np.cos(M) * np.cos(i)

    return dd_dx_partial * dx_dM_partial * dM_dp_partial \
           + dd_dy_partial * dy_dM_partial * dM_dp_partial


def grad_arrow(x_draw, x, y, grad):
    c = y - grad * x
    return grad * x_draw + c


while True:
    i_a = np.random.uniform(0, np.pi / 2)
    a_a = np.random.uniform(0, 10)
    p_a = np.random.uniform(0, 10)
    t0_a = np.random.uniform(0, 10)
    t_a = np.random.uniform(0, 10)
    d_a = d_func(i_a, a_a, p_a, t0_a, t_a)

    epsilon = 1e-6
    p_b = p_a + epsilon
    d_b = d_func(i_a, a_a, p_b, t0_a, t_a)

    d_a_grad = dd_dp_total(i_a, a_a, p_a, t0_a, t_a)

    plt.scatter(p_a, d_a,
                label='$p_1$')
    plt.scatter(p_b, d_b,
                label='$p_1 + \\epsilon$: $\\epsilon={}$'.format(epsilon))
    x_arrow = np.linspace(p_a, p_b, 2)
    plt.plot(x_arrow, grad_arrow(x_arrow, p_a, d_a, d_a_grad),
             label='Gradient: $\\frac{dd}{dp}$')

    plt.legend()
    plt.xlabel('p')
    plt.ylabel('$d$')
    plt.tight_layout()
    plt.show()
