import numpy as np
import matplotlib.pyplot as plt


# Config.
np.random.seed(111)
c_0 = 0.9
w = 0.1


def theta_func(d):
    C = np.array(
        [[0., -np.exp(2 * 1j * w)],
         [1., np.exp(1j * w) / (c_0 * d) * (c_0**2 + d**2 - 1.)]])
    eig_vals, eig_vectors = np.linalg.eig(C)
    eig_idx = np.argmax(eig_vals)
    eig_val = eig_vals[eig_idx]
    eig_vector = eig_vectors[eig_idx]

    # return eig_val
    # return np.log(eig_val) * (0. - 1.j)
    return np.angle(eig_val)


def dtheta_dd_total(d):
    C = np.array(
        [[0., -np.exp(2 * 1j * w)],
         [1., np.exp(1j * w) / (c_0 * d) * (c_0**2 + d**2 - 1.)]])
    eig_vals, eig_vectors = np.linalg.eig(C)
    eig_idx = np.argmax(eig_vals)
    eig_val = eig_vals[eig_idx]
    eig_vector = eig_vectors[eig_idx]

    dC_dd_total = np.array(
        [[0., 0.],
         [0., np.exp(1j * w) * (d**2 - c_0**2 + 1) / (d**2 * c_0)]])
    dlambda_dd_total = np.diag(np.linalg.inv(eig_vectors) @ dC_dd_total @ eig_vectors)[eig_idx]

    # return dlambda_dd_total
    print(eig_val, np.abs(eig_val), dlambda_dd_total, -1j / eig_val * dlambda_dd_total)
    return np.real(-1j / eig_val * dlambda_dd_total)


def grad_arrow(x_draw, x, y, grad):
    c = y - grad * x
    return grad * x_draw + c


while True:
    d_a = np.random.uniform(0.85, 1.15)
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
