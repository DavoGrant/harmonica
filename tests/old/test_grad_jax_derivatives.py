import numpy as np
from jax import grad
import jax.numpy as jnp
from jax.config import config
import matplotlib.pyplot as plt


# Config.
config.update("jax_enable_x64", True)
np.random.seed(111)
c_0 = 0.2
w = 0.5


def theta_func(d):
    C = jnp.array(
        [[0., -np.exp(2 * 1j * w)],
         [1., np.exp(1j * w) / (c_0 * d) * (c_0**2 + d**2 - 1.**2)]])
    eigs = jnp.linalg.eigvals(C)

    # return w - jax.numpy.arccos((c_0**2 + d**2 - 1**2) / (2 * c_0 * d))
    # return jax.numpy.log(eigs[0]) * (0. - 1.j)
    # NB. just for this test: select max eigenvalue to ensure we
    # get same eigenvalue between calls of this function, independent
    # of the index position jnp.linalg.eigvals orders.
    eig = jnp.max(eigs)
    # print(eig, jnp.abs(eig), jnp.angle(eig))
    return jnp.angle(eig)


def dtheta_dd_total(d):
    eig_grad = grad(theta_func, holomorphic=False)

    return eig_grad(d)


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
    print(theta_a_grad)

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

