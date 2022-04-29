import jax
import numpy as np
from jax import grad
from jax.config import config
import matplotlib.pyplot as plt


# Config.
config.update("jax_enable_x64", True)
np.random.seed(111)


def lambda_func(d):
    c_0 = 0.2
    w = 0.1
    C = jax.numpy.array(
        [[0., -np.exp(2 * 1j * w)],
         [1., np.exp(1j * w) / (c_0 * d) * (c_0**2 + d**2 - 1.**2)]])
    eigs = jax.numpy.linalg.eigvals(C)

    # return w - jax.numpy.arccos((c_0**2 + d**2 - 1**2) / (2 * c_0 * d))
    # return jax.numpy.real(jax.numpy.log(eigs[0]) * (0. + 1.j))
    return jax.numpy.angle(eigs[0])


def dlambda_dC_total(d):
    eig_grad = grad(lambda_func, holomorphic=False)

    return jax.numpy.real(eig_grad(d))


def grad_arrow(x_draw, x, y, grad):
    c = y - grad * x
    return grad * x_draw + c


while True:
    # NB. may need to newton polish roots to double
    # precision for good gradients.
    d_a = np.random.uniform(0.85, 1.15)
    lamb_a = lambda_func(d_a)

    epsilon = 1.e-6
    d_b = d_a + epsilon
    lamb_b = lambda_func(d_b)

    lamb_a_grad = dlambda_dC_total(d_a)
    print(lamb_a_grad)

    plt.scatter(d_a, lamb_a,
                label='$e_1$')
    plt.scatter(d_b, lamb_b,
                label='$e_1 + \\epsilon$: $\\epsilon={}$'.format(epsilon))
    x_arrow = np.linspace(d_a, d_b, 2)
    plt.plot(x_arrow, grad_arrow(
        x_arrow, d_a, lamb_a, lamb_a_grad),
        label='Gradient: $\\frac{d \sin{f}}{de}$')

    plt.legend()
    plt.xlabel('e')
    plt.ylabel('$\sin{f}$')
    plt.tight_layout()
    plt.show()

