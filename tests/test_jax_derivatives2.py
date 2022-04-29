import jax
import numpy as np
from jax import grad
import matplotlib.pyplot as plt


def lambda_func(eig):
    return jax.numpy.log(eig) * (0. + 1.j)


def dlambda_dC_total(d):
    eig_grad = grad(lambda_func, holomorphic=True)

    return eig_grad(d)


def grad_arrow(x_draw, x, y, grad):
    c = y - grad * x
    return grad * x_draw + c


while True:
    d_a = 8. + 3.j
    lamb_a = lambda_func(d_a)

    epsilon = 3.e-6 + 1.j * 1.e-6
    d_b = d_a + epsilon
    lamb_b = lambda_func(d_b)

    lamb_a_grad = dlambda_dC_total(d_a)

    print(lamb_a)
    print(lamb_b)
    print(lamb_a_grad)

    plt.scatter(np.real(d_a), np.real(lamb_a),
                label='$e_1$')
    plt.scatter(np.real(d_b), np.real(lamb_b),
                label='$e_1 + \\epsilon$: $\\epsilon={}$'.format(epsilon))
    x_arrow = np.linspace(np.real(d_a), np.real(d_b), 2)
    plt.plot(x_arrow, grad_arrow(
        x_arrow, np.real(d_a), np.real(lamb_a), np.real(lamb_a_grad)),
        label='Gradient: $\\frac{d \sin{f}}{de}$')

    plt.legend()
    plt.xlabel('e')
    plt.ylabel('$\sin{f}$')
    plt.tight_layout()
    plt.show()

