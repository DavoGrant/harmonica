import numpy as np
import matplotlib.pyplot as plt


def lambda_func(C):
    lambs, us = np.linalg.eig(C)
    return lambs[0]


def dlambda_dC_total(C):
    lambs, us = np.linalg.eig(C)

    dC_dd = np.array([[0., 1.],
                      [0., 0.]])

    print(np.matmul(us.conj().T, np.matmul(dC_dd, us)))
    return np.matmul(us.conj().T, np.matmul(dC_dd, us))[0]


def grad_arrow(x_draw, x, y, grad):
    c = y - grad * x
    return grad * x_draw + c


while True:
    d_a = 8.
    C_a = np.array([[1., -3. + d_a],
                    [-6. + 1.j, 9.]])
    lamb_a = lambda_func(C_a)

    epsilon = 1e-6
    d_b = d_a + epsilon
    C_b = np.array([[1., -3. + d_b],
                    [-6. + 1.j, 9.]])
    lamb_b = lambda_func(C_b)

    lamb_a_grad = dlambda_dC_total(C_a)

    plt.scatter(d_a, np.real(lamb_a),
                label='$e_1$')
    plt.scatter(d_b, np.real(lamb_b),
                label='$e_1 + \\epsilon$: $\\epsilon={}$'.format(epsilon))
    x_arrow = np.linspace(d_a, d_b, 2)
    plt.plot(x_arrow, grad_arrow(x_arrow, d_a, np.real(lamb_a), np.real(lamb_a_grad)),
             label='Gradient: $\\frac{d \sin{f}}{de}$')

    plt.legend()
    plt.xlabel('e')
    plt.ylabel('$\sin{f}$')
    plt.tight_layout()
    plt.show()
