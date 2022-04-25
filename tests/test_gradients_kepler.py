import numpy as np
import matplotlib.pyplot as plt

from harmonica import bindings


def kepler_func(M, ecc):
    sinf, cosf = bindings.test_solve_kepler(M, ecc)
    return sinf, cosf


def kepler_func_partial_derivatives(M, ecc, sinf, cosf):
    # df_dM = (1 + ecc * cosf)**2 / (1 - ecc**2)**1.5
    df_de = (2 + ecc * cosf) * sinf / (1 - ecc**2)
    return df_de * cosf, -df_de * sinf


def grad_arrow(x_draw, x, y, grad):
    c = y - grad * x
    return grad * x_draw + c


m0 = 0.3
ecc0 = 0.9
sinf0, cosf0 = kepler_func(m0, ecc0)

# m1 = m0 + 1e-6
ecc1 = ecc0 + 1e-6
sinf1, cosf1 = kepler_func(m0, ecc1)
print(sinf1 - sinf0)

plt.scatter(ecc0, cosf0)
plt.scatter(ecc1, cosf1)

x_arrow = np.linspace(ecc0, ecc1, 10)
sinf0_grad, cosf0_grad = kepler_func_partial_derivatives(m0, ecc0, sinf0, cosf0)
print(sinf0_grad)

plt.plot(x_arrow, grad_arrow(x_arrow, ecc0, cosf0, cosf0_grad))

plt.tight_layout()
plt.show()
