import numpy as np
import matplotlib.pyplot as plt


def M0_func(e, w):
    alpha = ((1 - e) ** 0.5 * np.cos(w)) / ((1 + e) ** 0.5 * (1 + np.sin(w)))
    E0 = 2 * np.arctan(alpha)
    M0 = E0 - e * np.sin(E0)
    return M0


def dM0_de(e, w):
    alpha = ((1 - e) ** 0.5 * np.cos(w)) / ((1 + e) ** 0.5 * (1 + np.sin(w)))
    E0 = 2 * np.arctan(alpha)

    dalpha_de = -np.cos(w) / ((1 + np.sin(w)) * (1 - e)**0.5 * (1 + e)**1.5)
    dE0_dalpha = 2 / (alpha**2 + 1)
    dE0_de = dE0_dalpha * dalpha_de

    dM0_de = dE0_de - np.sin(E0) - e * np.cos(E0) * dE0_de
    return dM0_de


def grad_arrow(x_draw, x, y, grad):
    c = y - grad * x
    return grad * x_draw + c


ecc0 = 0.04
w0 = 1.3
f0 = M0_func(ecc0, w0)

ecc1 = ecc0 + 1e-6
f1 = M0_func(ecc1, w0)

f0_grad = dM0_de(ecc0, w0)

plt.scatter(ecc0, f0)
plt.scatter(ecc1, f1)
x_arrow = np.linspace(ecc0, ecc1, 10)
plt.plot(x_arrow, grad_arrow(x_arrow, ecc0, f0, f0_grad))

plt.tight_layout()
plt.show()
