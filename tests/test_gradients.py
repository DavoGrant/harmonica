import numpy as np
import matplotlib.pyplot as plt


def simple_chain_func(x, y):
    f = x**2 + 3 * y
    return np.sin(f)


def simple_chain_func_derivative(x, y):
    f = x ** 2 + 3 * y
    df_dx = 2 * x
    return np.cos(f) * df_dx


def grad_arrow(x_draw, x, y, grad):
    c = y - grad * x
    return grad * x_draw + c


xs = 2.
ys = 9.
xs_epsilon = xs + 1e-6
plt.scatter(xs, simple_chain_func(xs, ys))
plt.scatter(xs_epsilon, simple_chain_func(xs_epsilon, ys))
x_arrow = np.linspace(xs, xs_epsilon, 10)
gradient = simple_chain_func_derivative(xs, ys)
plt.plot(x_arrow, grad_arrow(x_arrow, xs, simple_chain_func(xs, ys), gradient))

plt.tight_layout()
plt.show()
