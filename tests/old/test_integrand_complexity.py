import numpy as np
import matplotlib.pyplot as plt


# Config.
n_val = 1.
a_s = [0.1]
b_s = []
c_s = [a_s[0]]
for a_n, b_n in zip(a_s[1:], b_s):
    c_s.append((a_n - 1j * b_n) / 2)
    c_s.insert(0, (a_n + 1j * b_n) / 2)
c_s = np.array(c_s)
N_c = int((len(c_s) - 1) / 2)
N_c_s = np.arange(-N_c, N_c + 1, 1)
d = 0.8999
nu = np.pi


def r_p(theta):
    _r_p = 0.
    for i, n, in enumerate(N_c_s):
        _r_p += c_s[i] * np.exp(1j * n * theta)
    return np.real(_r_p)


def dr_p_dt(theta):
    _r_p = 0.
    for i, n, in enumerate(N_c_s):
        _r_p += 1j * n * c_s[i] * np.exp(1j * n * theta)
    return np.real(_r_p)


def z_p(theta):
    return (1. - d**2 - r_p(theta)**2 + 2 * d * r_p(theta) * np.cos(theta - nu))**0.5


def zeta(_z_p):
    return (1. - _z_p**(n_val + 2)) / ((n_val + 2) * (1 - _z_p**2))


def eta(theta):
    return r_p(theta)**2 - d * np.cos(theta - nu) * r_p(theta) \
           - d * np.sin(theta - nu) * dr_p_dt(theta)


def line_integral(theta):
    return zeta(z_p(theta)) * eta(theta)


thetas = np.linspace(-np.pi, np.pi, 1000000)
plt.plot(thetas * 180. / np.pi, line_integral(thetas))
plt.ylabel('Integrand n=1')
plt.xlabel('$\\theta$')
plt.tight_layout()
plt.show()
