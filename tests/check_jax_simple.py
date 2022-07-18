import numpy as np
from functools import partial
from jax import jit, jvp, grad
from jax.tree_util import tree_map

from harmonica.jax import harmonica_transit


times = 1.
t0 = 1.01
period = 3.735
a = 7.025
inc = 86.9 * np.pi / 180.
ecc = 0.1
omega = 0.1
u1 = 0.1
u2 = 0.2
r0 = 0.1
r1 = 0.001
r2 = 0.001
r3 = 0.001
r4 = 0.001
args = [times, t0, period, a, inc, ecc, omega, 'quadratic', u1, u2, r0, r1, r2, r3, r4]

i = 13

analytic_gradient = grad(harmonica_transit, argnums=i)(*args)
value = harmonica_transit(*args)

epsilon = 1.e-7
args[i] += epsilon

perturbed_value = harmonica_transit(*args)

numerical_gradient = (perturbed_value - value) / epsilon
print(analytic_gradient - numerical_gradient)




import matplotlib.pyplot as plt
aa = np.logspace(-16, -1, 15)
dd = []
for eps in aa:
    analytic_gradient = grad(harmonica_transit, argnums=i)(*args)
    value = harmonica_transit(*args)

    args[i] += eps

    perturbed_value = harmonica_transit(*args)

    numerical_gradient = (perturbed_value - value) / eps
    dd.append(np.abs(analytic_gradient - numerical_gradient))

plt.loglog(aa, dd, '-o')
plt.show()
