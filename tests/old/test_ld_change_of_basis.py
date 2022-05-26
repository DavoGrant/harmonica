import numpy as np
import matplotlib.pyplot as plt


z = np.linspace(1, 0, 1000)

# Quadratic: original form.
u = np.array([1., 0.40, 0.29])
u_tilde = np.array([1., -(1 - z), -(1 - z)**2])

# Quadratic: polynomial form.
p_tilde = np.array([1., z, z**2])
B_1 = np.array([[1., -1., -1.],
                [0., 1., 2.],
                [0., 0., -1.]])

print(np.matmul(B_1, u))
print(np.max(u_tilde.dot(u) - p_tilde.dot(np.matmul(B_1, u))))
plt.plot(z, u_tilde.dot(u))
plt.plot(z, p_tilde.dot(np.matmul(B_1, u)))
plt.show()


# Non-linear: original form.
u = np.array([1., 0.2, 0.3, 0.4, 0.5])
u_tilde = np.array([1., -(1 - z**0.5), -(1 - z), -(1 - z**1.5), -(1 - z**2)])

# Non-linear: polynomial form.
p_tilde = np.array([1., z**0.5, z, z**1.5, z**2])
B_2 = np.array([[1., -1., -1., -1, -1],
                [0., 1., 0., 0., 0.],
                [0., 0., 1., 0., 0.],
                [0., 0., 0., 1., 0.],
                [0., 0., 0., 0., 1.]])

print(np.matmul(B_2, u))
print(np.max(u_tilde.dot(u) - p_tilde.dot(np.matmul(B_2, u))))
plt.plot(z, u_tilde.dot(u))
plt.plot(z, p_tilde.dot(np.matmul(B_2, u)))
plt.show()
