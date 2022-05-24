from scipy.special import roots_legendre


N_l = 200
roots, weights = roots_legendre(N_l)
print('Roots.')
for i in range(N_l):
    print(roots[i])
print('Weights.')
for i in range(N_l):
    print(weights[i])
