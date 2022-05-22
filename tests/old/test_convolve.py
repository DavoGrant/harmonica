import numpy as np


def generate_complex_fourier_coeffs(a_s, b_s):
    c_s = [a_s[0]]
    for a_n, b_n in zip(a_s[1:], b_s):
        c_s.append((a_n - 1j * b_n) / 2)
        c_s.insert(0, (a_n + 1j * b_n) / 2)
    return c_s


ccc = generate_complex_fourier_coeffs([0.1, 0.002, -0.003], [0.001, 0.004])
conv = np.convolve(ccc, ccc)
for val in conv:
    print(val)
