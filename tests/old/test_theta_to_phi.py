import numpy as np


d = 1.
nu = np.pi
r_p = 0.1
intersections = [-np.pi + 0.1, np.pi - 0.1]
intersections = [np.pi - 0.1, -np.pi + 0.1 + 2. * np.pi]

phi_j = np.arctan2(
    -r_p * np.sin(intersections[0] - nu),
    -r_p * np.cos(intersections[0] - nu) + d)
phi_j_plus_1 = np.arctan2(
    -r_p * np.sin(intersections[1] - nu),
    -r_p * np.cos(intersections[1] - nu) + d)

print(phi_j * 180 / np.pi)
print(phi_j_plus_1 * 180 / np.pi)
print(phi_j_plus_1 - phi_j)
print()

d = 1.
nu = 0.
r_p = 0.1
intersections = [-0.1, 0.1]

phi_j = np.arctan2(
    -r_p * np.sin(intersections[0] - nu),
    -r_p * np.cos(intersections[0] - nu) + d)
phi_j_plus_1 = np.arctan2(
    -r_p * np.sin(intersections[1] - nu),
    -r_p * np.cos(intersections[1] - nu) + d)

print(phi_j * 180 / np.pi)
print(phi_j_plus_1 * 180 / np.pi)
print(phi_j_plus_1 - phi_j)
print()

d = 0.8
nu = np.pi
r_p = 2.
intersections = [-np.pi + 0.1, np.pi - 0.1]

phi_j = np.arctan2(
    -r_p * np.sin(intersections[0] - nu),
    -r_p * np.cos(intersections[0] - nu) + d)
phi_j_plus_1 = np.arctan2(
    -r_p * np.sin(intersections[1] - nu),
    -r_p * np.cos(intersections[1] - nu) + d)

print(phi_j * 180 / np.pi)
print(phi_j_plus_1 * 180 / np.pi)
print(phi_j_plus_1 - phi_j)
print()

d = 0.8
nu = 0.
r_p = 2.
intersections = [-0.1, 0.1]
intersections = [0.1, -0.1 + 2 * np.pi]

phi_j = np.arctan2(
    -r_p * np.sin(intersections[0] - nu),
    -r_p * np.cos(intersections[0] - nu) + d)
phi_j_plus_1 = np.arctan2(
    -r_p * np.sin(intersections[1] - nu),
    -r_p * np.cos(intersections[1] - nu) + d)

print(phi_j * 180 / np.pi)
print(phi_j_plus_1 * 180 / np.pi)
print(phi_j_plus_1 - phi_j)
print()

d = 0.5
nu = np.pi
r_p = 5.
intersections = [-np.pi, np.pi - 1e-15]
print(np.pi)
print(np.pi - 1e-16)
phi_j = np.arctan2(
    -r_p * np.sin(intersections[0] - nu),
    -r_p * np.cos(intersections[0] - nu) + d)
phi_j_plus_1 = np.arctan2(
    -r_p * np.sin(intersections[1] - nu),
    -r_p * np.cos(intersections[1] - nu) + d)

print(phi_j * 180 / np.pi)
print(phi_j_plus_1 * 180 / np.pi)
print(phi_j_plus_1 - phi_j)
print()
