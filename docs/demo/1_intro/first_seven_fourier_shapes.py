import numpy as np
import matplotlib.pyplot as plt


theta = np.linspace(-np.pi, np.pi, 10000)
colours = ['#FFD200', '#3cb557', '#007c82', '#003f5c']

fig = plt.figure(figsize=(8, 8))
ax1 = plt.subplot(1, 1, 1)
ax1.set_aspect('equal', 'box')
for i, c in zip(np.linspace(0.05, 0.15, 4), colours):
    transmission_string = i
    ax1.plot(transmission_string * np.cos(theta),
             transmission_string * np.sin(theta),
             lw=2.5, c=c, alpha=0.8)
    ax1.plot(0.1 * np.cos(theta),
             0.1 * np.sin(theta),
             lw=2.5, c='#000000', ls='--')
plt.axis('off')
ax1.set_xlim(-0.175, 0.175)
ax1.set_ylim(-0.175, 0.175)
plt.savefig('/Users/davidgrant/Desktop/fourier_n0.png', transparent=True)
# plt.show()

fig = plt.figure(figsize=(8, 8))
ax1 = plt.subplot(1, 1, 1)
ax1.set_aspect('equal', 'box')
for i, c in zip(np.linspace(-0.05, 0.05, 4), colours):
    transmission_string = 0.1 + i * np.cos(theta)
    ax1.plot(transmission_string * np.cos(theta),
             transmission_string * np.sin(theta),
             lw=2.5, c=c, alpha=0.8)
    ax1.plot(0.1 * np.cos(theta),
             0.1 * np.sin(theta),
             lw=2.5, c='#000000', ls='--')
plt.axis('off')
ax1.set_xlim(-0.175, 0.175)
ax1.set_ylim(-0.175, 0.175)
plt.savefig('/Users/davidgrant/Desktop/fourier_n1_a.png', transparent=True)
# plt.show()

fig = plt.figure(figsize=(8, 8))
ax1 = plt.subplot(1, 1, 1)
ax1.set_aspect('equal', 'box')
for i, c in zip(np.linspace(-0.05, 0.05, 4), colours):
    transmission_string = 0.1 + i * np.sin(theta)
    ax1.plot(transmission_string * np.cos(theta),
             transmission_string * np.sin(theta),
             lw=2.5, c=c, alpha=0.8)
    ax1.plot(0.1 * np.cos(theta),
             0.1 * np.sin(theta),
             lw=2.5, c='#000000', ls='--')
plt.axis('off')
ax1.set_xlim(-0.175, 0.175)
ax1.set_ylim(-0.175, 0.175)
plt.savefig('/Users/davidgrant/Desktop/fourier_n1_b.png', transparent=True)
# plt.show()

fig = plt.figure(figsize=(8, 8))
ax1 = plt.subplot(1, 1, 1)
ax1.set_aspect('equal', 'box')
for i, c in zip(np.linspace(-0.05, 0.05, 4), colours):
    transmission_string = 0.1 + i * np.cos(2 * theta)
    ax1.plot(transmission_string * np.cos(theta),
             transmission_string * np.sin(theta),
             lw=2.5, c=c, alpha=0.8)
    ax1.plot(0.1 * np.cos(theta),
             0.1 * np.sin(theta),
             lw=2.5, c='#000000', ls='--')
plt.axis('off')
ax1.set_xlim(-0.175, 0.175)
ax1.set_ylim(-0.175, 0.175)
plt.savefig('/Users/davidgrant/Desktop/fourier_n2_a.png', transparent=True)
# plt.show()

fig = plt.figure(figsize=(8, 8))
ax1 = plt.subplot(1, 1, 1)
ax1.set_aspect('equal', 'box')
for i, c in zip(np.linspace(-0.05, 0.05, 4), colours):
    transmission_string = 0.1 + i * np.sin(2 * theta)
    ax1.plot(transmission_string * np.cos(theta),
             transmission_string * np.sin(theta),
             lw=2.5, c=c, alpha=0.8)
    ax1.plot(0.1 * np.cos(theta),
             0.1 * np.sin(theta),
             lw=2.5, c='#000000', ls='--')
plt.axis('off')
ax1.set_xlim(-0.175, 0.175)
ax1.set_ylim(-0.175, 0.175)
plt.savefig('/Users/davidgrant/Desktop/fourier_n2_b.png', transparent=True)
# plt.show()

fig = plt.figure(figsize=(8, 8))
ax1 = plt.subplot(1, 1, 1)
ax1.set_aspect('equal', 'box')
for i, c in zip(np.linspace(-0.05, 0.05, 4), colours):
    transmission_string = 0.1 + i * np.cos(3 * theta)
    ax1.plot(transmission_string * np.cos(theta),
             transmission_string * np.sin(theta),
             lw=2.5, c=c, alpha=0.8)
    ax1.plot(0.1 * np.cos(theta),
             0.1 * np.sin(theta),
             lw=2.5, c='#000000', ls='--')
plt.axis('off')
ax1.set_xlim(-0.175, 0.175)
ax1.set_ylim(-0.175, 0.175)
plt.savefig('/Users/davidgrant/Desktop/fourier_n3_a.png', transparent=True)
# plt.show()

fig = plt.figure(figsize=(8, 8))
ax1 = plt.subplot(1, 1, 1)
ax1.set_aspect('equal', 'box')
for i, c in zip(np.linspace(-0.05, 0.05, 4), colours):
    transmission_string = 0.1 + i * np.sin(3 * theta)
    ax1.plot(transmission_string * np.cos(theta),
             transmission_string * np.sin(theta),
             lw=2.5, c=c, alpha=0.8)
    ax1.plot(0.1 * np.cos(theta),
             0.1 * np.sin(theta),
             lw=2.5, c='#000000', ls='--')
plt.axis('off')
ax1.set_xlim(-0.175, 0.175)
ax1.set_ylim(-0.175, 0.175)
plt.savefig('/Users/davidgrant/Desktop/fourier_n3_b.png', transparent=True)
# plt.show()
