import numpy as np
import matplotlib.pyplot as plt
from harmonica import HarmonicaTransit


n_checkpoints = 20
n_frames_per_checkpoint = 20
ht = HarmonicaTransit()
theta = np.linspace(-np.pi, np.pi, 10000)

a_0_checkpoints = np.random.uniform(0.07, 0.13, n_checkpoints)
a_1_checkpoints = np.random.uniform(-0.005, 0.005, n_checkpoints)
b_1_checkpoints = np.random.uniform(-0.005, 0.005, n_checkpoints)
a_2_checkpoints = np.random.uniform(-0.009, 0.009, n_checkpoints)
b_2_checkpoints = np.random.uniform(-0.009, 0.009, n_checkpoints)

frame_num = 0
for i in range(n_checkpoints):
    if i == n_checkpoints - 1:
        a_0s = np.linspace(a_0_checkpoints[i], a_0_checkpoints[0], n_frames_per_checkpoint)
        a_1s = np.linspace(a_1_checkpoints[i], a_1_checkpoints[0], n_frames_per_checkpoint)
        b_1s = np.linspace(b_1_checkpoints[i], b_1_checkpoints[0], n_frames_per_checkpoint)
        a_2s = np.linspace(a_2_checkpoints[i], a_2_checkpoints[0], n_frames_per_checkpoint)
        b_2s = np.linspace(b_2_checkpoints[i], b_2_checkpoints[0], n_frames_per_checkpoint)
    else:
        a_0s = np.linspace(a_0_checkpoints[i], a_0_checkpoints[i + 1], n_frames_per_checkpoint)
        a_1s = np.linspace(a_1_checkpoints[i], a_1_checkpoints[i + 1], n_frames_per_checkpoint)
        b_1s = np.linspace(b_1_checkpoints[i], b_1_checkpoints[i + 1], n_frames_per_checkpoint)
        a_2s = np.linspace(a_2_checkpoints[i], a_2_checkpoints[i + 1], n_frames_per_checkpoint)
        b_2s = np.linspace(b_2_checkpoints[i], b_2_checkpoints[i + 1], n_frames_per_checkpoint)
    for a_0, a_1, b_1, a_2, b_2 in zip(a_0s, a_1s, b_1s, a_2s, b_2s):
        fig = plt.figure(figsize=(8, 8))
        ax1 = plt.subplot(1, 1, 1)
        # ax2 = plt.subplot(1, 3, 2)
        # ax3 = plt.subplot(1, 3, 3)
        ax1.set_aspect('equal', 'box')
        # ax2.set_aspect('equal', 'box')
        # ax3.set_aspect('equal', 'box')

        # ht.set_planet_transmission_string(np.array([a_0]))
        # transmission_string = ht.get_planet_transmission_string(theta)
        # ax1.plot(transmission_string * np.cos(theta),
        #          transmission_string * np.sin(theta),
        #          c='#FFD200', alpha=1.0)
        #
        # ht.set_planet_transmission_string(np.array([0.1, a_1, b_1]))
        # transmission_string = ht.get_planet_transmission_string(theta)
        # ax2.plot(transmission_string * np.cos(theta),
        #          transmission_string * np.sin(theta),
        #          c='#FFD200', alpha=1.0)

        ht.set_planet_transmission_string(np.array([a_0, a_1, b_1, 0, 0, 0, 0, 0, 0, a_2, b_2]))
        transmission_string = ht.get_planet_transmission_string(theta)
        ax1.plot(transmission_string * np.cos(theta),
                 transmission_string * np.sin(theta),
                 c='#FFD200', alpha=1.0)

        ax1.axis('off')
        # ax2.axis('off')
        # ax3.axis('off')
        ax1.set_xlim(-0.15, 0.15)
        ax1.set_ylim(-0.15, 0.15)
        # ax2.set_xlim(-0.15, 0.15)
        # ax2.set_ylim(-0.15, 0.15)
        # ax3.set_xlim(-0.15, 0.15)
        # ax3.set_ylim(-0.15, 0.15)
        plt.tight_layout()
        plt.savefig('/Users/davidgrant/Desktop/string_animation/{}.jpg'.format(
            frame_num), transparent=True)
        # plt.show()

        frame_num += 1
