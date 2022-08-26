import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d


def convolve_and_rebin(_pixel_wvs, _model_wvs, _model_signal, psf=0.):
    # Rebin to pixels.
    model_wvs_rebin = []
    model_signal_rebin = []
    for pix_wv_s, pix_wv_m, pix_wv_e in zip(
            _pixel_wvs, _pixel_wvs[1:], _pixel_wvs[2:]):

        # Define pixel wavelengths.
        pix_wv_start = pix_wv_m - (pix_wv_m - pix_wv_s) / 2.
        pix_wv_end = pix_wv_m + (pix_wv_e - pix_wv_m) / 2.

        model_bin_mask = (pix_wv_start <= _model_wvs) & (_model_wvs < pix_wv_end)
        model_wvs_rebin.append(pix_wv_m)
        model_signal_rebin.append(np.mean(_model_signal[model_bin_mask]))

    model_wvs_rebin = np.array(model_wvs_rebin)
    model_signal_rebin = np.array(model_signal_rebin)

    if psf > 0.:
        # Convolve with psf.
        model_signal_rebin = gaussian_filter1d(model_signal_rebin, psf)

    return model_wvs_rebin, model_signal_rebin


# Config.
n_bins_model = 400
bin_w = 5
a = 2.6
b = 3.3
c = 4.0
d = 4.7
exaggeration_factor = 50
colours = ['#FFD200', '#3cb557', '#007c82', '#003f5c']


fortney_750 = pd.read_csv('fortney_750.txt', delimiter=' ', header=None).values
fortney_1000 = pd.read_csv('fortney_1000.txt', delimiter=' ', header=None).values
fortney_1250 = pd.read_csv('fortney_1250.txt', delimiter=' ', header=None).values

wvs = np.linspace(2.3, 5.2, n_bins_model)
wvs_fortney_750, tds_fortney_750 = convolve_and_rebin(
    wvs, fortney_750[:, 0], fortney_750[:, 1] * 100., psf=0.)
wvs_fortney_1000, tds_fortney_1000 = convolve_and_rebin(
    wvs, fortney_1000[:, 0], fortney_1000[:, 1] * 100., psf=0.)
wvs_fortney_1250, tds_fortney_1250 = convolve_and_rebin(
    wvs, fortney_1250[:, 0], fortney_1250[:, 1] * 100., psf=0.)

# 2.6 um.
idx = (np.abs(wvs_fortney_750 - a)).argmin()
wv_750_a = wvs_fortney_750[idx]
td_750_a = np.mean(tds_fortney_750[idx - bin_w: idx + bin_w + 1])
print('wv={} td_750k={}'.format(wv_750_a, td_750_a))

idx = (np.abs(wvs_fortney_1000 - a)).argmin()
wv_1000_a = wvs_fortney_1000[idx]
td_1000_a = np.mean(tds_fortney_1000[idx - bin_w: idx + bin_w + 1])
print('wv={} td_1000k={}'.format(wv_1000_a, td_1000_a))

idx = (np.abs(wvs_fortney_1250 - a)).argmin()
wv_1250_a = wvs_fortney_1250[idx]
td_1250_a = np.mean(tds_fortney_1250[idx - bin_w: idx + bin_w + 1])
print('wv={} td_1250k={}'.format(wv_1250_a, td_1250_a))

# 3.3 um.
idx = (np.abs(wvs_fortney_750 - b)).argmin()
wv_750_b = wvs_fortney_750[idx]
td_750_b = np.mean(tds_fortney_750[idx - bin_w: idx + bin_w + 1])
print('wv={} td_750k={}'.format(wv_750_b, td_750_b))

idx = (np.abs(wvs_fortney_1000 - b)).argmin()
wv_1000_b = wvs_fortney_1000[idx]
td_1000_b = np.mean(tds_fortney_1000[idx - bin_w: idx + bin_w + 1])
print('wv={} td_1000k={}'.format(wv_1000_b, td_1000_b))

idx = (np.abs(wvs_fortney_1250 - b)).argmin()
wv_1250_b = wvs_fortney_1250[idx]
td_1250_b = np.mean(tds_fortney_1250[idx - bin_w: idx + bin_w + 1])
print('wv={} td_1250k={}'.format(wv_1250_b, td_1250_b))

# 4.0 um.
idx = (np.abs(wvs_fortney_750 - c)).argmin()
wv_750_c = wvs_fortney_750[idx]
td_750_c = np.mean(tds_fortney_750[idx - bin_w: idx + bin_w + 1])
print('wv={} td_750k={}'.format(wv_750_c, td_750_c))

idx = (np.abs(wvs_fortney_1000 - c)).argmin()
wv_1000_c = wvs_fortney_1000[idx]
td_1000_c = np.mean(tds_fortney_1000[idx - bin_w: idx + bin_w + 1])
print('wv={} td_1000k={}'.format(wv_1000_c, td_1000_c))

idx = (np.abs(wvs_fortney_1250 - c)).argmin()
wv_1250_c = wvs_fortney_1250[idx]
td_1250_c = np.mean(tds_fortney_1250[idx - bin_w: idx + bin_w + 1])
print('wv={} td_1250k={}'.format(wv_1250_c, td_1250_c))

# 4.7 um.
idx = (np.abs(wvs_fortney_750 - d)).argmin()
wv_750_d = wvs_fortney_750[idx]
td_750_d = np.mean(tds_fortney_750[idx - bin_w: idx + bin_w + 1])
print('wv={} td_750k={}'.format(wv_750_d, td_750_d))

idx = (np.abs(wvs_fortney_1000 - d)).argmin()
wv_1000_d = wvs_fortney_1000[idx]
td_1000_d = np.mean(tds_fortney_1000[idx - bin_w: idx + bin_w + 1])
print('wv={} td_1000k={}'.format(wv_1000_d, td_1000_d))

idx = (np.abs(wvs_fortney_1250 - d)).argmin()
wv_1250_d = wvs_fortney_1250[idx]
td_1250_d = np.mean(tds_fortney_1250[idx - bin_w: idx + bin_w + 1])
print('wv={} td_1250k={}'.format(wv_1250_d, td_1250_d))


fig, ax1 = plt.subplots(1, 1, figsize=(8, 6.5))
ax1.plot(wvs_fortney_750, tds_fortney_750,
         alpha=0.25, color='#000000', linewidth=1.2, zorder=1)
ax1.scatter([wv_750_a, wv_750_b, wv_750_c, wv_750_d],
            [td_750_a, td_750_b, td_750_c, td_750_d],
            color=np.flip(colours), s=230, zorder=2, edgecolors='#000000')
ax1.set_xlabel('$\lambda$ / $\mu m$', fontsize=28)
ax1.set_ylabel('Transit depth ($\\theta=0$) / %', fontsize=28, labelpad=14)
ax1.set_ylim(2.58, 2.67)
ax1.set_xticks([3.0, 4.0, 5.0])
ax1.set_yticks([2.59, 2.62, 2.65])
ax1.tick_params(axis='both', which='major', labelsize=23)
ax1.spines['left'].set_linewidth(1.5)
ax1.spines['bottom'].set_linewidth(1.5)
plt.tight_layout()
plt.savefig('/Users/davidgrant/Desktop/Harmonica/fortney_spectrum_theta_0.png', transparent=True)
plt.show()

fig, ax1 = plt.subplots(1, 1, figsize=(8, 6.5))
ax1.plot(wvs_fortney_1000, tds_fortney_1000,
         alpha=0.25, color='#000000', linewidth=1.2, zorder=1)
ax1.scatter([wv_1000_a, wv_1000_b, wv_1000_c, wv_1000_d],
            [td_1000_a, td_1000_b, td_1000_c, td_1000_d],
            color=np.flip(colours), s=230, zorder=2, edgecolors='#000000')
ax1.set_xlabel('$\lambda$ / $\mu m$', fontsize=28)
ax1.set_ylabel('Transit depth ($\\theta=\pi/2$) / %', fontsize=28, labelpad=14)
ax1.set_ylim(2.58, 2.67)
ax1.set_xticks([3.0, 4.0, 5.0])
ax1.set_yticks([2.59, 2.62, 2.65])
ax1.tick_params(axis='both', which='major', labelsize=23)
ax1.spines['left'].set_linewidth(1.5)
ax1.spines['bottom'].set_linewidth(1.5)
plt.tight_layout()
plt.savefig('/Users/davidgrant/Desktop/Harmonica/fortney_spectrum_theta_90.png', transparent=True)
plt.show()

fig, ax1 = plt.subplots(1, 1, figsize=(8, 6.5))
ax1.plot(wvs_fortney_1000, tds_fortney_1000,
         alpha=0.25, color='#000000', linewidth=1.2, zorder=1)
ax1.scatter([wv_1000_a, wv_1000_b, wv_1000_c, wv_1000_d],
            [td_1000_a, td_1000_b, td_1000_c, td_1000_d],
            color=np.flip(colours), s=230, zorder=2, edgecolors='#000000')
ax1.set_xlabel('$\lambda$ / $\mu m$', fontsize=28)
ax1.set_ylabel('Transit depth ($\\theta=-\pi/2$) / %', fontsize=28, labelpad=14)
ax1.set_ylim(2.58, 2.67)
ax1.set_xticks([3.0, 4.0, 5.0])
ax1.set_yticks([2.59, 2.62, 2.65])
ax1.tick_params(axis='both', which='major', labelsize=23)
ax1.spines['left'].set_linewidth(1.5)
ax1.spines['bottom'].set_linewidth(1.5)
plt.tight_layout()
plt.savefig('/Users/davidgrant/Desktop/Harmonica/fortney_spectrum_theta_-90.png', transparent=True)
plt.show()

fig, ax1 = plt.subplots(1, 1, figsize=(8, 6.5))
ax1.plot(wvs_fortney_1250, tds_fortney_1250,
         alpha=0.25, color='#000000', linewidth=1.2, zorder=1)
ax1.scatter([wv_1250_a, wv_1250_b, wv_1250_c, wv_1250_d],
            [td_1250_a, td_1250_b, td_1250_c, td_1250_d],
            color=np.flip(colours), s=230, zorder=2, edgecolors='#000000')
ax1.set_xlabel('$\lambda$ / $\mu m$', fontsize=28)
ax1.set_ylabel('Transit depth ($\\theta=2\pi$) / %', fontsize=28, labelpad=14)
ax1.set_ylim(2.58, 2.67)
ax1.set_xticks([3.0, 4.0, 5.0])
ax1.set_yticks([2.59, 2.62, 2.65])
ax1.tick_params(axis='both', which='major', labelsize=23)
ax1.spines['left'].set_linewidth(1.5)
ax1.spines['bottom'].set_linewidth(1.5)
plt.tight_layout()
plt.savefig('/Users/davidgrant/Desktop/Harmonica/fortney_spectrum_theta_180.png', transparent=True)
plt.show()


def radius_RM(r_evening, r_morning, alpha=0., gamma=0., sigma=1.):
    _r_p = []
    _thetas = np.linspace(-np.pi / 2, np.pi / 2, 1000)
    for _theta in _thetas:
        _theta -= gamma
        r_term_bar = 0.5 * (r_evening + r_morning)
        r_term_delta = r_evening - r_morning
        if _theta <= -alpha/2.:
            _r_p.append(r_evening)
        elif -alpha/2. < _theta < alpha/2.:
            _r_p.append(r_term_bar - _theta / (alpha/2.) * r_term_delta/2.)
        elif _theta >= alpha/2.:
            _r_p.append(r_morning)

    _thetas = np.concatenate([_thetas, _thetas + np.pi])
    _thetas = _thetas - np.pi / 2  # Ryan defined north as zero angle.
    _r_p = np.concatenate([_r_p, np.flip(_r_p)])

    _r_p = gaussian_filter1d(_r_p, sigma=sigma)

    return _thetas, _r_p


fig, ax1 = plt.subplots(1, 1, figsize=(8, 8))
ax1.set_aspect('equal', 'box')
plt.scatter([0], [0], s=1, c='#000000')

# 2.7um.
mid_rp = (td_1000_a / 100.)**0.5
cold_rp = (td_750_a / 100.)**0.5
cold_rp = mid_rp + (cold_rp - mid_rp) * exaggeration_factor
hot_rp = (td_1250_a / 100.)**0.5
hot_rp = mid_rp + (hot_rp - mid_rp) * exaggeration_factor
mean_rp = np.mean([hot_rp, cold_rp])
thetas, r_p = radius_RM(hot_rp, cold_rp,
                        alpha=0.5, gamma=0., sigma=50.)
idx = (np.abs(thetas - 0.)).argmin()
print('Morning (east) rp', np.round(r_p[idx], 4))
idx = (np.abs(thetas - np.pi / 2)).argmin()
print('Evening (west) rp', np.round(r_p[idx], 4))
ax1.plot(r_p * np.cos(thetas), r_p * np.sin(thetas),
         lw=2.5, c=colours[3], alpha=0.8)

# 3.3um.
cold_rp = (td_750_b / 100.)**0.5
cold_rp = mid_rp + (cold_rp - mid_rp) * exaggeration_factor
hot_rp = (td_1250_b / 100.)**0.5
hot_rp = mid_rp + (hot_rp - mid_rp) * exaggeration_factor
mean_rp = np.mean([hot_rp, cold_rp])
thetas, r_p = radius_RM(hot_rp, cold_rp,
                        alpha=0.5, gamma=0., sigma=50.)
idx = (np.abs(thetas - 0.)).argmin()
print('Morning (east) rp', np.round(r_p[idx], 4))
idx = (np.abs(thetas - np.pi / 2)).argmin()
print('Evening (west) rp', np.round(r_p[idx], 4))
ax1.plot(r_p * np.cos(thetas), r_p * np.sin(thetas),
         lw=2.5, c=colours[2], alpha=0.8)

# 4.0um.
cold_rp = (td_750_c / 100.)**0.5
cold_rp = mid_rp + (cold_rp - mid_rp) * exaggeration_factor
hot_rp = (td_1250_c / 100.)**0.5
hot_rp = mid_rp + (hot_rp - mid_rp) * exaggeration_factor
mean_rp = np.mean([hot_rp, cold_rp])
thetas, r_p = radius_RM(hot_rp, cold_rp,
                        alpha=0.5, gamma=0., sigma=50.)
idx = (np.abs(thetas - 0.)).argmin()
print('Morning (east) rp', np.round(r_p[idx], 4))
idx = (np.abs(thetas - np.pi / 2)).argmin()
print('Evening (west) rp', np.round(r_p[idx], 4))
ax1.plot(r_p * np.cos(thetas), r_p * np.sin(thetas),
         lw=2.5, c=colours[1], alpha=0.8)

# 4.7um.
cold_rp = (td_750_d / 100.)**0.5
cold_rp = mid_rp + (cold_rp - mid_rp) * exaggeration_factor
hot_rp = (td_1250_d / 100.)**0.5
hot_rp = mid_rp + (hot_rp - mid_rp) * exaggeration_factor
mean_rp = np.mean([hot_rp, cold_rp])
thetas, r_p = radius_RM(hot_rp, cold_rp,
                        alpha=0.5, gamma=0., sigma=50.)
idx = (np.abs(thetas - 0.)).argmin()
print('Morning (east) rp', np.round(r_p[idx], 4))
idx = (np.abs(thetas - np.pi / 2)).argmin()
print('Evening (west) rp', np.round(r_p[idx], 4))
ax1.plot(r_p * np.cos(thetas), r_p * np.sin(thetas),
         lw=2.5, c=colours[0], alpha=0.8)

plt.axis('off')
plt.savefig('/Users/davidgrant/Desktop/Harmonica/fortney_transmission_string.png', transparent=True)
plt.show()
