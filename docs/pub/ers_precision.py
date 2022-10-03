import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt


# Setup paths.
stellar_spec = 'stellar-spec-W39-G395H-NRS1-Grant.nc'  # NRS2 stellar spec.
data_dir = '/Users/davidgrant/research/data/jwst/nirspec/g395h/wasp39/visit_01366'
version_dir = os.path.join(data_dir, 'reduction_v9')
stage_2_dir = os.path.join(version_dir, 'stage_2')

spec_path = os.path.join(stage_2_dir, stellar_spec)
print("Stellar spectra ={}".format(spec_path))
ds_ = xr.open_dataset(spec_path)

# For harmonica test: 1um region corresponds to detector noise of 99ppm.
a = 600
b = 2000
no_mask_flux = np.sum(ds_['flux'].values[:, a:b], axis=1)
no_mask_wlc_err_ = np.sqrt(np.sum(np.square(ds_['flux_error'].values[:, a:b]), axis=1)) / no_mask_flux
print(ds_['wavelength'].values[a], ds_['wavelength'].values[b])
print(np.median(no_mask_wlc_err_) * 1e6)
plt.plot(ds_['flux_error'].values[0] / ds_['flux'].values[0] * 1e6)
plt.ylim(0, 5000)
plt.show()
