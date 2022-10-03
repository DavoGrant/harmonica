import pickle
import numpy as np
import matplotlib.pyplot as plt


with open('/Users/davidgrant/Downloads/ETC-calculation95e6e345-7e99-4542-8bc7-1cfc8539ae80e.p', 'rb') as in_f:
    t_dict = pickle.load(in_f)

print(t_dict.keys())
print(t_dict['OriginalInput'].keys())
print(t_dict['RawData'].keys())
print(t_dict['FinalSpectrum'].keys())
print(t_dict['PandeiaOutTrans'].keys())

a = 0
b = 1
no_mask_flux = np.sum(t_dict['RawData']['electrons_out'][a:b])
no_mask_wlc_err_ = np.sqrt(np.sum(t_dict['RawData']['var_out'][a:b])) / no_mask_flux
print(t_dict['RawData']['wave'][a], t_dict['RawData']['wave'][b])
print(np.median(no_mask_wlc_err_) * 1e6)


plt.plot(t_dict['RawData']['electrons_in'])
plt.plot(t_dict['RawData']['electrons_out'])
print(np.median(t_dict['RawData']['electrons_in']**0.5 / t_dict['RawData']['electrons_in']) * 1e6)
plt.show()

plt.plot(t_dict['RawData']['wave'], np.sqrt(t_dict['RawData']['var_out']) / t_dict['RawData']['electrons_out'] * 1e6)
plt.ylim(0, 1000)
plt.show()

