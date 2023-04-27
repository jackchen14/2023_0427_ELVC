import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib import cm

file_name = "stage2_converted/NL01_241"
filepath_np = os.path.join("/home/bioasp/Desktop/mcc_np/", f"{file_name}.npy")
mfcc_data = np.load(filepath_np)
mfcc_data = mfcc_data[:,1:]
print(mfcc_data.shape)
lastshape = (mfcc_data.shape)[-1]
mfcc_data = mfcc_data.reshape(-1,lastshape)
mfcc_data = np.transpose(mfcc_data)

fig, ax = plt.subplots()
mfcc_data= np.swapaxes(mfcc_data, 0 ,1)
cax = ax.imshow(mfcc_data, interpolation='nearest', cmap=cm.coolwarm, origin='lower')
ax.set_title(file_name + '_mcc')
fig.colorbar(cax)
plot_path = f"/home/bioasp/Desktop/mcc_plot/converted_mcc_plot/{file_name}.png"
plt.savefig(plot_path)