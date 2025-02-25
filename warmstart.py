import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from defaults import defaults
from seasonal_setup import get_setup_data
from seasonal import seasonal_run

# Normalize star.
star = defaults['star'].strip().upper()

# Get setup data.
setup_data = get_setup_data()

# Extract variables from setup_data.
jmx  = setup_data['jmx']
fl   = setup_data['fl']    # Ensure these exist in your setup data.
fw   = setup_data['fw']
delt = setup_data['delt']

# For testing, define these variables:
dummy_delt = delt
results = seasonal_run()
L_out = results['L_out']
W_out = results['W_out']
h_out = results['h_out']
dummy_n = L_out.shape[1]
# Define dummy_phi as a linear space from -90 to 90 (one value per grid cell).
dummy_phi = np.linspace(-90, 90, jmx)

scaleQkeep       = []
meanicelin_list  = []
meanTg_list      = []

# Loop over a range of scaleQ values.
for i in range(1, 22):
    scaleQ_i = 1.35 - (i * 0.05)
    scaleQkeep.append(scaleQ_i)
    
    # Update defaults with current scaleQ value.
    defaults['scaleQdef'] = scaleQ_i

    # Run the seasonal simulation (this will now use the updated scaleQ).
    results = seasonal_run()
    L_out = results['L_out']
    W_out = results['W_out']
    h_out = results['h_out']
    
    # Compute a mean temperature metric Tg.
    Tg = (np.dot(L_out.T, fl) + np.dot(W_out.T, fw)) / jmx
    
    # Create a time vector.
    days = np.arange((1 / dummy_delt) - 1, -1, -1)
    tt   = dummy_n - days - 1  # adjust for 0-indexing
    tt_int = tt.astype(int)
    
    # Define indices corresponding to the northern hemisphere.
    j_nh = np.arange(int(jmx / 2), jmx)
    
    # Extract the last year's water temperatures and sea ice thickness.
    W_lastyear = W_out[np.ix_(j_nh, tt_int)]
    h_output   = h_out[np.ix_(j_nh, tt_int)]
    
    # Adjust W_lastyear by subtracting a small value that depends on grid cell and time step.
    for idx in range(W_lastyear.shape[0]):
        for jdx in range(W_lastyear.shape[1]):
            W_lastyear[idx, jdx] -= 0.000001 * (idx + 1) + 0.000001 * (jdx + 1)
    
    # Calculate the "ice line" for each day of the last year.
    icelin = np.empty(360)
    for daystep in range(360):
        x_vals = W_lastyear[:, daystep]
        y_vals = dummy_phi[j_nh]
        f_interp = interp1d(x_vals, y_vals, bounds_error=False, fill_value=np.nan)
        ic_val = f_interp(-2.013)
        if np.isnan(ic_val):
            if np.min(x_vals) < -2.013:
                ic_val = 0
            elif np.min(x_vals) > -2.013:
                ic_val = 90
        icelin[daystep] = ic_val
    
    mean_icelin = np.nanmean(icelin)
    meanicelin_list.append(mean_icelin)
    
    meanTg_val = np.mean(Tg[tt_int])
    meanTg_list.append(meanTg_val)

# Combine scaleQ, mean ice line, and mean temperature into one array.
d_a = np.array([scaleQkeep, meanicelin_list, meanTg_list])
d_c = d_a.T

# Save to a text file with tab delimiters and 6-digit precision.
np.savetxt('G_dwarf_ws.txt', d_c, delimiter='\t', fmt='%.6f')

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['DejaVu Serif'] 

# Create a figure with 2 rows of subplots that share the x-axis.
fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Top subplot: Mean Temperature vs. scaleQ.
axs[0].plot(scaleQkeep, meanTg_list, 'o-', color='red', markersize=8)
axs[0].set_ylabel('Mean Temperature')
axs[0].set_title('Mean Temperature vs. scaleQ')
axs[0].grid(True)

# Bottom subplot: Mean Ice Line vs. scaleQ.
axs[1].plot(scaleQkeep, meanicelin_list, 's-', color='red', markersize=8)
axs[1].set_xlabel('scaleQ')
axs[1].set_ylabel('Mean Ice Line')
axs[1].set_title('Mean Ice Line vs. scaleQ')
axs[1].grid(True)

plt.tight_layout()
plt.show()

