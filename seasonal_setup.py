"""
seasonal_setup.py

Python translation of the MATLAB seasonal_setup code for the water ice warm‐start case.
It uses defaults from defaults.py, broadband albedo parameters from get_broadband_albedo.py,
insolation from seasonal_solar.py, and albedo calculations from albedo_seasonal.py.
"""

import numpy as np
from scipy.io import loadmat
from defaults import defaults

(jmx, runlength, scaleQ, A, B, Dmag, nu, Cl, Cw, coldstartflag, 
 hadleyflag, albedoflag, obl, ecc, per, star,ice_model, land, casename) = (
    defaults['jmxdef'],
    defaults['runlengthdef'],
    defaults['scaleQdef'],
    defaults['Adef'],
    defaults['Bdef'],
    defaults['Dmagdef'],
    defaults['nudef'],
    defaults['Cldef'],
    defaults['Cwdef'],
    defaults['coldstartdef'],
    defaults['hadleyflagdef'],
    defaults['albedoflagdef'],
    defaults['obldef'],
    defaults['eccdef'],
    defaults['perdef'],
    defaults['ice_modeldef'],
    defaults['landdef'],
    defaults['casenamedef'],
    defaults['star']
)



from get_broadband_albedo import get_broadband_albedo
from albedo_seasonal import albedo_seasonal
from seasonal_solar import sun  # function: sun(xi, obl, ecc, long, star)

# --- Size of domain ---
# Ensure jmx is even (MATLAB: jmx = 2*floor(jmx/2))
jmx = 2 * (jmx // 2)

# --- Physical constants and time loop parameters ---
Tfrz = -2         # Freezing temperature of ocean [°C]
conduct = 2
Lfice = 9.8 * 83.5 / 50.0   # latent heat related constant

# Time loop parameters
ts = 90                      # first day of run (present-day equinox)
tf = runlength - 0.25
nstepinyear = 360
delt = 1.0 / nstepinyear
nts = int(np.floor(tf / delt))
# n_out: array of time-step indices (MATLAB: 1:nts). In Python, we use 1-indexing convention here.
n_out = np.arange(1, nts + 1)

# --- Set up spatial grid ---
delx = 2.0 / jmx
# x: grid from -1+delx to 1-delx (MATLAB uses column vectors)
x = np.arange(-1.0 + delx, 1.0, delx)
# xfull: grid shifted by half a grid-cell
xfull = np.arange(-1 + delx/2, 1, delx)
phi = np.arcsin(xfull) * 180.0 / np.pi  # convert to degrees

# --- Set up initial temperature profile ---
Toffset = -40.0 if coldstartflag else 0.0
L = 7.5 + 20 * (1 - 2 * xfull**2) + Toffset
W = 7.5 + 20 * (1 - 2 * xfull**2) + Toffset

# --- Heat diffusion coefficient ---
if hadleyflag:
    D = Dmag * (1 + 9 * np.exp(- (x / np.sin(25 * np.pi / 180.0))**6))
else:
    D = Dmag * np.ones_like(x)

# --- Land fraction ---
fl = 0.05 * np.ones(jmx)
# Check the first five characters of the land string
if land[:5] == 'Preca':  # Pangea (middle Cambrian)
    j = np.where(phi <= -45)[0]
    fl[j] = 0.3
    j = np.where((phi > -45) & (phi <= 30))[0]
    fl[j] = 0.5
elif land[:5] == 'Ordov':  # Gondwanaland
    j = np.where(phi <= -60)[0]
    fl[j] = 0.95
    j = np.where((phi > -60) & (phi <= 70))[0]
    fl[j] = 0.3
elif land[:5] == 'Symme':  # Symmetric
    fl[:] = 0.34
elif land[:5] == 'Aquap':  # Aquaplanet
    fl[:] = 0.01
elif land[:5] == 'Landp':  # Landplanet
    # If you have a function 'landfraction', call it here; otherwise, set fl as needed.
    # For now, we leave fl unchanged.
    pass
else:  # Modern case
    fl = 0.38 * np.ones(jmx)
    j = np.where(phi <= -60)[0]
    fl[j] = 0.95
    j = np.where((phi > -60) & (phi <= -40))[0]
    fl[j] = 0.05
    j = np.where((phi > -40) & (phi <= 20))[0]
    fl[j] = 0.25
    j = np.where((phi > 20) & (phi <= 70))[0]
    fl[j] = 0.5

# --- Ocean fraction ---
fw = 1 - fl

# --- Obtain annual array of daily averaged insolation ---
# Call the sun function from seasonal_solar.py.
# (MATLAB calls seasonal_solarRD with xfull, obl, ecc, per. Our sun() expects xi, obl, ecc, long, star.)
# Here, we assume that 'per' (from defaults) plays the role of the fourth parameter.
# Also, we need a star type—here we assume 'G' (or you could use a variable).
star = 'G'
insol, distance, delt_arr = sun(xfull, obl, ecc, per, star)
# Rearrange columns: MATLAB uses insol(:, [360 1:359]) to shift the last column to the front.
# In Python (0-indexed): take last column and then columns 0 to 358.
insol = scaleQ * np.concatenate((insol[:, -1][:, np.newaxis], insol[:, :-1]), axis=1)

# --- Precompute some scaling factors ---
Cw_delt = Cw / delt
Cl_delt = Cl / delt
delt_Lf = delt / Lfice

nu_fw = nu / fw
nu_fl = nu / fl

# --- Construct the diffusion operator ---
lam = D / (delx**2) * (1 - x**2)   # element-wise (x is a vector of length jmx)
a = np.concatenate(([0], -lam))
c = np.concatenate((-lam, [0]))
b = -a - c
# Build Diff_Op as a tridiagonal matrix of size jmx x jmx.
Diff_Op = - (np.diag(b[:jmx]) +
             np.diag(c[:jmx-1], k=1) +
             np.diag(a[1:jmx], k=-1))

# --- Construct matrices for the energy balance equations ---
bw = Cw_delt + B + nu_fw - (a[:jmx] + c[:jmx])
bl = Cl_delt + B + nu_fl - (a[:jmx] + c[:jmx])

Mw = np.diag(bw) + np.diag(c[:jmx-1], k=1) + np.diag(a[1:jmx], k=-1)
Ml = np.diag(bl) + np.diag(c[:jmx-1], k=1) + np.diag(a[1:jmx], k=-1)

# Build the full matrix M of size 2*jmx x 2*jmx.
M = np.zeros((2*jmx, 2*jmx))
for j in range(jmx):
    # In MATLAB, rows for land are at odd indices and for ocean at even indices.
    # Here we use 0-indexing: let land row = 2*j and ocean row = 2*j+1.
    # Set land row: fill columns 0,2,4,... with Ml[j,:]
    M[2*j, 0:2*jmx:2] = Ml[j, :]
    M[2*j, 2*j+1] = -nu_fl[j]
    # Set ocean row: fill columns 1,3,5,... with Mw[j,:]
    M[2*j+1, 1:2*jmx:2] = Mw[j, :]
    M[2*j+1, 2*j] = -nu_fw[j]

invM = np.linalg.inv(M)

# --- Climatological albedo (if no albedo feedback) ---
# If albedoflag is true, we want to create annual arrays of daily averaged albedo.
if albedoflag:
    # Load temperature fields from a MAT file.
    # Assumes that 'temperatures.mat' contains variables:
    #    thedays: array of day indices (likely 1-indexed from MATLAB),
    #    Lann: land temperature array with shape (jmx, 360),
    #    Wann: water temperature array with shape (jmx, 360)
    temps = loadmat('temperatures.mat')
    # Ensure that thedays is a 1D array.
    thedays = temps['thedays'].squeeze()
    Lann = temps['Lann']
    Wann = temps['Wann']
    clim_alb_l = np.zeros((jmx, 360))
    clim_alb_w = np.zeros((jmx, 360))
    n = 0
    for t in thedays:
        # MATLAB: [clim_alb_l(:,t), clim_alb_w(:,t)] = albedo_seasonal(Lann(:,n), Wann(:,n), xfull);
        # In Python, we assume that Lann and Wann are arrays of shape (jmx, number_of_days).
        alb_l, alb_w = albedo_seasonal(Lann[:, n], Wann[:, n], xfull)
        # Adjust index: if thedays comes from MATLAB (1-indexed), subtract 1.
        idx = int(t) - 1
        clim_alb_l[:, idx] = alb_l
        clim_alb_w[:, idx] = alb_w
        n += 1
else:
    clim_alb_l = None
    clim_alb_w = None
setup_data = {
        'star':star,
        'jmx': jmx,
        'runlength': runlength,
        'scaleQ': scaleQ,
        'A': A,
        'B': B,
        'Dmag': Dmag,
        'nu': nu,
        'Cl': Cl,
        'Cw': Cw,
        'coldstartflag': coldstartflag,
        'hadleyflag': hadleyflag,
        'albedoflag': albedoflag,
        'obl': obl,
        'ecc': ecc,
        'per': per,
        'ice_model': ice_model,
        'land': land,
        'casename': casename,
        'Tfrz': Tfrz,
        'conduct': conduct,
        'Lfice': Lfice,
        'ts': ts,
        'tf': tf,
        'nstepinyear': nstepinyear,
        'delt': delt,
        'nts': nts,
        'n_out': n_out,
        'delx': delx,
        'x': x,
        'xfull': xfull,
        'phi': phi,
        'L': L,
        'W': W,
        'D': D,
        'fl': fl,
        'fw': fw,
        'insol': insol,
        'distance': distance,
        'delt_arr': delt_arr,   # declination from sun() function
        'Cw_delt': Cw_delt,
        'Cl_delt': Cl_delt,
        'delt_Lf': delt_Lf,
        'nu_fw': nu_fw,
        'nu_fl': nu_fl,
        'Diff_Op': Diff_Op,
        'a': a[:jmx],
        'c': c[:jmx],
        'b': b[:jmx],
        'bw': bw,
        'bl': bl,
        'Mw': Mw,
        'Ml': Ml,
        'M': M,
        'invM': invM,
        'clim_alb_l': clim_alb_l,
        'clim_alb_w': clim_alb_w,
        'phi': phi
    }
# --- Package all computed variables into a dictionary for later use ---
def get_setup_data():
    # ... compute Tfrz, conduct, Lfice, ts, tf, etc. ...
    setup_data = {
        'star': star,
        'jmx': jmx,
        'runlength': runlength,
        'scaleQ': scaleQ,
        'A': A,
        'B': B,
        'Dmag': Dmag,
        'nu': nu,
        'Cl': Cl,
        'Cw': Cw,
        'coldstartflag': coldstartflag,
        'hadleyflag': hadleyflag,
        'albedoflag': albedoflag,
        'obl': obl,
        'ecc': ecc,
        'per': per,
        'ice_model': ice_model,
        'land': land,
        'casename': casename,
        'Tfrz': Tfrz,
        'conduct': conduct,
        'Lfice': Lfice,
        'ts': ts,
        'tf': tf,
        'nstepinyear': nstepinyear,
        'delt': delt,
        'nts': nts,
        'n_out': n_out,
        'delx': delx,
        'x': x,
        'xfull': xfull,
        'phi': phi,
        'L': L,
        'W': W,
        'D': D,
        'fl': fl,
        'fw': fw,
        'insol': insol,
        'distance': distance,
        'delt_arr': delt_arr,   # declination from sun() function
        'Cw_delt': Cw_delt,
        'Cl_delt': Cl_delt,
        'delt_Lf': delt_Lf,
        'nu_fw': nu_fw,
        'nu_fl': nu_fl,
        'Diff_Op': Diff_Op,
        'a': a[:jmx],
        'c': c[:jmx],
        'b': b[:jmx],
        'bw': bw,
        'bl': bl,
        'Mw': Mw,
        'Ml': Ml,
        'M': M,
        'invM': invM,
        'clim_alb_l': clim_alb_l,
        'clim_alb_w': clim_alb_w,
        'phi': phi
    }
    return setup_data


