"""
seasonal.py

Main time-stepping routine for the seasonal Energy Balance Model (EBM).
This routine is a translation of the MATLAB code by North & Coakley with
sea ice modifications. It uses:
  - seasonal_setup (to initialize the domain, parameters, and state)
  - seasonal_solar (function sun)
  - albedo_seasonal (function albedo_seasonal)
  - icebalance (function icebalance)
  
All required parameters are assumed to be set in seasonal_setup and returned
in the dictionary "setup_data".
"""

import numpy as np
from scipy.io import loadmat
from defaults import defaults

# Unpack defaults.
(jmx, runlength, scaleQ, A, B, Dmag, nu, Cl, Cw, coldstartflag, 
 hadleyflag, albedoflag, obl, ecc, per, star, land, casenamedef) = (
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
    defaults['star'],
    defaults['landdef'],
    defaults['casenamedef']
)

from get_broadband_albedo import get_broadband_albedo
from albedo_seasonal import albedo_seasonal
from seasonal_solar import sun  # function: sun(xi, obl, ecc, long, star)
from seasonal_setup import get_setup_data
from icebalance import icebalance

# Normalize star and get broadband albedo parameters.
star = defaults['star'].strip().upper()
print("Star value:", repr(star))
broadband_params = get_broadband_albedo(star)
A_o = broadband_params['A_o']
A_l = broadband_params['A_l']
A_50 = broadband_params['A_50']


def seasonal_run():
    # Get the setup data (assumed provided by seasonal_setup).
    setup_data = get_setup_data()
    
    # *** Incorporate the current scaleQ into the setup ***
    # If your setup_data contains a "base" insolation (e.g., 'insol_base'),
    # then update the insolation using the current defaults['scaleQdef'].
     # Incorporate the current scaleQ into the setup.
    current_scaleQ = defaults['scaleQdef']
    # If the base insolation isn't stored yet, store it.
    if 'insol_base' not in setup_data:
        setup_data['insol_base'] = setup_data['insol'].copy()
    # Now update the insolation array using the current scaleQ.
    setup_data['insol'] = current_scaleQ * setup_data['insol_base']
    # Unpack necessary variables from setup_data.
    jmx         = setup_data['jmx']            # number of grid cells
    nts         = setup_data['nts']            # total number of time steps
    nstepinyear = setup_data['nstepinyear']
    delt        = setup_data['delt']
    ts          = setup_data['ts']             # starting time index for insolation
    n_out       = setup_data['n_out']          # output time indices (an array)
    insol       = setup_data['insol']          # insolation array (should now reflect scaleQ)
    A           = setup_data['A']
    Cl_delt     = setup_data['Cl_delt']
    Cw_delt     = setup_data['Cw_delt']
    Tfrz        = setup_data['Tfrz']
    albedoflag  = setup_data['albedoflag']
    L           = setup_data['L'].copy()       # land temperature (vector length jmx)
    W           = setup_data['W'].copy()       # ocean temperature (vector length jmx)
    xfull       = setup_data['xfull']          # full spatial grid for albedo
    rghflag     = setup_data.get('rghflag', 0)   # if not defined, assume 0
    ice_model   = setup_data['ice_model']
    
    # Matrices used in the "no ice" branch.
    invM        = setup_data['invM']
    B           = setup_data['B']
    nu_fw       = setup_data['nu_fw']
    fw          = setup_data['fw']
    delt_Lf     = setup_data['delt_Lf']
    Lfice       = setup_data['Lfice']
    Cw          = setup_data['Cw']
    Diff_Op     = setup_data['Diff_Op']
    M           = setup_data['M']
    
    # For use in icebalance, assume current ice thickness is stored.
    h           = setup_data.get('h', np.zeros(jmx))
    # For climatological albedo.
    clim_alb_l  = setup_data.get('clim_alb_l', None)
    clim_alb_w  = setup_data.get('clim_alb_w', None)
    thedays_setup = setup_data.get('thedays', None)
    conduct     = setup_data.get('conduct', 2.0)
    delx        = setup_data['delx']
    
    # Initialize the right-hand side vector r (size 2*jmx)
    r = np.zeros(2 * jmx)
    
    # --- Initial warm-start conditions ---
    if ice_model:
        ice    = np.where(W < Tfrz)[0]
        notice = np.where(W >= Tfrz)[0]
        h      = np.zeros(jmx)
        h[ice] = 2.0
        
        # Determine albedo.
        if albedoflag and (clim_alb_l is not None) and (thedays_setup is not None):
            alb_l = clim_alb_l[:, int(thedays_setup[0]) - 1]
            alb_w = clim_alb_w[:, int(thedays_setup[0]) - 1]
        else:
            alb_l, alb_w = albedo_seasonal(L, W, xfull, A_o, A_l, A_50)
        
        # Get the insolation on day ts (adjust for 0-indexing)
        S = insol[:, int(ts) - 1]
        rprimel = A - ((1 - alb_l) * S)
        rprimew = A - ((1 - alb_w) * S)
        r[0::2] = L * Cl_delt - rprimel
        r[1::2] = W * Cw_delt - rprimew
        
        # Call icebalance to update temperatures and ice thickness.
        T, L, W, Fnet = icebalance(jmx, ice, notice, conduct, h, Tfrz, rprimew,
                                    Cw_delt, M, r, Diff_Op, B, nu_fw, fw, delx, Cw, W)
    else:
        ice = np.array([])
    
    # --- Preallocate output arrays ---
    num_out = len(n_out)
    L_out = np.zeros((jmx, num_out))
    W_out = np.zeros((jmx, num_out))
    h_out = np.zeros((jmx, num_out)) if ice_model else None
    
    tday = np.zeros(nts)
    yr   = np.zeros(nts)
    day  = np.zeros(nts, dtype=int)
    idx_out = 0
    
    # --- Main time-stepping loop ---
    for n in range(1, nts + 1):
        tday[n - 1] = ts + 2 + 1 + (n - 1) * 360 * delt
        yr[n - 1]   = np.floor((-1 + tday[n - 1]) / 360)
        day[n - 1]  = int(np.floor(tday[n - 1] - yr[n - 1] * 360))
        
        if albedoflag and (clim_alb_l is not None):
            nn = day[n - 1] - int(360 * delt)
            if nn < 0:
                nn = int(360 - 360 * delt * 0.5)
            alb_l = clim_alb_l[:, nn - 1]
            alb_w = clim_alb_w[:, nn - 1]
        else:
            alb_l, alb_w = albedo_seasonal(L, W, xfull, A_o, A_l, A_50)
        
        S = insol[:, day[n - 1] - 1]
        ghw = np.where(W > 46.2)[0]
        ghl = np.where(L > 46.2)[0]
        
        rprimel = A - ((1 - alb_l) * S)
        rprimew = A - ((1 - alb_w) * S)
        if rghflag:
            A1 = 300
            rprimew[ghw] = A1 - (1 - alb_w[ghw]) * S[ghw]
            rprimel[ghl] = A1 - (1 - alb_l[ghl]) * S[ghl]
        
        r[0::2] = L * Cl_delt - rprimel
        r[1::2] = W * Cw_delt - rprimew
        
        if ice_model:
            ice    = np.where(h > 0.001)[0]
            notice = np.where(h <= 0.001)[0]
            T, L, W, Fnet = icebalance(jmx, ice, notice, conduct, h, Tfrz, rprimew,
                                        Cw_delt, M, r, Diff_Op, B, nu_fw, fw, delx, Cw, W)
            h[ice] = np.maximum(0.0, h[ice] - delt_Lf * Fnet[ice])
            
            T_ocean = T[1::2]
            cold    = np.where(T_ocean[notice] < Tfrz)[0]
            new     = notice[cold]
            if len(new) > 0:
                h[new] = -Cw / Lfice * (W[new] - Tfrz)
                W[new] = Tfrz
        else:
            T = np.linalg.solve(invM, r)
            L = T[0::2]
            W = T[1::2]
        
        if idx_out < num_out and n == n_out[idx_out]:
            L_out[:, idx_out] = L
            W_out[:, idx_out] = W
            if ice_model:
                h_out[:, idx_out] = h
            idx_out += 1
    
    # Create a descending day vector and extract outputs corresponding to the last year.
    days_vec = np.arange(nstepinyear - 1, -1, -1)
    tt_idx   = idx_out - days_vec  # indices for output (adjust as needed)
    thedays  = day[tt_idx.astype(int) - 1]
    Lann     = L_out[:, tt_idx.astype(int) - 1]
    Wann     = W_out[:, tt_idx.astype(int) - 1]
    
    output = {
        'L_out': L_out,
        'W_out': W_out,
        'h_out': h_out,
        'thedays': thedays,
        'Lann': Lann,
        'Wann': Wann,
        'tday': tday,
        'yr': yr,
        'day': day,
        'delt': delt  # include delt so the test code can use it
    }
    return output



