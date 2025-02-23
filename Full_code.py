import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# --------------------------------------------------------------------
# Placeholder for defaults() -- replace with your actual default settings
# --------------------------------------------------------------------
def defaults():
    global jmxdef, runlengthdef, scaleQdef, Adef, Bdef, Dmagdef, nudef
    global Cldef, Cwdef, coldstartdef, hadleyflagdef, albedoflagdef
    global obldef, eccdef, perdef, ice_modeldef, landdef, casenamedef
    # Default values (adjust as needed)
    jmxdef = 360
    runlengthdef = 300
    scaleQdef = 0.55
    Adef = 203.3
    Bdef = 2.08
    Dmagdef = 0.44
    nudef = 3
    Cldef = 0.45
    Cwdef = 9.8
    coldstartdef = 0
    hadleyflagdef = 1.0
    albedoflagdef = 0.0
    obldef = 0
    eccdef = 0
    perdef = 102.0651294475608
    ice_modeldef = 0
    landdef = 'modern'
    casenamedef = 'Control'

# --------------------------------------------------------------------
# Placeholder for albedo_seasonal() -- replace with your own routine
# --------------------------------------------------------------------
def albedo_seasonal(Lann_col, Wann_col, xfull):
    # Dummy implementation that returns constant albedo arrays.
    return np.full_like(xfull, 0.3), np.full_like(xfull, 0.3)
import numpy as np

def albedo_seasonal(Lann_col, Wann_col, xfull):
    """
    Recalculate albedo for land (alb_l) and water (alb_w) based on the input arrays.

    Parameters
    ----------
    L : array_like
        Array of land temperatures or related variable.
    W : array_like
        Array of water temperatures or related variable.
    x : array_like
        Array (e.g., latitude transformed variable) used in albedo calculation.

    Returns
    -------
    alb_l : ndarray
        Calculated albedo for land.
    alb_w : ndarray
        Calculated albedo for water.
    """
    # Calculate base albedo values
    alb_w = 0.31948 + 0.08 * (3 * x**2 - 1) / 2 - 0.05
    alb_l = 0.42480 + 0.08 * (3 * x**2 - 1) / 2 + 0.05

    # Reassign albedo values based on threshold conditions:
    alb_w[W <= -2] = 0.51363
    alb_l[L <= -2] = 0.51363

    return alb_l, alb_w


# --------------------------------------------------------------------
# seasonal_solar: Computes the insolation and day offset (from MATLAB code)
# --------------------------------------------------------------------
def seasonal_solar(xi, obl, ecc, lon):
    xi = np.asarray(xi)
    npts = len(xi)
    t1 = 2808 / 2.0754
    dr_conv = np.pi / 180.0
    rd_conv = 1 / dr_conv
    lmr = 0.0  # reference day (March 21)
    beta = np.sqrt(1 - ecc**2)
    lm0 = lmr - 2 * (
        (0.5 * ecc + 0.125 * ecc**3) * (1 + beta) * np.sin(lmr - lon * dr_conv)
        - 0.25 * ecc**2 * (0.5 + beta) * np.sin(2 * (lmr - lon * dr_conv))
        + 0.125 * ecc**3 * ((1/3) + beta) * np.sin(3 * (lmr - lon * dr_conv))
    )
    it = 360
    lm = np.linspace(lm0, lm0 + 2 * np.pi - 2 * np.pi / it, it)
    calendarlongitude = (
        lm
        + (2 * ecc - 0.25 * ecc**3) * np.sin(lm - lon * dr_conv)
        + (5/4) * (ecc**2) * np.sin(2 * (lm - lon * dr_conv))
        + (13/12) * (ecc**3) * np.sin(3 * (lm - lon * dr_conv))
    )
    calendarlongitude = np.mod(calendarlongitude, 2 * np.pi)
    lme = np.concatenate((lm, lm, lm))
    calendarlongitudee = np.concatenate((
        calendarlongitude - 2 * np.pi,
        calendarlongitude,
        calendarlongitude + 2 * np.pi
    ))
    nts = 360
    t = np.linspace(1, 360, nts)
    tl = (t - 90) * dr_conv
    tl = np.mod(tl, 2 * np.pi)
    sort_idx = np.argsort(calendarlongitudee)
    f_interp = interp1d(calendarlongitudee[sort_idx], lme[sort_idx],
                        kind='linear', fill_value="extrapolate")
    astrolon = f_interp(tl)
    astroloni = 90 + (180 / np.pi) * astrolon
    astroloni = np.mod(astroloni, 360)
    dayoffset = (astroloni - t) * 365 / 360
    ti = astroloni
    distance = (1 - ecc**2) / (1 + ecc * np.cos(dr_conv * (450 + ti - lon)))
    s_delt = -np.sin(dr_conv * obl) * np.cos(dr_conv * ti)
    c_delt = np.sqrt(1.0 - s_delt**2)
    t_delt = s_delt / c_delt
    delt = np.arcsin(s_delt) * rd_conv  # in degrees
    phi = np.arcsin(xi) * rd_conv         # in degrees
    wk = np.zeros((npts, nts))
    for i in range(nts):
        for j in range(npts):
            if delt[i] > 0.0:
                if phi[j] >= 90 - delt[i]:
                    wk[j, i] = t1 * xi[j] * s_delt[i] / (distance[i]**2)
                elif (-phi[j] >= (90 - delt[i])) and (phi[j] < 0):
                    wk[j, i] = 0
                else:
                    c_h0 = -np.tan(dr_conv * phi[j]) * t_delt[i]
                    c_h0 = np.clip(c_h0, -1, 1)
                    h0 = np.arccos(c_h0)
                    wk[j, i] = t1 * (h0 * xi[j] * s_delt[i] +
                                     np.cos(dr_conv * phi[j]) * c_delt[i] * np.sin(h0)) \
                              / (distance[i]**2 * np.pi)
            else:
                if phi[j] >= (90 + delt[i]):
                    wk[j, i] = 0
                elif (-phi[j] >= (90 + delt[i])) and (phi[j] < 0):
                    wk[j, i] = t1 * xi[j] * s_delt[i] / (distance[i]**2)
                else:
                    c_h0 = -np.tan(dr_conv * phi[j]) * t_delt[i]
                    c_h0 = np.clip(c_h0, -1, 1)
                    h0 = np.arccos(c_h0)
                    wk[j, i] = t1 * (h0 * xi[j] * s_delt[i] +
                                     np.cos(dr_conv * phi[j]) * c_delt[i] * np.sin(h0)) \
                              / (np.pi * distance[i]**2)
    insol = wk
    return insol, dayoffset

# --------------------------------------------------------------------
# seasonal_setup: Initializes default parameters and global variables.
# --------------------------------------------------------------------
def seasonal_setup():
    global jmx, runlength, scaleQ, A, B, Dmag, nu, Cl, Cw, coldstartflag, hadleyflag
    global albedoflag, obl, ecc, per, ice_model, land, casename, Tfrz, conduct, Lfice
    global ts, tf, nstepinyear, delt, nts, n_out, delx, x, xfull, phi, Toffset, L, W
    global D, fl, fw, insol, Cw_delt, Cl_delt, delt_Lf, nu_fw, nu_fl, lambda_, a, c, b
    global Diff_Op, bw, bl, Mw, Ml, M, invM, clim_alb_l, clim_alb_w, thedays, Lann, Wann

    # Load defaults
    defaults()

    # Check and set values if not provided
    if 'jmx' not in globals(): jmx = jmxdef
    if 'runlength' not in globals(): runlength = runlengthdef
    if 'scaleQ' not in globals(): scaleQ = scaleQdef
    if 'A' not in globals(): A = Adef
    if 'B' not in globals(): B = Bdef
    if 'Dmag' not in globals(): Dmag = Dmagdef
    if 'nu' not in globals(): nu = nudef
    if 'Cl' not in globals(): Cl = Cldef
    if 'Cw' not in globals(): Cw = Cwdef
    if 'coldstartflag' not in globals(): coldstartflag = coldstartdef
    if 'hadleyflag' not in globals(): hadleyflag = hadleyflagdef
    if 'albedoflag' not in globals(): albedoflag = albedoflagdef
    if 'obl' not in globals(): obl = obldef
    if 'ecc' not in globals(): ecc = eccdef
    if 'per' not in globals(): per = perdef
    if 'ice_model' not in globals(): ice_model = ice_modeldef
    if 'land' not in globals(): land = landdef
    if 'casename' not in globals(): casename = casenamedef

    # Size of domain, ensure even number
    jmx = 2 * (jmx // 2)

    # Freezing temperature of ocean in °C and related parameters
    Tfrz = -2
    conduct = 2               # conductivity of sea ice (for explicit sea ice)
    Lfice = 9.8 * 83.5 / 50     # Lf/Cp * depth of mixed layer

    # Time loop parameters
    ts = 90           # First day of run is present-day equinox
    tf = runlength - 0.25
    nstepinyear = 60
    delt = 1.0 / nstepinyear
    nts = int(np.floor(tf / delt))
    n_out = np.arange(1, nts + 1)

    # Set up x array (domain from -1 to 1)
    delx = 2.0 / jmx
    x = np.arange(-1.0 + delx, 1.0, delx)
    xfull = np.arange(-1 + delx / 2, 1 - delx / 2, delx)
    phi = np.arcsin(xfull) * 180 / np.pi

    # Set up initial temperature profile
    if coldstartflag:
        Toffset = -40
    else:
        Toffset = 0.0
    L = 7.5 + 20 * (1 - 2 * xfull**2) + Toffset
    W = 7.5 + 20 * (1 - 2 * xfull**2) + Toffset

    # Heat diffusion coefficient
    if hadleyflag:
        D = Dmag * (1 + 9 * np.exp(-(x / np.sin(25 * np.pi / 180))**6))
    else:
        D = Dmag * np.ones_like(x)

    # Land fraction
    fl = 0.05 * np.ones(jmx)
    if 'Preca' in land[:5]:
        # Pangea (middle Cambrian)
        j = phi <= -45
        fl[j] = 0.3
        j = (phi > -45) & (phi <= 30)
        fl[j] = 0.5
    elif 'Ordov' in land[:5]:
        # Gondwanaland
        j = phi <= -60
        fl[j] = 0.95
        j = (phi > -60) & (phi <= 70)
        fl[j] = 0.3
    elif 'Symme' in land[:5]:
        # Symmetric
        fl.fill(0.01)
    else:  # Modern
        fl = 0.38 * np.ones(jmx)
        j = phi <= -60
        fl[j] = 0.95
        j = (phi > -60) & (phi <= -40)
        fl[j] = 0.05
        j = (phi > -40) & (phi <= 20)
        fl[j] = 0.25
        j = (phi > 20) & (phi <= 70)
        fl[j] = 0.5
    fl = fl * 0.01 / np.mean(fl)

    # Ocean fraction
    fw = 1 - fl

    # Obtain annual array of daily averaged insolation.
    # Note: seasonal_solar returns (insol, dayoffset); here we only use insol.
    insol, _ = seasonal_solar(xfull, obl, ecc, per)
    insol = scaleQ * insol[:, np.concatenate(([359], np.arange(360)))]  # Reorder to match

    # Set up deltas and diffusion parameters
    Cw_delt = Cw / delt
    Cl_delt = Cl / delt
    delt_Lf = delt / Lfice
    nu_fw = nu / fw
    nu_fl = nu / fl
    lambda_ = D / delx / delx * (1 - x**2)
    a = np.concatenate(([0], -lambda_))
    c = np.concatenate(([-lambda_], [0]))
    b = -a - c
    Diff_Op = -(np.diag(b) + np.diag(c[:-1], 1) + np.diag(a[1:], -1))
    bw = Cw_delt + B + nu_fw - a - c
    bl = Cl_delt + B + nu_fl - a - c
    Mw = np.diag(bw) + np.diag(c[:-1], 1) + np.diag(a[1:], -1)
    Ml = np.diag(bl) + np.diag(c[:-1], 1) + np.diag(a[1:], -1)

    # Matrix M for implicit solution
    M = np.zeros((2 * jmx, 2 * jmx))
    for j in range(jmx):
        M[2 * j, 0::2 * jmx] = Ml[j, :]
        M[2 * j, 2 * j] = -nu_fl[j]
        M[2 * j + 1, 1::2 * jmx] = Mw[j, :]
        M[2 * j + 1, 2 * j - 1] = -nu_fw[j]
    invM = np.linalg.inv(M)

    # Climatological albedo if no albedo feedback (dummy implementation)
    if albedoflag:
        clim_alb_l = np.zeros((jmx, 360))
        clim_alb_w = clim_alb_l.copy()
        n = 0
        # Create dummy arrays for demonstration
        thedays = np.arange(360)
        Lann = np.ones((jmx, 360)) * 7.5
        Wann = np.ones((jmx, 360)) * 7.5
        for t in thedays:
            clim_alb_l[:, t], clim_alb_w[:, t] = albedo_seasonal(Lann[:, n], Wann[:, n], xfull)
            n += 1

# --------------------------------------------------------------------
# seasonal: Dummy seasonal simulation that uses the setup values.
# --------------------------------------------------------------------
define_seasonal()
import numpy as np

# Initial setup for variables and parameters, assumes `seasonal_setup` is executed earlier
# seasonal_setup()

r = np.zeros(2 * jmx)

if ice_model:
    # Set up initial sea ice thickness of 2 m where freezing
    ice = np.where(W < Tfrz)[0]
    notice = np.where(W >= Tfrz)[0]
    h = np.zeros(jmx)
    k = np.zeros_like(h)
    h[ice] = 2
    
    if albedoflag:
        alb_l = clim_alb_l[:, thedays[0]]
        alb_w = clim_alb_w[:, thedays[0]]
    else:
        alb_l, alb_w = albedo_seasonal(L, W, xfull)
        
    S = insol[:, ts]
    rprimew = A - (1 - alb_w) * S
    rprimel = A - (1 - alb_l) * S
    r[0:2:2*jmx] = L * Cl_delt - rprimel
    r[1:2:2*jmx] = W * Cw_delt - rprimew
    
    icebalance()
else:
    ice = []

L_out = np.zeros((jmx, len(n_out)))
W_out = np.zeros_like(L_out)

if ice_model:
    h_out = W_out

# Clear temporary variables
tday, yr, day = [], [], []

# Begin loop
idx_out = 0
for n in range(nts):
    tday.append(ts + 2 + 1 + (n - 1) * 360 * delt)
    yr.append(np.floor((-1 + tday[n]) / 360))
    day.append(np.floor(tday[n] - yr[n] * 360))
    
    # Create initial albedo
    if albedoflag:
        nn = day[n]
        nn = nn - 360 * delt
        if nn < 0:
            nn = 360 - 360 * delt * 0.5
        alb_l = clim_alb_l[:, nn]
        alb_w = clim_alb_w[:, nn]
    else:
        alb_l, alb_w = albedo_seasonal(L, W, xfull)

    # Calculate insolation
    S = insol[:, int(day[n])]
    
    # Source terms
    rprimew = A - (1 - alb_w) * S
    rprimel = A - (1 - alb_l) * S
    r[0:2:2*jmx] = L * Cl_delt - rprimel
    r[1:2:2*jmx] = W * Cw_delt - rprimew

    if ice_model:
        # First consider where sea ice already exists
        ice = np.where(h > 0.001)[0]
        notice = np.where(h <= 0.001)[0]
        if len(ice) > 0:
            icebalance()
            h[ice] = h[ice] - delt_Lf * Fnet(ice)
            h[ice] = np.maximum(0., h[ice])

        # Second consider the conditions of new ice growth over the ocean
        cold = np.where(T[2 * notice] < Tfrz)[0]
        new = notice[cold]
        if len(new) > 0:
            h[new] = -Cw / Lfice * (W[new] - Tfrz)
            W[new] = Tfrz
    else:
        T = np.linalg.inv(M) @ r
        L = T[0::2]
        W = T[1::2]

    # Output
    if n == n_out[idx_out]:
        L_out[:, idx_out] = L
        W_out[:, idx_out] = W
        if ice_model:
            h_out[:, idx_out] = h
        idx_out += 1

# Final processing and output
n = idx_out - 1
days = np.arange(nstepinyear - 1, -1, -1)
tt = n - days
thedays = day[tt]
Lann = L_out[:, tt]
Wann = W_out[:, tt]

# Optionally, save results
# np.save('temperatures.npy', (Lann, Wann, thedays))

# Plot output
plotoutput()


# --------------------------------------------------------------------
# Main simulation and analysis
# --------------------------------------------------------------------
if __name__ == "__main__":
    # Ensure seasonal_setup() is called so that global parameters are initialized.
    seasonal_setup()
    defaults()
    # Diagnostics arrays
    ind = 0
    scaleQkeep = []
    meanicelin = []
    meanTg = []
    mean60minusEq = []

    # Loop over a range of parameter values (e.g., scaling the stellar constant)
    for i in range(1, 11):
        scaleQ = (i * 0.01) + 0.14
        scaleQkeep.append(scaleQ)
        ind += 1

        # Run the seasonal simulation (this updates globals: W_out, Tg, tt, phi)
        seasonal()

        # For the Northern Hemisphere, use indices from jmx/2 to jmx.
        j_nh = range(jmx // 2, jmx)
        # Extract the temperature field for the last 60 days (tt)
        W_lastyear = W_out[np.ix_(list(j_nh), tt)]
        icelin = []
        for daystep in range(W_lastyear.shape[1]):
            # Interpolate to find the latitude (using phi) where temperature equals -2°C.
            icelin.append(np.interp(-2, W_lastyear[:, daystep], phi[j_nh]))
        meanicelin.append(np.mean(icelin))
        print(f"Mean ice latitude is {meanicelin[-1]}")

        meanTg.append(np.mean(Tg[tt]))
        # Dummy temperature difference (using arbitrary grid indices)
        mean60minusEq.append(np.mean(W_out[47, tt] - W_out[26, tt]))

    # Adjust meanicelin values if needed
    try:
        maxW_lastyear = np.max(W_lastyear)
    except NameError:
        maxW_lastyear = 0
    meanicelin = [0 if np.isnan(val) else (90 if maxW_lastyear < -2 else val) for val in meanicelin]

    # Save diagnostics to file
    d_a = np.array([scaleQkeep, meanicelin, meanTg]).T
    np.savetxt('F_dwarf_recal_blueice_co2_3bar_300yr_1perc.txt', d_a, delimiter='\t', fmt='%.6f')

    # Plot the diagnostics
    fig, axs = plt.subplots(3, 1, figsize=(10, 10))
    axs[0].plot(scaleQkeep, meanTg, 'o-')
    axs[0].set_xlabel('Stellar Constant')
    axs[0].set_ylabel('Global Mean Temperature')

    axs[1].plot(scaleQkeep, meanicelin, 'o-')
    axs[1].set_xlabel('Stellar Constant')
    axs[1].set_ylabel('Mean Ice Line Latitude')

    axs[2].plot(scaleQkeep, mean60minusEq, 'o-')
    axs[2].set_xlabel('Stellar Constant')
    axs[2].set_ylabel('Mean Temp. (60N minus Equator)')

    plt.tight_layout()
    plt.show()
