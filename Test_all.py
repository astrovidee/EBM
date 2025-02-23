import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# ============================================================================
# 1. Basic functions and global defaults
# ============================================================================

def defaults():
    global jmxdef, runlengthdef, scaleQdef, Adef, Bdef, Dmagdef, nudef
    global Cldef, Cwdef, coldstartdef, hadleyflagdef, albedoflagdef
    global obldef, eccdef, perdef, ice_modeldef, landdef, casenamedef
    # Use these default parameters to mimic your MATLAB version.
    jmxdef = 60           # grid cells
    runlengthdef = 100    # runlength in years (100 as in MATLAB)
    scaleQdef = 0.55
    Adef = 203.3
    Bdef = 2.09         # use 2.09 as in MATLAB
    Dmagdef = 0.44
    nudef = 3
    Cldef = 0.45
    Cwdef = 9.8
    coldstartdef = 0      # warm start
    hadleyflagdef = 1.0
    albedoflagdef = 0.0   # albedo feedback on (0 means feedback on)
    # In your MATLAB code these are set to 0.
    obldef = 0
    eccdef = 0
    perdef = 102.07
    ice_modeldef = 1      # enable sea ice
    landdef = 'modern'
    casenamedef = 'Control'

def ice_balance(jmx, ice, notice, h, conduct, Tfrz, r, rprimew, Cw_delt, 
                M, Diff_Op, B, nu_fw, fw, delx):
    eps = 1e-6
    k = np.zeros(jmx)
    # Only compute k where there is ice (avoid division by zero)
    k[ice] = conduct / np.maximum(h[ice], eps)
    r[2 * ice + 1] = k[ice] * Tfrz - rprimew[ice]
    dev = np.zeros(2 * jmx)
    dev[2 * ice + 1] = -Cw_delt + k[ice]
    # Regularize the matrix slightly to improve conditioning
    Mt = M + np.diag(dev) + 1e-8 * np.eye(2 * jmx)
    I_full = np.linalg.solve(Mt, r)
    T = I_full.copy()
    T[2 * ice + 1] = np.minimum(Tfrz, I_full[2 * ice + 1])
    L = T[0::2]
    W = T[1::2]
    I_water = W.copy()
    Fnet = np.zeros(jmx)
    Fnet[ice] = (np.dot(Diff_Op[ice, :], I_water) - rprimew[ice] -
                 B * W[ice] - nu_fw[ice] * (I_water[ice] - L[ice]))
    nhice = ice[ice >= (jmx // 2 + 1)]
    shice = ice[ice < (jmx // 2)]
    nhocn = notice[notice >= (jmx // 2 + 1)]
    shocn = notice[notice < (jmx // 2)]
    nhicearea = np.sum(fw[nhice])
    shicearea = np.sum(fw[shice])
    nhmax = np.sum(fw[jmx // 2:])
    shmax = np.sum(fw[:jmx // 2])
    nhfw = 2 * min(2 - 2 * (nhicearea - delx) / nhmax, 2)
    shfw = 2 * min(2 - 2 * (shicearea - delx) / shmax, 2)
    Fnet[nhice] += nhfw
    Fnet[shice] += shfw
    nhocnarea = nhmax - nhicearea
    shocnarea = shmax - shicearea
    nhdW = nhfw * nhicearea / nhocnarea / Cw_delt
    shdW = shfw * shicearea / shocnarea / Cw_delt
    W[nhocn] -= nhdW
    W[shocn] -= shdW
    return T, L, W, Fnet

def seasonal_solar(xi, obl, ecc, lon):
    xi = np.asarray(xi)
    npts = len(xi)
    t1 = 2808 / 2.0754
    dr_conv = np.pi / 180.0
    rd_conv = 1 / dr_conv
    lmr = 0.0
    beta = np.sqrt(1 - ecc**2)
    lm0 = lmr - 2 * ((0.5 * ecc + 0.125 * ecc**3) * (1 + beta) * np.sin(lmr - lon * dr_conv) -
                     0.25 * ecc**2 * (0.5 + beta) * np.sin(2 * (lmr - lon * dr_conv)) +
                     0.125 * ecc**3 * ((1 / 3) + beta) * np.sin(3 * (lmr - lon * dr_conv)))
    it = 360
    lm = np.linspace(lm0, lm0 + 2 * np.pi - 2 * np.pi / it, it)
    calendarlongitude = (lm +
                         (2 * ecc - 0.25 * ecc**3) * np.sin(lm - lon * dr_conv) +
                         (5 / 4) * ecc**2 * np.sin(2 * (lm - lon * dr_conv)) +
                         (13 / 12) * ecc**3 * np.sin(3 * (lm - lon * dr_conv)))
    calendarlongitude = np.mod(calendarlongitude, 2 * np.pi)
    lme = np.concatenate((lm, lm, lm))
    calendarlongitudee = np.concatenate((calendarlongitude - 2 * np.pi,
                                          calendarlongitude,
                                          calendarlongitude + 2 * np.pi))
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
    dec = np.arcsin(s_delt) * (180 / np.pi)
    t_delt = s_delt / np.sqrt(1.0 - s_delt**2)
    wk = np.zeros((npts, nts))
    for i in range(nts):
        for j in range(npts):
            phi_val = np.arcsin(np.clip(xi[j], -1, 1)) * (180 / np.pi)
            if abs(dec[i]) > 0:
                if phi_val >= 90 - abs(dec[i]):
                    wk[j, i] = t1 * xi[j] * s_delt[i] / (distance[i]**2)
                elif ((-phi_val >= (90 - abs(dec[i]))) and (phi_val < 0)):
                    wk[j, i] = 0
                else:
                    c_h0 = -np.tan(np.deg2rad(phi_val)) * t_delt[i]
                    c_h0 = np.clip(c_h0, -1, 1)
                    h0 = np.arccos(c_h0)
                    wk[j, i] = t1 * (h0 * xi[j] * s_delt[i] +
                                     np.cos(np.deg2rad(phi_val)) * np.sqrt(1 - s_delt[i]**2) * np.sin(h0)) \
                              / (distance[i]**2 * np.pi)
            else:
                if phi_val >= (90 + abs(dec[i])):
                    wk[j, i] = 0
                elif ((-phi_val >= (90 + abs(dec[i]))) and (phi_val < 0)):
                    wk[j, i] = t1 * xi[j] * s_delt[i] / (distance[i]**2)
                else:
                    c_h0 = -np.tan(np.deg2rad(phi_val)) * t_delt[i]
                    c_h0 = np.clip(c_h0, -1, 1)
                    h0 = np.arccos(c_h0)
                    wk[j, i] = t1 * (h0 * xi[j] * s_delt[i] +
                                     np.cos(np.deg2rad(phi_val)) * np.sqrt(1 - s_delt[i]**2) * np.sin(h0)) \
                              / (np.pi * distance[i]**2)
    insol = wk
    return insol, dayoffset, dec

def albedo_seasonal(L, W, x, dec, star='G'):
    """
    Compute albedos with zenith-angle dependence.
    x: cell-center values (non-dimensional, used to compute phi = asin(x))
    dec: current declination (in degrees)
    L, W: land and water temperatures (°C)
    star: star type; default 'G'
    """
    phi_arr = np.arcsin(np.clip(x, -1, 1))  # in radians
    zenith = np.abs(phi_arr - (dec * np.pi / 180.0))  # in radians
    params = {
        'F': {'A_o': 0.32865, 'A_l': 0.41428, 'A_50': 0.53664},
        'G': {'A_o': 0.31948, 'A_l': 0.41484, 'A_50': 0.51363},
        'K': {'A_o': 0.30235, 'A_l': 0.40133, 'A_50': 0.47708},
        'M': {'A_o': 0.23372, 'A_l': 0.33165, 'A_50': 0.31546},
    }
    p = params.get(star, params['G'])
    A_o = p['A_o']
    A_l_param = p['A_l']
    A_50 = p['A_50']
    alb_w = A_o + 0.08 * (3 * np.sin(zenith)**2 - 1) / 2 - 0.05
    alb_l = A_l_param + 0.08 * (3 * np.sin(zenith)**2 - 1) / 2 + 0.05
    alb_w = np.where(W <= -2, A_50, alb_w)
    alb_l = np.where(L <= -2, A_50, alb_l)
    return alb_l, alb_w

def seasonal_setup():
    global jmx, runlength, scaleQ, A, B, Dmag, nu, Cl, Cw, coldstartflag, hadleyflag
    global albedoflag, obl, ecc, per, ice_model, land, casename, Tfrz, conduct, Lfice
    global ts, tf, nstepinyear, delt, nts, n_out, delx, x, xfull, phi, Toffset, L, W
    global D, fl, fw, insol, Cw_delt, Cl_delt, delt_Lf, nu_fw, nu_fl, lambda_, a, c, b
    global Diff_Op, bw, bl, Mw, Ml, M, invM, scaleQdef, dec
    defaults()
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
    jmx = 2 * (jmx // 2)
    Tfrz = -2
    conduct = 2
    Lfice = 9.8 * 83.5 / 50
    ts = 90
    tf = runlength - 0.25
    # Use 240 time steps per year.
    nstepinyear = 240
    delt = 1.0 / nstepinyear
    nts = int(np.floor(tf / delt))
    n_out = np.arange(1, nts + 1)
    delx = 2.0 / jmx
    x = np.linspace(-1 + delx, 1 - delx, jmx - 1)
    xfull = np.linspace(-1 + delx / 2, 1 - delx / 2, jmx)
    phi = np.arcsin(np.clip(xfull, -1, 1)) * 180 / np.pi
    if coldstartflag:
        Toffset = -40
    else:
        Toffset = 0.0
    L = 7.5 + 20 * (1 - 2 * xfull**2) + Toffset
    W = 7.5 + 20 * (1 - 2 * xfull**2) + Toffset
    if hadleyflag:
        D_x = Dmag * (1 + 9 * np.exp(-(x / np.sin(25 * np.pi / 180))**6))
    else:
        D_x = Dmag * np.ones_like(x)
    lambda_ = D_x / (delx**2) * (1 - x**2)
    a = np.concatenate((np.array([0]), -lambda_))
    c = np.concatenate((-lambda_, np.array([0])))
    b = -a - c
    Diff_Op = -(np.diag(b) + np.diag(c[:-1], 1) + np.diag(a[1:], -1))
    # Use the piecewise-defined land fraction (fl) with values between 0 and 1.
    fl = 0.38 * np.ones(jmx)
    jidx = phi <= -60
    fl[jidx] = 0.95
    jidx = (phi > -60) & (phi <= -40)
    fl[jidx] = 0.05
    jidx = (phi > -40) & (phi <= 20)
    fl[jidx] = 0.25
    jidx = (phi > 20) & (phi <= 70)
    fl[jidx] = 0.5
    fw = 1 - fl
    insol, dayoffset, dec = seasonal_solar(xfull, obl, ecc, per)
    insol = scaleQ * insol[:, np.concatenate(([359], np.arange(360)))]
    Cw_delt = Cw / delt
    Cl_delt = Cl / delt
    delt_Lf = delt / Lfice
    nu_fw = nu / fw
    nu_fl = nu / fl
    bw = Cw_delt + B + nu_fw - a - c
    bl = Cl_delt + B + nu_fl - a - c
    Mw = np.diag(bw) + np.diag(c[:-1], 1) + np.diag(a[1:], -1)
    Ml = np.diag(bl) + np.diag(c[:-1], 1) + np.diag(a[1:], -1)
    M = np.zeros((2 * jmx, 2 * jmx))
    for j in range(jmx):
        M[2 * j, 0::2] = Ml[j, :]
        M[2 * j, 2 * j] = -nu_fl[j]
        M[2 * j + 1, 1::2] = Mw[j, :]
        if j > 0:
            M[2 * j + 1, 2 * j - 1] = -nu_fw[j]
    invM = np.linalg.inv(M)

def plotoutput():
    plt.figure()
    plt.plot(Lann.mean(axis=0))
    plt.xlabel('Time step')
    plt.ylabel('Mean Land Temperature')
    plt.title('Annual Mean Land Temperature')
    plt.show()

def run_seasonal():
    global L, W, insol, ts, Cl_delt, Cw_delt, delt, Tfrz, conduct, M, Diff_Op, B, nu_fw, fw, delx, dec, phi
    r = np.zeros(2 * jmx)
    if ice_model:
        # Initial ice assignment based on water temperature:
        ice = np.where(W < Tfrz)[0]
        notice = np.where(W >= Tfrz)[0]
        h = np.zeros(jmx)
        h[ice] = 2.0
        if albedoflag:
            alb_l = None
            alb_w = None
        else:
            alb_l, alb_w = albedo_seasonal(L, W, phi, dec[ts - 1])
        S = insol[:, ts]
        rprimew = A - (1 - alb_w) * S
        rprimel = A - (1 - alb_l) * S
        r[0::2] = L * Cl_delt - rprimel
        r[1::2] = W * Cw_delt - rprimew
        T, L, W, Fnet = ice_balance(jmx, ice, notice, h, conduct, Tfrz, r, rprimew,
                                     Cw_delt, M, Diff_Op, B, nu_fw, fw, delx)
    else:
        h = np.zeros(jmx)
    L_out = np.zeros((jmx, len(n_out)))
    W_out = np.zeros((jmx, len(n_out)))
    if ice_model:
        h_out = np.zeros((jmx, len(n_out)))
    tday = np.zeros(nts)
    yr = np.zeros(nts, dtype=int)
    day = np.zeros(nts, dtype=int)
    idx_out = 0
    for i in range(nts):
        if ice_model:
            ice = np.where(h > 0.001)[0]
            notice = np.where(h <= 0.001)[0]
        n = i + 1
        tday[i] = ts + 3 + i * 360 * delt
        yr[i] = int(np.floor((tday[i] - 1) / 360))
        day[i] = int(np.floor(tday[i] - yr[i] * 360))
        alb_l, alb_w = albedo_seasonal(L, W, phi, dec[day[i] - 1])
        S = insol[:, day[i]]
        rprimew = A - (1 - alb_w) * S
        rprimel = A - (1 - alb_l) * S
        r[0::2] = L * Cl_delt - rprimel
        r[1::2] = W * Cw_delt - rprimew
        if ice_model:
            if ice.size > 0:
                T, L, W, Fnet = ice_balance(jmx, ice, notice, h, conduct, Tfrz, r, rprimew,
                                             Cw_delt, M, Diff_Op, B, nu_fw, fw, delx)
                Fnet = np.clip(Fnet, -100, 100)
                h[ice] = np.maximum(0.0, h[ice] - delt_Lf * Fnet[ice])
            water_temps = T[2 * notice + 1]
            cold = np.where(water_temps < Tfrz)[0]
            new_indices = notice[cold]
            if new_indices.size > 0:
                h[new_indices] = -Cw / Lfice * (W[new_indices] - Tfrz)
                W[new_indices] = Tfrz
        else:
            T = invM @ r
            L = T[0::2]
            W = T[1::2]
        if np.any(np.isnan(L)) or np.any(np.isnan(W)):
            print("Warning: NaNs encountered at time step", i)
            L = np.where(np.isnan(L), Tfrz, L)
            W = np.where(np.isnan(W), Tfrz, W)
        L = np.clip(L, -50, 50)
        W = np.clip(W, -50, 50)
        if n == n_out[idx_out]:
            L_out[:, idx_out] = L
            W_out[:, idx_out] = W
            if ice_model:
                h_out[:, idx_out] = h
            idx_out += 1
    days_arr = np.arange(nstepinyear - 1, -1, -1)
    tt = (L_out.shape[1] - 1) - days_arr
    thedays = np.array(day)[tt]
    Lann = L_out[:, tt]
    Wann = W_out[:, tt]
    Tg = (np.sum(L_out * fl[:, None], axis=0) + np.sum(W_out * fw[:, None], axis=0)) / jmx
    return {'L_out': L_out, 'W_out': W_out, 'tt': tt, 'thedays': thedays,
            'Lann': Lann, 'Wann': Wann, 'Tg': Tg}

# ============================================================================
# Outer loop: Vary scaleQ, run simulation, and compute ice line and average W
# ============================================================================
scaleQkeep = []
meanicelin = []
meanTg = []
avg_W_all = []  # to store avg_W for each run

# Loop for 5 values as in your MATLAB code
for ind in range(1, 6):
    scaleQ_value = 1.35 - (ind * 0.05)
    scaleQkeep.append(scaleQ_value)
    scaleQ = scaleQ_value
    seasonal_setup()
    result = run_seasonal()
    # Compute iceline using the last year's output for the northern hemisphere.
    j_nh = np.arange(jmx // 2, jmx)
    L_lastyear = result['L_out'][j_nh, :]
    W_lastyear = result['W_out'][j_nh, :]
    phi_nh = phi[j_nh]
    icelin = np.zeros(360)
    for daystep in range(360):
        # For each day, get the average water temperature profile
        avg_temp = (L_lastyear[:, daystep] + W_lastyear[:, daystep]) / 2
        # Add a tiny perturbation (like MATLAB does) to enforce strict monotonicity.
        # MATLAB subtracts 1e-6 * (grid index) + 1e-6 * (day index)
        pert = np.array([(i + 1) * 1e-6 for i in range(len(avg_temp))])
        avg_temp = avg_temp - pert - (daystep + 1) * 1e-6
        # Reverse the arrays so that the temperature becomes increasing
        avg_temp_profile = avg_temp[::-1]
        phi_profile = phi_nh[::-1]
        ice_day = np.interp(-2.013, avg_temp_profile, phi_profile, left=np.nan, right=np.nan)
        if np.isnan(ice_day):
            if np.min(avg_temp) < -2.013:
                ice_day = 0
            else:
                ice_day = 90
        icelin[daystep] = ice_day
    meanicelin_value = np.nanmean(icelin)
    meanicelin.append(meanicelin_value)
    tt = result['tt']
    Tg_sim = result['Tg']
    meanTg_value = np.nanmean(Tg_sim[tt])
    meanTg.append(meanTg_value)
    print("For scaleQ = %.4f, mean ice latitude = %.4f, temperature = %.4f" %
          (scaleQ_value, meanicelin_value, meanTg_value))

# Stack results into three columns: scaleQ, mean iceline, mean temperature (°C)
d_a = np.vstack((scaleQkeep, meanicelin, meanTg))
d_c = d_a.T
np.savetxt('F_dwarf_recal_blueice_co2_3bar_300yr_1perc.txt', d_c, delimiter='\t', fmt='%.4f\t%.4f\t%.4f')

plt.figure()
plt.subplot(211)
plt.plot(scaleQkeep, meanTg, marker='o')
plt.xlabel('Stellar Constant (scaleQ)')
plt.ylabel('Global Mean Temperature [°C]')
plt.subplot(212)
plt.plot(scaleQkeep, meanicelin, marker='o')
plt.xlabel('Stellar Constant (scaleQ)')
plt.ylabel('Mean Ice Line Latitude [°]')
plt.show()
