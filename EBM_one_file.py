import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.io import loadmat

# DEFAULT CONFIGURATION

DEFAULTS = {
    "jmx": 120,
    "runlength": 100,
    "scaleQ": 1.0,
    "A": 203.3,
    "B": 2.09,
    "Dmag": 0.44,
    "Toffset": -30.0,
    "obl": 0.0,
    "ecc": 0.0,
    "per": 102.07,
    "star": "G",
    "land": "Fillet",
    "casename": "Control",
    "hadleyflag": 1.0,
    "albedoflag": 0.0,
    "ice_model": 1.0,
    "coldstart": 0.0,
    "Cl": 0.45,
    "Cw": 9.8,
    "nu": 3.0,
    "rghflag": 0.0,
}



# BROADBAND ALBEDO PARAMETERS

def get_broadband_albedo(star: str):
    star = star.strip().upper()

    if star == 'F':
        Asnow = 0.66833
        A_75  = 0.59884
        A_50  = 0.53664
        A_25  = 0.47961
        A_bi  = 0.42542
        A_o   = 0.32865
        A_l   = 0.41428
        c_2k  = 0.905
        c_1   = 0.994
        c_2   = 0.993
        c_5   = 0.990
        c_20  = 0.983
        c_100 = 0.969
        c_200 = 0.960
        m     = 0.83
    elif star == 'G':
        Asnow = 0.64622
        A_75  = 0.57636
        A_50  = 0.51363
        A_25  = 0.45585
        A_bi  = 0.40093
        A_o   = 0.31948
        A_l   = 0.41484
        c_2k  = 0.901
        c_1   = 0.992
        c_2   = 0.991
        c_5   = 0.988
        c_20  = 0.979
        c_100 = 0.963
        c_200 = 0.954
        m     = 0.80
    elif star == 'K':
        Asnow = 0.60392
        A_75  = 0.53716
        A_50  = 0.47708
        A_25  = 0.42150
        A_bi  = 0.36870
        A_o   = 0.30235
        A_l   = 0.40133
        c_2k  = 0.886
        c_1   = 0.990
        c_2   = 0.988
        c_5   = 0.984
        c_20  = 0.974
        c_100 = 0.955
        c_200 = 0.944
        m     = 0.78
    elif star == 'M':
        Asnow = 0.39800
        A_75  = 0.35478
        A_50  = 0.31546
        A_25  = 0.28406
        A_bi  = 0.24317
        A_o   = 0.23372
        A_l   = 0.33165
        c_2k  = 0.790
        c_1   = 0.973
        c_2   = 0.970
        c_5   = 0.959
        c_20  = 0.936
        c_100 = 0.900
        c_200 = 0.881
        m     = 0.63
    else:
        raise ValueError("Invalid star type. Must be one of 'F', 'G', 'K', or 'M'.")

    return {
        'Asnow': Asnow,
        'A_75': A_75,
        'A_50': A_50,
        'A_25': A_25,
        'A_bi': A_bi,
        'A_o': A_o,
        'A_l': A_l,
        'c_2k': c_2k,
        'c_1': c_1,
        'c_2': c_2,
        'c_5': c_5,
        'c_20': c_20,
        'c_100': c_100,
        'c_200': c_200,
        'm': m,
    }


# ALBEDO FEEDBACK

def albedo_seasonal(L, W, x, A_o, A_l, A_50):
    alb_w = A_o + 0.08 * (3 * x**2 - 1) / 2 - 0.05
    alb_l = A_l + 0.08 * (3 * x**2 - 1) / 2 + 0.05

    idx_w = np.where(W <= -2)[0]
    idx_l = np.where(L <= -2)[0]
    alb_w[idx_w] = A_50
    alb_l[idx_l] = A_50

    return alb_l, alb_w



# SEASONAL INSOLATION

def sun(xi, obl, ecc, long, star):
    npts = len(xi)
    t1 = 2808 / 2.0754
    dr_conv = np.pi / 180.0
    rd_conv = 1.0 / dr_conv

    long_adj = long + 180.0
    fix = long_adj
    fws = (360.0 - long_adj + fix) * dr_conv
    while fws < 0:
        fws += 2 * np.pi
    while fws >= 2 * np.pi:
        fws -= 2 * np.pi

    cosE = (np.cos(fws) + ecc) / (1.0 + ecc * np.cos(fws))
    if fws < np.pi:
        Ews = np.arccos(cosE)
    else:
        Ews = 2 * np.pi - np.arccos(cosE)

    lm0 = Ews - ecc * np.sin(Ews) + long_adj * dr_conv
    while lm0 < 0:
        lm0 += 2 * np.pi
    while lm0 >= 2 * np.pi:
        lm0 -= 2 * np.pi

    it = 360
    lm = np.linspace(lm0, lm0 + 2 * np.pi - 2 * np.pi / it, it)
    if ecc <= 0.3:
        calendarlongitude = (
            lm
            + (2 * ecc - 0.25 * ecc**3) * np.sin(lm - long_adj * dr_conv)
            + (5 / 4) * (ecc**2) * np.sin(2 * (lm - long_adj * dr_conv))
            + (13 / 12) * (ecc**3) * np.sin(3 * (lm - long_adj * dr_conv))
        )
    else:
        calendarlongitude = np.zeros_like(lm)
        for i in range(len(lm)):
            MA = lm[i] - long_adj * dr_conv
            EA = MA + np.sign(np.sin(MA)) * 0.85 * ecc
            di_3 = 1.0
            while abs(di_3) > 1e-15:
                fi = EA - ecc * np.sin(EA) - MA
                fi_1 = 1.0 - ecc * np.cos(EA)
                fi_2 = ecc * np.sin(EA)
                fi_3 = ecc * np.cos(EA)
                di_1 = -fi / fi_1
                di_2 = -fi / (fi_1 + 0.5 * di_1 * fi_2)
                di_3 = -fi / (fi_1 + 0.5 * di_2 * fi_2 + (1.0 / 6.0) * di_2**2 * fi_3)
                EA += di_3
            while EA >= 2 * np.pi:
                EA -= 2 * np.pi
            while EA < 0:
                EA += 2 * np.pi
            if EA > np.pi:
                calendarlongitude[i] = (
                    2 * np.pi - np.arccos((np.cos(EA) - ecc) / (1.0 - ecc * np.cos(EA)))
                ) + long_adj * dr_conv
            else:
                calendarlongitude[i] = np.arccos((np.cos(EA) - ecc) / (1.0 - ecc * np.cos(EA))) + long_adj * dr_conv

    for n in range(len(calendarlongitude)):
        if calendarlongitude[n] >= 2 * np.pi:
            calendarlongitude[n] -= 2 * np.pi
        elif calendarlongitude[n] < 0:
            calendarlongitude[n] += 2 * np.pi

    nts = 360
    ti = calendarlongitude * rd_conv
    distance = (1 - ecc**2) / (1 + ecc * np.cos(dr_conv * (ti - long_adj)))

    s_delt = np.sin(dr_conv * obl) * np.sin(dr_conv * ti)
    c_delt = np.sqrt(np.maximum(1.0 - s_delt**2, 0.0))
    t_delt = np.divide(s_delt, c_delt, out=np.zeros_like(s_delt), where=c_delt != 0)
    delt = np.arcsin(s_delt) * rd_conv

    phi = np.arcsin(xi) * rd_conv
    wk = np.zeros((npts, nts))

    for i in range(nts):
        for j in range(npts):
            if delt[i] > 0.0:
                if phi[j] >= 90 - delt[i]:
                    wk[j, i] = t1 * xi[j] * s_delt[i] / (distance[i]**2)
                elif ((-phi[j] >= (90 - delt[i])) and (phi[j] < 0)):
                    wk[j, i] = 0.0
                else:
                    c_h0 = -np.tan(dr_conv * phi[j]) * t_delt[i]
                    c_h0 = np.clip(c_h0, -1.0, 1.0)
                    h0 = np.arccos(c_h0)
                    wk[j, i] = t1 * (
                        h0 * xi[j] * s_delt[i] + np.cos(dr_conv * phi[j]) * c_delt[i] * np.sin(h0)
                    ) / (distance[i]**2 * np.pi)
            else:
                if phi[j] >= (90 + delt[i]):
                    wk[j, i] = 0.0
                elif ((-phi[j] >= (90 + delt[i])) and (phi[j] < 0)):
                    wk[j, i] = t1 * xi[j] * s_delt[i] / (distance[i]**2)
                else:
                    c_h0 = -np.tan(dr_conv * phi[j]) * t_delt[i]
                    c_h0 = np.clip(c_h0, -1.0, 1.0)
                    h0 = np.arccos(c_h0)
                    wk[j, i] = t1 * (
                        h0 * xi[j] * s_delt[i] + np.cos(dr_conv * phi[j]) * c_delt[i] * np.sin(h0)
                    ) / (np.pi * distance[i]**2)

    insol = wk
    return insol, distance, delt



# ICE BALANCE

def icebalance(jmx, ice, notice, conduct, h, Tfrz, rprimew, Cw_delt, M, r, Diff_Op, B, nu_fw, fw, delx, Cw, W):
    k = np.zeros(jmx)
    Fnet = np.zeros(jmx)

    r_ice = 2 * ice + 1
    k[ice] = conduct / np.maximum(h[ice], 1e-12)
    r[r_ice] = k[ice] * Tfrz - rprimew[ice]

    dev = np.zeros(2 * jmx)
    dev[r_ice] = -Cw_delt + k[ice]
    Mt = M + np.diag(dev)

    I = np.linalg.solve(Mt, r)
    T = I.copy()
    T[r_ice] = np.minimum(Tfrz, I[r_ice])

    L = T[0::2]
    W_new = T[1::2]
    I_ocean = I[1::2]

    if len(ice) > 0:
        Fnet[ice] = (Diff_Op[ice, :] @ I_ocean) - rprimew[ice] - B * W_new[ice] - nu_fw[ice] * (I_ocean[ice] - L[ice])

    threshold = jmx // 2
    nhice = ice[ice >= threshold]
    shice = ice[ice < threshold]
    nhocn = notice[notice >= threshold]
    shocn = notice[notice < threshold]

    nhicearea = np.sum(fw[nhice]) if len(nhice) else 0.0
    shicearea = np.sum(fw[shice]) if len(shice) else 0.0
    nhmax = np.sum(fw[threshold:jmx])
    shmax = np.sum(fw[0:threshold])

    nhfw = 2 * min(2 - 2 * (nhicearea - delx) / nhmax, 2) if nhmax > 0 else 0.0
    shfw = 2 * min(2 - 2 * (shicearea - delx) / shmax, 2) if shmax > 0 else 0.0

    if len(nhice):
        Fnet[nhice] += nhfw
    if len(shice):
        Fnet[shice] += shfw

    nhocnarea = nhmax - nhicearea
    shocnarea = shmax - shicearea

    if len(nhocn) and nhocnarea > 0:
        nhdW = nhfw * nhicearea / nhocnarea / Cw_delt
        W_new[nhocn] = W_new[nhocn] - nhdW
    if len(shocn) and shocnarea > 0:
        shdW = shfw * shicearea / shocnarea / Cw_delt
        W_new[shocn] = W_new[shocn] - shdW

    return T, L, W_new, Fnet



# SETUP HELPERS

def build_land_fraction(phi, land, jmx):
    fl = 0.05 * np.ones(jmx)
    key = land.strip().lower()

    if key.startswith('preca'):
        j = np.where(phi <= -45)[0]
        fl[j] = 0.3
        j = np.where((phi > -45) & (phi <= 30))[0]
        fl[j] = 0.5
    elif key.startswith('ordov'):
        j = np.where(phi <= -60)[0]
        fl[j] = 0.95
        j = np.where((phi > -60) & (phi <= 70))[0]
        fl[j] = 0.3
    elif key.startswith('symme'):
        fl[:] = 0.34
    elif key.startswith('aquap'):
        fl[:] = 0.01
    elif key.startswith('landp'):
        fl[:] = 0.99
    elif key.startswith('fillet'):
        fl[:] = 0.25
    elif key.startswith('modern') or key.startswith('earth') or key.startswith('cont'):
        lat_nodes = np.array([-90, -60, -40, 20, 70, 90], dtype=float)
        fl_nodes  = np.array([0.95, 0.95, 0.05, 0.25, 0.50, 0.38], dtype=float)
        fl = np.interp(phi, lat_nodes, fl_nodes)
    else:
        fl = 0.38 * np.ones(jmx)
        j = np.where(phi <= -60)[0]
        fl[j] = 0.95
        j = np.where((phi > -60) & (phi <= -40))[0]
        fl[j] = 0.05
        j = np.where((phi > -40) & (phi <= 20))[0]
        fl[j] = 0.25
        j = np.where((phi > 20) & (phi <= 70))[0]
        fl[j] = 0.5

    return fl


def build_setup(cfg):
    jmx = int(cfg['jmx'])
    jmx = 2 * (jmx // 2)

    runlength = float(cfg['runlength'])
    scaleQ = float(cfg['scaleQ'])
    A = float(cfg['A'])
    B = float(cfg['B'])
    Dmag = float(cfg['Dmag'])
    nu = float(cfg['nu'])
    Cl = float(cfg['Cl'])
    Cw = float(cfg['Cw'])
    coldstartflag = float(cfg['coldstart'])
    hadleyflag = float(cfg['hadleyflag'])
    albedoflag = float(cfg['albedoflag'])
    obl = float(cfg['obl'])
    ecc = float(cfg['ecc'])
    per = float(cfg['per'])
    star = cfg['star'].strip().upper()
    ice_model = float(cfg['ice_model'])
    land = cfg['land']
    casename = cfg['casename']
    rghflag = float(cfg.get('rghflag', 0.0))

    Tfrz = -2.0
    conduct = 2.0
    Lfice = 9.8 * 83.5 / 50.0

    ts = 90
    tf = runlength - 0.25
    nstepinyear = 360
    delt = 1.0 / nstepinyear
    nts = int(np.floor(tf / delt))
    n_out = np.arange(1, nts + 1)

    delx = 2.0 / jmx
    x = np.arange(-1.0 + delx, 1.0, delx)
    xfull = np.arange(-1 + delx / 2, 1, delx)
    phi = np.arcsin(np.clip(xfull, -1.0, 1.0)) * 180.0 / np.pi

    Toffset = -40.0 if coldstartflag else 0.0
    L = 7.5 + 20 * (1 - 2 * xfull**2) + Toffset
    W = 7.5 + 20 * (1 - 2 * xfull**2) + Toffset

    if hadleyflag:
        D = Dmag * (1 + 9 * np.exp(- (x / np.sin(25 * np.pi / 180.0))**6))
    else:
        D = Dmag * np.ones_like(x)

    fl = build_land_fraction(phi, land, jmx)
    fw = 1.0 - fl

    insol, distance, delt_arr = sun(xfull, obl, ecc, per, star)
    insol = scaleQ * np.concatenate((insol[:, -1][:, np.newaxis], insol[:, :-1]), axis=1)

    Cw_delt = Cw / delt
    Cl_delt = Cl / delt
    delt_Lf = delt / Lfice

    fw_safe = np.maximum(fw, 1e-6)
    fl_safe = np.maximum(fl, 1e-6)
    nu_fw = nu / fw_safe
    nu_fl = nu / fl_safe

    lam = D / (delx**2) * (1 - x**2)
    a = np.concatenate(([0], -lam))
    c = np.concatenate((-lam, [0]))
    b = -a - c

    Diff_Op = -(np.diag(b[:jmx]) + np.diag(c[:jmx - 1], k=1) + np.diag(a[1:jmx], k=-1))

    bw = Cw_delt + B + nu_fw - (a[:jmx] + c[:jmx])
    bl = Cl_delt + B + nu_fl - (a[:jmx] + c[:jmx])

    Mw = np.diag(bw) + np.diag(c[:jmx - 1], k=1) + np.diag(a[1:jmx], k=-1)
    Ml = np.diag(bl) + np.diag(c[:jmx - 1], k=1) + np.diag(a[1:jmx], k=-1)

    M = np.zeros((2 * jmx, 2 * jmx))
    for j in range(jmx):
        M[2 * j, 0:2 * jmx:2] = Ml[j, :]
        M[2 * j, 2 * j + 1] = -nu_fl[j]
        M[2 * j + 1, 1:2 * jmx:2] = Mw[j, :]
        M[2 * j + 1, 2 * j] = -nu_fw[j]

    broadband_params = get_broadband_albedo(star)
    A_o = broadband_params['A_o']
    A_l = broadband_params['A_l']
    A_50 = broadband_params['A_50']

    clim_alb_l = None
    clim_alb_w = None
    thedays = None
    if albedoflag:
        try:
            temps = loadmat('temperatures.mat')
            thedays = temps['thedays'].squeeze()
            Lann = temps['Lann']
            Wann = temps['Wann']
            clim_alb_l = np.zeros((jmx, 360))
            clim_alb_w = np.zeros((jmx, 360))
            n = 0
            for t in thedays:
                alb_l, alb_w = albedo_seasonal(Lann[:, n], Wann[:, n], xfull, A_o, A_l, A_50)
                idx = int(t) - 1
                clim_alb_l[:, idx] = alb_l
                clim_alb_w[:, idx] = alb_w
                n += 1
        except FileNotFoundError:
            clim_alb_l = None
            clim_alb_w = None
            thedays = None

    return {
        'cfg': cfg.copy(),
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
        'rghflag': rghflag,
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
        'insol_base': insol.copy(),
        'distance': distance,
        'delt_arr': delt_arr,
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
        'clim_alb_l': clim_alb_l,
        'clim_alb_w': clim_alb_w,
        'thedays': thedays,
        'A_o': A_o,
        'A_l': A_l,
        'A_50': A_50,
    }



# MAIN SEASONAL SOLVER

def seasonal_run(cfg=None):
    if cfg is None:
        cfg = DEFAULTS.copy()
    else:
        merged = DEFAULTS.copy()
        merged.update(cfg)
        cfg = merged

    setup_data = build_setup(cfg)

    current_scaleQ = cfg['scaleQ']
    setup_data['insol'] = current_scaleQ * setup_data['insol_base']

    jmx = setup_data['jmx']
    nts = setup_data['nts']
    nstepinyear = setup_data['nstepinyear']
    delt = setup_data['delt']
    ts = setup_data['ts']
    n_out = setup_data['n_out']
    insol = setup_data['insol']
    A = setup_data['A']
    Cl_delt = setup_data['Cl_delt']
    Cw_delt = setup_data['Cw_delt']
    Tfrz = setup_data['Tfrz']
    albedoflag = setup_data['albedoflag']
    L = setup_data['L'].copy()
    W = setup_data['W'].copy()
    xfull = setup_data['xfull']
    rghflag = setup_data.get('rghflag', 0)
    ice_model = bool(setup_data['ice_model'])
    B = setup_data['B']
    nu_fw = setup_data['nu_fw']
    fw = setup_data['fw']
    delt_Lf = setup_data['delt_Lf']
    Lfice = setup_data['Lfice']
    Cw = setup_data['Cw']
    Diff_Op = setup_data['Diff_Op']
    M = setup_data['M']
    h = np.zeros(jmx)
    clim_alb_l = setup_data.get('clim_alb_l', None)
    clim_alb_w = setup_data.get('clim_alb_w', None)
    thedays_setup = setup_data.get('thedays', None)
    conduct = setup_data.get('conduct', 2.0)
    delx = setup_data['delx']

    A_o = setup_data['A_o']
    A_l = setup_data['A_l']
    A_50 = setup_data['A_50']

    r = np.zeros(2 * jmx)

    if ice_model:
        ice = np.where(W < Tfrz)[0]
        notice = np.where(W >= Tfrz)[0]
        h = np.zeros(jmx)
        h[ice] = 2.0

        if albedoflag and (clim_alb_l is not None) and (thedays_setup is not None):
            alb_l = clim_alb_l[:, int(thedays_setup[0]) - 1]
            alb_w = clim_alb_w[:, int(thedays_setup[0]) - 1]
        else:
            alb_l, alb_w = albedo_seasonal(L, W, xfull, A_o, A_l, A_50)

        S = insol[:, int(ts) - 1]
        rprimel = A - ((1 - alb_l) * S)
        rprimew = A - ((1 - alb_w) * S)
        r[0::2] = L * Cl_delt - rprimel
        r[1::2] = W * Cw_delt - rprimew

        T, L, W, Fnet = icebalance(
            jmx, ice, notice, conduct, h, Tfrz, rprimew,
            Cw_delt, M, r, Diff_Op, B, nu_fw, fw, delx, Cw, W
        )
    else:
        ice = np.array([], dtype=int)

    num_out = len(n_out)
    L_out = np.zeros((jmx, num_out))
    W_out = np.zeros((jmx, num_out))
    h_out = np.zeros((jmx, num_out)) if ice_model else None
    alb_l_out = np.zeros((jmx, num_out))
    alb_w_out = np.zeros((jmx, num_out))

    tday = np.zeros(nts)
    yr = np.zeros(nts)
    day = np.zeros(nts, dtype=int)
    idx_out = 0

    for n in range(1, nts + 1):
        tday[n - 1] = ts + 2 + 1 + (n - 1) * 360 * delt
        yr[n - 1] = np.floor((-1 + tday[n - 1]) / 360)
        day[n - 1] = int(np.floor(tday[n - 1] - yr[n - 1] * 360))
        day_idx = max(1, min(day[n - 1], 360))

        if albedoflag and (clim_alb_l is not None):
            nn = day_idx - int(360 * delt)
            if nn < 0:
                nn = int(360 - 360 * delt * 0.5)
            nn = max(1, min(nn, 360))
            alb_l = clim_alb_l[:, nn - 1]
            alb_w = clim_alb_w[:, nn - 1]
        else:
            alb_l, alb_w = albedo_seasonal(L, W, xfull, A_o, A_l, A_50)

        S = insol[:, day_idx - 1]
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
            ice = np.where(h > 0.001)[0]
            notice = np.where(h <= 0.001)[0]
            T, L, W, Fnet = icebalance(
                jmx, ice, notice, conduct, h, Tfrz, rprimew,
                Cw_delt, M, r, Diff_Op, B, nu_fw, fw, delx, Cw, W
            )
            if len(ice):
                h[ice] = np.maximum(0.0, h[ice] - delt_Lf * Fnet[ice])

            T_ocean = T[1::2]
            cold = np.where(T_ocean[notice] < Tfrz)[0]
            new = notice[cold]
            if len(new) > 0:
                h[new] = -Cw / Lfice * (W[new] - Tfrz)
                W[new] = Tfrz
        else:
            T = np.linalg.solve(M, r)
            L = T[0::2]
            W = T[1::2]

        if idx_out < num_out and n == n_out[idx_out]:
            L_out[:, idx_out] = L
            W_out[:, idx_out] = W
            alb_l_out[:, idx_out] = alb_l
            alb_w_out[:, idx_out] = alb_w
            if ice_model:
                h_out[:, idx_out] = h
            idx_out += 1

    thedays = ((day[:num_out] - 1) % 360) + 1
    if num_out >= nstepinyear:
        Lann = L_out[:, -nstepinyear:]
        Wann = W_out[:, -nstepinyear:]
        alb_l_ann = alb_l_out[:, -nstepinyear:]
        alb_w_ann = alb_w_out[:, -nstepinyear:]
        thedays_ann = thedays[-nstepinyear:]
        if ice_model:
            h_ann = h_out[:, -nstepinyear:]
        else:
            h_ann = None
    else:
        Lann = L_out
        Wann = W_out
        alb_l_ann = alb_l_out
        alb_w_ann = alb_w_out
        thedays_ann = thedays
        h_ann = h_out

    return {
        'setup': setup_data,
        'L_out': L_out,
        'W_out': W_out,
        'h_out': h_out,
        'alb_l_out': alb_l_out,
        'alb_w_out': alb_w_out,
        'thedays': thedays_ann,
        'Lann': Lann,
        'Wann': Wann,
        'alb_l_ann': alb_l_ann,
        'alb_w_ann': alb_w_ann,
        'h_ann': h_ann,
        'final_L': L,
        'final_W': W,
        'final_h': h,
    }



# DIAGNOSTICS

def annual_means(results):
    return final_year_annual_means(results)

def final_year_annual_means(results):
    """
    Annual means over the FINAL orbit only.
    This is the quantity FILLET expects.
    """
    setup = results['setup']
    fl = setup['fl']
    fw = setup['fw']
    lat = setup['phi']

    # --- annual mean temperature ---
    T_land = np.mean(results['Lann'], axis=1)
    T_ocean = np.mean(results['Wann'], axis=1)
    T_avg = fl * T_land + fw * T_ocean
    Tglob = np.mean(T_avg) +273.15

    # --- annual mean albedo ---
    # IMPORTANT: average the instantaneous albedo over the year
    # do NOT recompute albedo from annual-mean temperature
    A_land = np.mean(results['alb_l_ann'], axis=1)
    A_ocean = np.mean(results['alb_w_ann'], axis=1)
    A_avg = fl * A_land + fw * A_ocean

    return {
        'lat': lat,
        'T_land': T_land,
        'T_ocean': T_ocean,
        'T_avg': T_avg,
        'Tglob': Tglob,
        'A_land': A_land,
        'A_ocean': A_ocean,
        'A_avg': A_avg,
    }


def mean_iceline(results, threshold=-2.013):
    setup = results['setup']
    jmx = setup['jmx']
    phi = np.linspace(-90, 90, jmx)
    delt = setup['delt']

    W_out = results['W_out']
    h_out = results['h_out']
    if h_out is None:
        return np.nan

    dummy_n = W_out.shape[1]
    days = np.arange((1 / delt) - 1, -1, -1)
    tt = dummy_n - days - 1
    tt_int = tt.astype(int)

    j_nh = np.arange(int(jmx / 2), jmx)
    W_lastyear = W_out[np.ix_(j_nh, tt_int)].copy()

    ii = np.arange(W_lastyear.shape[0])[:, None]
    jj = np.arange(W_lastyear.shape[1])[None, :]
    W_lastyear -= 0.000001 * (ii + 1) + 0.000001 * (jj + 1)

    icelin = np.empty(W_lastyear.shape[1])
    for daystep in range(W_lastyear.shape[1]):
        x_vals = W_lastyear[:, daystep]
        y_vals = phi[j_nh]
        f_interp = interp1d(x_vals, y_vals, bounds_error=False, fill_value=np.nan)
        ic_val = f_interp(threshold)
        if np.isnan(ic_val):
            if np.min(x_vals) < threshold:
                ic_val = 0
            elif np.min(x_vals) > threshold:
                ic_val = 90
        icelin[daystep] = ic_val

    return float(np.nanmean(icelin))


#WARMSTART 

def warmstart_sweep(cfg=None, scaleQ_values=None, save_txt=False, make_plot=True):
    if cfg is None:
        cfg = DEFAULTS.copy()
    else:
        merged = DEFAULTS.copy()
        merged.update(cfg)
        cfg = merged

    if scaleQ_values is None:
        scaleQ_values = [1.35 - (i * 0.05) for i in range(1, 22)]

    base_setup = build_setup(cfg)
    fl = base_setup['fl']
    fw = base_setup['fw']
    delt = base_setup['delt']

    scaleQkeep = []
    meanicelin_list = []
    meanTg_list = []

    for scaleQ_i in scaleQ_values:
        case_cfg = cfg.copy()
        case_cfg['scaleQ'] = float(scaleQ_i)
        scaleQkeep.append(scaleQ_i)

        results = seasonal_run(case_cfg)
        L_out = results['L_out']
        W_out = results['W_out']

        Tg = (np.dot(L_out.T, fl) + np.dot(W_out.T, fw)) / base_setup['jmx']

        dummy_n = L_out.shape[1]
        days = np.arange((1 / delt) - 1, -1, -1)
        tt = dummy_n - days - 1
        tt_int = tt.astype(int)

        mean_icelin = mean_iceline(results)
        meanicelin_list.append(mean_icelin)
        meanTg_list.append(float(np.mean(Tg[tt_int])))

    d_a = np.array([scaleQkeep, meanicelin_list, meanTg_list])
    d_c = d_a.T

    if save_txt:
        np.savetxt('G_dwarf_ws.txt', d_c, delimiter='\t', fmt='%.6f')

    if make_plot:
        plt.figure(figsize=(10, 8))
        plt.subplot(2, 1, 1)
        plt.plot(scaleQkeep, meanTg_list, 'o-', markersize=6)
        plt.ylabel('Mean Temperature')
        plt.title('Mean Temperature vs. scaleQ')
        plt.grid(True)

        plt.subplot(2, 1, 2)
        plt.plot(scaleQkeep, meanicelin_list, 's-', markersize=6)
        plt.xlabel('scaleQ')
        plt.ylabel('Mean Ice Line')
        plt.title('Mean Ice Line vs. scaleQ')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return {
        'scaleQ': np.array(scaleQkeep),
        'mean_iceline': np.array(meanicelin_list),
        'mean_Tg': np.array(meanTg_list),
    }



# EXAMPLE MAIN

def main():
    # Run EBM with warmstart sweep over scaleQ values
    print("Running EBM warmstart sweep over scaleQ values...")
    print("=" * 60)
    
    cfg = DEFAULTS.copy()
    cfg['runlength'] = 50  # Reduce runlength for faster computation
    
    # Define scaleQ range (fewer points for demo, but still shows the relationship)
    scaleQ_values = [1.30 - (i * 0.10) for i in range(8)]  # Reduced from 21 to 8 values
    
    results_sweep = warmstart_sweep(cfg, scaleQ_values=scaleQ_values, 
                                     save_txt=False, make_plot=False)
        # Save results to text file
    print("\nSaving results to text file...")
    output_data = np.column_stack([results_sweep['scaleQ'], results_sweep['mean_iceline']])
    np.savetxt('scaleQ_vs_iceline.txt', output_data, 
            fmt='%.6f', delimiter='\t', 
            header='scaleQ\tIceline_Latitude', comments='')
    print("Data saved to 'scaleQ_vs_iceline.txt'")
    
    # Plot scaleQ vs iceline latitude
    print("\nGenerating plot of scaleQ vs iceline latitude...")
    plt.figure(figsize=(12, 8))
    
    # Main plot
    plt.plot(results_sweep['scaleQ'], results_sweep['mean_iceline'], 
             'o-', markersize=8, linewidth=2.5, color='steelblue', 
             label='Mean Iceline Latitude')
    
    plt.xlabel('scaleQ (Solar Insolation Scaling Factor)', fontsize=14, fontweight='bold')
    plt.ylabel('Iceline Latitude [degrees]', fontsize=14, fontweight='bold')
    plt.title('Energy Balance Model: Iceline Latitude vs. Solar Insolation', 
              fontsize=16, fontweight='bold')
    
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(fontsize=12, loc='best')
    
    # Add some statistics
    plt.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('scaleQ_vs_iceline.png', dpi=300, bbox_inches='tight')
    print("Plot saved as 'scaleQ_vs_iceline.png'")
    plt.show()
    
    # Print summary
    print("\n" + "=" * 60)
    print("WARMSTART SWEEP SUMMARY")
    print("=" * 60)
    for i, (sq, ice) in enumerate(zip(results_sweep['scaleQ'], results_sweep['mean_iceline'])):
        if not np.isnan(ice):
            print(f"scaleQ = {sq:.4f}  →  Iceline Latitude = {ice:7.3f}°")
        else:
            print(f"scaleQ = {sq:.4f}  →  Iceline Latitude = NaN (no ice or all-ice)")
    
    print("=" * 60)


if __name__ == '__main__':
    main()
