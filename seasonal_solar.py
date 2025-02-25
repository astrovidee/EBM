"""
    Computes insolation, orbital distance, and declination for an eccentric planet,
    using formulas from Berger/Pan (JAS, 35, 1978).
    
    Parameters:
        xi   : 1D array of sin(latitude) values.
        obl  : Obliquity in degrees.
        ecc  : Orbital eccentricity.
        long : Longitude in degrees.
        star : Stellar type (e.g., 'F', 'G', 'K', or 'M') used to get broadband albedo parameters.
        
    Returns:
        insol    : 2D array (npts x nts) of insolation (W/m²).
        distance : 1D array of normalized orbital distances.
        delt     : 1D array of declination values (in degrees).
    """
import numpy as np
from get_broadband_albedo import get_broadband_albedo
from albedo_seasonal import albedo_seasonal

def sun(xi, obl, ecc, long, star):

    npts = len(xi)
    t1 = 2808 / 2.0754
    dr_conv = np.pi / 180.0
    rd_conv = 1 / dr_conv

    # Adjust longitude by 180 degrees
    long_adj = long + 180.0

    # True anomaly at winter solstice (Northern hemisphere); time = 0
    fix = long_adj  # (Change this value to 0, 90, etc. to shift the starting point)
    fws = (360.0 - long_adj + fix) * dr_conv
    while fws < 0:
        fws += 2 * np.pi
    while fws >= 2 * np.pi:
        fws -= 2 * np.pi

    # Get the eccentric anomaly at winter solstice
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

    beta = np.sqrt(1 - ecc**2)

    # Get the calendar dates (mean longitudes) for one orbital cycle.
    it = 360
    lm = np.linspace(lm0, lm0 + 2 * np.pi - 2 * np.pi / it, it)
    if ecc <= 0.3:
        calendarlongitude = (lm +
                              (2 * ecc - 0.25 * ecc**3) * np.sin(lm - long_adj * dr_conv) +
                              (5 / 4) * (ecc**2) * np.sin(2 * (lm - long_adj * dr_conv)) +
                              (13 / 12) * (ecc**3) * np.sin(3 * (lm - long_adj * dr_conv)))
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
                calendarlongitude[i] = (2 * np.pi - np.arccos((np.cos(EA) - ecc) / (1.0 - ecc * np.cos(EA)))) + long_adj * dr_conv
            else:
                calendarlongitude[i] = np.arccos((np.cos(EA) - ecc) / (1.0 - ecc * np.cos(EA))) + long_adj * dr_conv

    # Adjust calendarlongitude to be within [0, 2*pi)
    for n in range(len(calendarlongitude)):
        if calendarlongitude[n] >= 2 * np.pi:
            calendarlongitude[n] -= 2 * np.pi
        elif calendarlongitude[n] < 0:
            calendarlongitude[n] += 2 * np.pi

    nts = 360
    ti = calendarlongitude * rd_conv  # Convert from radians to degrees

    distance = (1 - ecc**2) / (1 + ecc * np.cos(dr_conv * (ti - long_adj)))
    # (The MATLAB version saves distance.mat; in Python, we simply keep the variable.)

    s_delt = np.sin(dr_conv * obl) * np.sin(dr_conv * ti)
    c_delt = np.sqrt(1.0 - s_delt**2)
    t_delt = s_delt / c_delt
    delt = np.arcsin(s_delt) * rd_conv  # Declination in degrees

    phi = np.arcsin(xi) * rd_conv        # Convert sin(latitude) to latitude in degrees
    wk = np.zeros((npts, nts))

    # Loop over latitudes and days to compute insolation
    for i in range(nts):
        for j in range(npts):
            if delt[i] > 0.0:
                if phi[j] >= 90 - delt[i]:
                    wk[j, i] = t1 * xi[j] * s_delt[i] / (distance[i]**2)
                elif ((-phi[j] >= (90 - delt[i])) and (phi[j] < 0)):
                    wk[j, i] = 0.0
                else:
                    c_h0 = -np.tan(dr_conv * phi[j]) * t_delt[i]
                    h0 = np.arccos(c_h0)
                    wk[j, i] = t1 * (h0 * xi[j] * s_delt[i] + np.cos(dr_conv * phi[j]) * c_delt[i] * np.sin(h0)) / (distance[i]**2 * np.pi)
            else:
                if phi[j] >= (90 + delt[i]):
                    wk[j, i] = 0.0
                elif ((-phi[j] >= (90 + delt[i])) and (phi[j] < 0)):
                    wk[j, i] = t1 * xi[j] * s_delt[i] / (distance[i]**2)
                else:
                    c_h0 = -np.tan(dr_conv * phi[j]) * t_delt[i]
                    h0 = np.arccos(c_h0)
                    wk[j, i] = t1 * (h0 * xi[j] * s_delt[i] + np.cos(dr_conv * phi[j]) * c_delt[i] * np.sin(h0)) / (np.pi * distance[i]**2)
                    
    insol = wk

    # ---- Using inputs from broadband albedo and albedo_seasonal ----
    # Get broadband albedo parameters for the given star type.
    broadband_params = get_broadband_albedo(star)
    # For example, extract water and land base albedo values.
    A_o = broadband_params['A_o']
    A_l = broadband_params['A_l']
    A_50 = broadband_params['A_50']
    
    # Now, if you have temperature fields L and W and an array x (for example, related to sin(latitude))
    # you could call albedo_seasonal to compute the albedo.
    # For demonstration, suppose we set:
    # (Note: In a real application, L and W would be your temperature arrays.)
    # L_temp = np.linspace(-10, 40, npts)
    # W_temp = np.linspace(-10, 40, npts)
    # x_arr = np.linspace(-1, 1, npts)
    # alb_l, alb_w = albedo_seasonal(L_temp, W_temp, x_arr, A_o, A_l, A_50)
    #
    # You can then use these albedo values in further energy balance or radiative calculations.
    
    return insol, distance, delt


