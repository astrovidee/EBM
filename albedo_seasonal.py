"""
albedo_seasonal module
======================

This module provides a function to recalculate albedo values for water and land
based on input temperature arrays and spatial parameters. It adjusts the albedo
for regions with temperatures below a threshold.

Functions:
    albedo_seasonal(L, W, x, A_o, A_l, A_50)
        Recalculates the albedo for water and land.
"""

from get_broadband_albedo import get_broadband_albedo
import numpy as np

def albedo_seasonal(L, W, x, A_o, A_l, A_50):
    """
    Recalculate albedo for water and land.

    The function computes the albedo by applying a polynomial correction factor
    to a base albedo parameter. For temperatures less than or equal to -2, the
    albedo is set to a given threshold (A_50).

    Parameters:
        L (array-like): Array of land temperatures (or similar variable).
        W (array-like): Array of water temperatures (or similar variable).
        x (array-like): Array representing spatial parameters (e.g., latitude).
        A_o (float): Base water albedo parameter.
        A_l (float): Base land albedo parameter.
        A_50 (float): Albedo value when temperature <= -2.

    Returns:
        tuple: (alb_l, alb_w) where:
            alb_l (array-like): Calculated albedo for land.
            alb_w (array-like): Calculated albedo for water.
    """
    # Compute albedo based on a Legendre polynomial-like term (3*x^2 - 1)
    alb_w = A_o + 0.08 * (3 * x**2 - 1) / 2 - 0.05
    alb_l = A_l + 0.08 * (3 * x**2 - 1) / 2 + 0.05

    # For indices where water or land temperature is <= -2, set albedo to A_50
    idx_w = np.where(W <= -2)[0]
    idx_l = np.where(L <= -2)[0]
    alb_w[idx_w] = A_50
    alb_l[idx_l] = A_50

    return alb_l, alb_w
