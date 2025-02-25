"""
albedo_seasonal module
======================

This module provides a function to recalculate albedo values for water and land based on
input temperature arrays and other parameters. The albedo is computed using a polynomial
correction factor applied to a base albedo for water and land. Additionally, for regions
where the temperature is less than or equal to -2, the albedo is set to a specified value.

Functions:
    albedo_seasonal(L, W, x, A_o, A_l, A_50)
        Computes the seasonal albedo for both water and land.
"""
from get_broadband_albedo import get_broadband_albedo
import numpy as np

def albedo_seasonal(L, W, x, A_o, A_l, A_50):
    """
    Recalculate albedo for water and land.

    The function calculates the albedo by applying a correction term based on a Legendre
    polynomial-like expression. For water and land regions, if the temperature is less than or 
    equal to -2, the albedo is adjusted to a specified threshold value (A_50).

    Parameters:
        L (array-like): Array of land temperatures (or other relevant variable).
        W (array-like): Array of water temperatures (or other relevant variable).
        x (array-like): Array related to spatial parameters (e.g., latitude) used in the albedo formula.
        A_o (float): Base water albedo parameter.
        A_l (float): Base land albedo parameter.
        A_50 (float): Albedo value to use when temperature is <= -2.

    Returns:
        tuple:
            alb_l (array-like): Recalculated albedo for land.
            alb_w (array-like): Recalculated albedo for water.
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