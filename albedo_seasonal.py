from get_broadband_albedo import get_broadband_albedo
import numpy as np

def albedo_seasonal(L, W, x, A_o, A_l, A_50):
    """
    Recalculate albedo for water and land.
    
    Parameters:
        L   : Array of land temperature (or other relevant variable)
        W   : Array of water temperature (or other relevant variable)
        x   : Array (e.g., related to latitude; used in the albedo formula)
        A_o : Base water albedo parameter
        A_l : Base land albedo parameter
        A_50: Albedo value to use when temperature <= -2
        
    Returns:
        alb_l : Recalculated albedo for land
        alb_w : Recalculated albedo for water
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