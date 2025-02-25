"""
    Performs ice balance for the non-RGH case.
    
    Parameters:
      jmx       : Number of grid cells (integer)
      ice       : 1D numpy array of grid cell indices (0-indexed) where ice exists
      notice    : 1D numpy array of grid cell indices (0-indexed) where no ice is present (ocean)
      conduct   : Conduction constant (scalar)
      h         : Sea ice thickness array (length jmx)
      Tfrz      : Freezing temperature (scalar, e.g., -2)
      rprimew   : Radiative forcing for water (array of length jmx)
      Cw_delt   : Ocean scaling factor (scalar; typically Cw/delt)
      M         : Matrix of size (2*jmx, 2*jmx)
      r         : Right-hand side vector of length (2*jmx,)
      Diff_Op   : Diffusion operator matrix of size (jmx, jmx)
      B         : Parameter (scalar)
      nu_fw     : Array of water nu values (length jmx)
      fw        : Ocean fraction array (length jmx)
      delx      : Spatial grid spacing (scalar)
      Cw        : Parameter for ocean (scalar)
      W         : Water temperature array (length jmx)
      
    Returns:
      T       : Full temperature vector (length 2*jmx; interleaved land and ocean)
      L       : Land temperature array (length jmx)
      W_new   : Updated water temperature array (length jmx)
      Fnet    : Net flux array (flattened to 1D, length jmx)
"""
import numpy as np

def icebalance(jmx, ice, notice, conduct, h, Tfrz, rprimew, Cw_delt, M, r, Diff_Op, B, nu_fw, fw, delx, Cw, W):

    # Initialize k and Fnet (both of length jmx)
    k = np.zeros(jmx)
    Fnet = np.zeros(jmx)
    
    # In MATLAB, r(2*ice) accesses the ocean temperatures in the interleaved vector.
    # In Python (0-indexed), if grid cell i corresponds to land temperature at index 2*i and
    # ocean temperature at index 2*i+1, then the ocean indices for the grid cells in 'ice' are:
    r_ice = 2 * ice + 1  # array of indices in r corresponding to ocean at grid cells with ice
    
    # Update k for grid cells with ice
    k[ice] = conduct / h[ice]
    
    # Update the forcing vector r for the ocean part at grid cells with ice
    r[r_ice] = k[ice] * Tfrz - rprimew[ice]
    
    # Set up a deviation vector "dev" of length 2*jmx
    dev = np.zeros(2 * jmx)
    dev[r_ice] = -Cw_delt + k[ice]
    
    # Modify the system matrix: Mt = M + diag(dev)
    Mt = M + np.diag(dev)
    
    # Solve for I: I = inv(Mt) * r  (using a linear solver)
    I = np.linalg.solve(Mt, r)
    
    # Let T be the temperature vector, then adjust the ocean part (for grid cells in ice)
    T = I.copy()
    T[r_ice] = np.minimum(Tfrz, I[r_ice])
    
    # Extract land and ocean temperatures from the interleaved vector T:
    L = T[0::2]   # Land temperatures: indices 0,2,4,...
    W_new = T[1::2]  # Ocean temperatures: indices 1,3,5,...
    # Also, let I_ocean be the ocean part of I:
    I_ocean = I[1::2]
    
    # Compute the net flux for grid cells with ice:
    # In MATLAB:
    #   Fnet(ice) = Diff_Op(ice,:)*I - rprimew(ice) - B*W(ice) - nu_fw(ice).*(I(ice)-L(ice));
    # In Python, since I_ocean, L, and W_new are arrays (length jmx), for each grid cell in "ice":
    Fnet[ice] = (Diff_Op[ice, :] @ I_ocean) - rprimew[ice] - B * W_new[ice] - nu_fw[ice] * (I_ocean[ice] - L[ice])
    Fnet = Fnet.flatten()
    
    # --- Compute ocean heat flux adjustments ---
    # Separate northern and southern hemisphere grid cells.
    # MATLAB uses: nhice = ice(find(ice > jmx/2+1)) and shice = ice(find(ice < jmx/2));
    # Here we set a threshold. (Note: Adjust according to your model’s conventions.)
    threshold = jmx // 2  # For example, if jmx = 60, threshold = 30
    nhice = ice[ice >= threshold]
    shice = ice[ice < threshold]
    nhocn = notice[notice >= threshold]
    shocn = notice[notice < threshold]
    
    # Calculate the ice area in northern and southern hemispheres:
    nhicearea = np.sum(fw[nhice])
    shicearea = np.sum(fw[shice])
    # Maximum possible ocean/ice area in the northern and southern hemispheres:
    nhmax = np.sum(fw[threshold:jmx])
    shmax = np.sum(fw[0:threshold])
    
    # Calculate ocean under-ice flux (nhfw and shfw)
    nhfw = 2 * min(2 - 2*(nhicearea - delx)/nhmax, 2)
    shfw = 2 * min(2 - 2*(shicearea - delx)/shmax, 2)
    
    # Adjust net flux Fnet for grid cells with ice
    Fnet[nhice] += nhfw
    Fnet[shice] += shfw
    
    # Calculate ice-free ocean areas and adjust water temperature accordingly
    nhocnarea = nhmax - nhicearea
    shocnarea = shmax - shicearea
    nhdW = nhfw * nhicearea / nhocnarea / Cw_delt
    shdW = shfw * shicearea / shocnarea / Cw_delt
    W_new[nhocn] = W_new[nhocn] - nhdW
    W_new[shocn] = W_new[shocn] - shdW
    
    return T, L, W_new, Fnet

