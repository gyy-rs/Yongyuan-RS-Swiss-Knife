import sys
import cupy as cp

# ==========================================
# Author Information
# ==========================================
__author__ = "Yongyuan Gao"
__date__ = "2023-10-27"
__version__ = "2.1.8"

# Print system and library information
print(f"Python Executable: {sys.executable}")
print(f"Python Version: {sys.version}")
print(f"CuPy Version: {cp.__version__}")


def Ross_thick(sZenith, vZenith, rAzimuth):
    """
    Calculates the RossThick kernel using CuPy for GPU acceleration.
    """
    cosxi = cp.cos(sZenith) * cp.cos(vZenith) + cp.sin(sZenith) * cp.sin(vZenith) * cp.cos(rAzimuth)
    
    # Use clip to prevent values from exceeding the valid domain of arccos [-1, 1]
    xi = cp.arccos(cp.clip(cosxi, -1, 1))
    
    k1 = (cp.pi / 2 - xi) * cosxi + cp.sin(xi)
    k = k1 / (cp.cos(sZenith) + cp.cos(vZenith)) - cp.pi / 4
    return k


def Li_Transit(sZenith, vZenith, rAzimuth):
    """
    Calculates the LiTransit kernel using CuPy for GPU acceleration.
    """
    # Create a copy (abs returns a new array) to avoid modifying the original input
    rAzimuth = cp.abs(rAzimuth)
    
    # Adjust azimuth range where it exceeds PI
    # Using boolean indexing which works natively in CuPy
    mask = rAzimuth >= cp.pi
    rAzimuth[mask] = 2 * cp.pi - rAzimuth[mask]

    brratio = 1
    hbratio = 2
    t1 = brratio * cp.tan(sZenith)
    theta_ip = cp.arctan(t1)
    t2 = brratio * cp.tan(vZenith)
    theta_vp = cp.arctan(t2)
    
    temp1 = cp.cos(theta_ip)
    temp2 = cp.cos(theta_vp)
    
    cosxip = temp1 * temp2 + cp.sin(theta_ip) * cp.sin(theta_vp) * cp.cos(rAzimuth)
    
    D1 = cp.tan(theta_ip) ** 2 + cp.tan(theta_vp) ** 2 - 2 * cp.tan(theta_ip) * cp.tan(theta_vp) * cp.cos(rAzimuth)
    D = cp.sqrt(D1)
    
    cost1 = cp.tan(theta_ip) * cp.tan(theta_vp) * cp.sin(rAzimuth)
    cost2 = D1 + cost1 ** 2
    temp3 = 1 / temp1 + 1 / temp2
    
    cost = hbratio * cp.sqrt(cost2) / temp3
    
    # Clip values to stay within the valid domain for arccos
    cost = cp.clip(cost, -1, 1)
    t = cp.arccos(cost)

    O = (t - cp.sin(t) * cost) * temp3 / cp.pi
    B = temp3 - O
    
    # Use cp.where for element-wise conditional logic (equivalent to np.where)
    k = cp.where(B > 2, (1 + cosxip) / (temp2 * temp1 * B) - 2, -B + (1 + cosxip) / (2 * temp2 * temp1))

    return k


def BRDF_degree_vectorized(i, v, r, iso, vol, geo):
    """
    Main driver function to calculate BRDF using vectorized CuPy operations.
    Inputs are expected to be pandas Series or similar array-likes.
    """
    
    try:
        # Convert pandas.Series (CPU memory) to CuPy arrays (GPU memory)
        i_array = cp.array(i.values)
        v_array = cp.array(v.values)
        r_array = cp.array(r.values)
        iso_array = cp.array(iso.values)
        vol_array = cp.array(vol.values)
        geo_array = cp.array(geo.values)

        # Convert degrees to radians on the GPU
        i_rad = cp.radians(i_array)
        v_rad = cp.radians(v_array)
        r_rad = cp.radians(r_array)
        
        # Perform kernel calculations on the GPU
        R = iso_array + vol_array * Ross_thick(i_rad, v_rad, r_rad) + geo_array * Li_Transit(i_rad, v_rad, r_rad)

        # Transfer the result back to CPU memory (NumPy array)
        return cp.asnumpy(R)

    except Exception as e:
        print(f"Error during CuPy calculation: {e}")
        # Return None or handle the fallback appropriately
        return None