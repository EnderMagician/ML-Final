import numpy as np

# Extinction coefficients for different filters
# Used for calculating true Flux
EXTINCTION_COEFFS = {
    'u': 4.81,
    'g': 3.64,
    'r': 2.70,
    'i': 2.06,
    'z': 1.58,
    'y': 1.31
}

def get_total_extinction(ebv, band):
    """
    Calculate total extinction for a given band using E(B-V) value.

    Parameters:
    ebv (float or np.ndarray): E(B-V) value(s)
    band (str): Photometric band ('u', 'g', 'r', 'i', 'z', 'y')

    Returns:
    float or np.ndarray: Total extinction for the specified band
    """
    if band not in EXTINCTION_COEFFS:
        raise ValueError(f"Band '{band}' is not recognized. Valid bands are: {list(EXTINCTION_COEFFS.keys())}")
    
    return EXTINCTION_COEFFS[band] * ebv