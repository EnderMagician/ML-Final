import numpy as np
import pandas as pd
import sncosmo
from astropy.cosmology import Planck18 as cosmo

from scipy.stats import linregress

def clean_data(log_df: pd.DataFrame, lightcurves_df: pd.DataFrame):
    """
    Recover Intrinsic Luminosity from Observed Flux.
    Corrections:
    1. Distance: Inverse Square Law using 'Z'.
    2. Extinction: Milky Way dust using 'EBV'.
    """
    # 1. Create Copies
    meta = log_df.copy()
    lc = lightcurves_df.copy()

    # 2. Merge Physics Variables (Z, EBV)
    required_cols = ['object_id', 'Z', 'EBV']
    
    # Validation
    for col in required_cols:
        if col not in meta.columns:
            raise ValueError(f"Metadata missing: '{col}'")

    lc = lc.merge(meta[required_cols], on='object_id', how='left')

    # 3. Distance Correction (Inverse Square Law)
    # Optimization: Calc distance for unique Zs only
    unique_z = meta['Z'].unique()
    dist_mpc = cosmo.luminosity_distance(unique_z).value
    
    # Map Z -> Distance -> Factor (Squared)
    z_map = dict(zip(unique_z, dist_mpc))
    lc['d_lum_mpc'] = lc['Z'].map(z_map)
    dist_factor = lc['d_lum_mpc'] ** 2

    # 4. Extinction Correction (F99 Dust Law)
    # LSST Filter Map
    band_map = {
        'u': 'lsstu', 'g': 'lsstg', 'r': 'lsstr', 
        'i': 'lssti', 'z': 'lsstz', 'y': 'lssty'
    }

    # Setup Dust Model
    dust_law = sncosmo.F99Dust(r_v=3.1)
    dust_law.set(ebv=1.0) # Set ref EBV to calc coefficients
    
    lc['dust_factor'] = 1.0
    
    for ds_band, sn_band in band_map.items():
        try:
            wave_eff = sncosmo.get_bandpass(sn_band).wave_eff
        except Exception:
            continue
            
        # Calc transmission for EBV=1.0
        # propagate([wavelength], [flux]) -> returns flux array
        transmission_ref = dust_law.propagate(np.array([wave_eff]), np.array([1.0]))[0]
        
        # Convert T to Mag Coefficient: A_lambda = -2.5 * log10(T)
        if transmission_ref <= 0:
            a_lambda_ref = 0
        else:
            a_lambda_ref = -2.5 * np.log10(transmission_ref)
        
        # Apply: 10^(0.4 * A_lambda * Actual_EBV)
        mask = lc['Filter'] == ds_band
        if mask.any():
            lc.loc[mask, 'dust_factor'] = 10 ** (0.4 * a_lambda_ref * lc.loc[mask, 'EBV'])

    # 5. Apply Corrections to Flux and Error
    total_correction = dist_factor * lc['dust_factor']
    
    lc['Flux'] = lc['Flux'] * total_correction
    lc['Flux_err'] = lc['Flux_err'] * total_correction
    
    # Cleanup
    lc = lc.drop(columns=['d_lum_mpc', 'dust_factor', 'Z', 'EBV'])
    
    return meta, lc

def get_slope(time, flux):
    """
    Calculates robust linear slope. 
    Returns 0.0 if not enough data OR if all data points are on the same night.
    """
    # Need at least 2 points
    if len(time) < 2:
        return 0.0
    
    # CRITICAL FIX: Check if all time points are identical
    # If max time == min time, variance is 0, and linregress crashes.
    if np.max(time) == np.min(time):
        return 0.0

    try:
        slope, _, _, _, _ = linregress(time, flux)
        # Handle edge case where slope itself is NaN
        if np.isnan(slope):
            return 0.0
        return slope
    except ValueError:
        return 0.0

def extract_features(clean_lc: pd.DataFrame):
    """
    Aggregates light curves into a single row of features per object.
    Input: clean_lc (contains 'log_flux', 'Time (MJD)', 'Filter', 'object_id')
    Output: DataFrame with 1 row per object_id
    """
    # 1. Pre-calculate Log Flux (if not already done)
    # We clip to 1.0 to avoid log(negative)
    if 'log_flux' not in clean_lc.columns:
        clean_lc['log_flux'] = np.log10(clean_lc['Flux'].clip(lower=1.0))

    # 2. Group by Object
    # We will process each object's group to extract stats
    features_list = []
    
    # Iterate over groups (This is the most CPU intensive part)
    # We use a simple loop here. For massive scale, we'd use .apply() with a custom function
    # but a loop is easier to debug and robust for this scale.
    for obj_id, group in clean_lc.groupby('object_id'):
        obj_stats = {'object_id': obj_id}
        
        # --- A. Peak Features ---
        max_idx = group['log_flux'].argmax()
        peak_mjd = group['Time (MJD)'].iloc[max_idx]
        obj_stats['peak_flux'] = group['log_flux'].iloc[max_idx]
        
        # --- B. Slope Features (Rise/Fall) ---
        rise = group[group['Time (MJD)'] < peak_mjd]
        fall = group[group['Time (MJD)'] > peak_mjd]
        
        obj_stats['rise_slope'] = get_slope(rise['Time (MJD)'], rise['log_flux'])
        obj_stats['fall_slope'] = get_slope(fall['Time (MJD)'], fall['log_flux'])
        
        # --- C. Color Features (Average Flux per Band) ---
        # We calculate the mean log-flux for every band present
        band_means = group.groupby('Filter')['log_flux'].mean()
        
        # Extract specific bands (default to 0 if missing)
        u_flux = band_means.get('u', 0)
        z_flux = band_means.get('z', 0)
        
        # Color Ratio (Blue - Red in magnitude/log space)
        if u_flux > 0 and z_flux > 0:
            obj_stats['color_u_z'] = u_flux - z_flux
        else:
            obj_stats['color_u_z'] = 0
            
        features_list.append(obj_stats)

    return pd.DataFrame(features_list)