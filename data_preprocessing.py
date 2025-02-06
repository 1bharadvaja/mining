import pandas as pd
import numpy as np
from math import sin, cos, radians

def load_and_merge_data(collar_path, sample_path):
    # Load data
    collar_df = pd.read_csv(collar_path)
    sample_df = pd.read_csv(sample_path)
    
    # Merge datasets
    merged = pd.merge(sample_df, collar_df, on='HoleID', how='left')
    merged = merged.dropna(subset=['Easting', 'Northing', 'Elevation'])
    
    # Calculate sample midpoint depth
    merged['Midpoint'] = (merged['From_m'] + merged['To_m']) / 2
    
    # Convert dip to radians
    merged['Dip_rad'] = np.radians(abs(merged['Dip']))
    
    # Calculate 3D coordinates
    merged['Horizontal_Displacement'] = merged['Midpoint'] * np.cos(merged['Dip_rad'])
    merged['Vertical_Displacement'] = merged['Midpoint'] * np.sin(merged['Dip_rad'])
    
    merged['Easting_3D'] = merged['Easting'] + \
        merged['Horizontal_Displacement'] * np.sin(np.radians(merged['Azimuth']))
    
    merged['Northing_3D'] = merged['Northing'] + \
        merged['Horizontal_Displacement'] * np.cos(np.radians(merged['Azimuth']))
    
    merged['Elevation_3D'] = merged['Elevation'] - merged['Vertical_Displacement']
    
    return merged[['Easting_3D', 'Northing_3D', 'Elevation_3D', 
                  'Ag_ppm', 'Pb_pct', 'Zn_pct', 'BD_tonnes_m3']]
