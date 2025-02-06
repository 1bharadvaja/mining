# geospatial/utils.py
import numpy as np
import pandas as pd

def calculate_3d_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    """Convert drillhole data to 3D Cartesian coordinates"""
    df = df.copy()
    df['Depth'] = (df['From_m'] + df['To_m']) / 2
    
    azimuth = np.radians(df['Azi'])
    dip = np.radians(df['Dip'])
    
    df['X'] = np.sin(azimuth) * np.cos(dip) * df['Depth']
    df['Y'] = np.cos(azimuth) * np.cos(dip) * df['Depth']
    df['Z'] = np.sin(dip) * df['Depth']
    
    return df

def create_interpolation_grid(df: pd.DataFrame, grid_size: int = 50) -> tuple:
    """Generate 3D grid with 15% buffer around data points"""
    buff = 0.15
    coords = ['X', 'Y', 'Z']
    
    ranges = {c: (df[c].min(), df[c].max()) for c in coords}
    grids = [
        np.linspace(r[0] - (r[1]-r[0])*buff, 
                   r[1] + (r[1]-r[0])*buff, 
                   grid_size)
        for r in ranges.values()
    ]
    
    return tuple(grids)
