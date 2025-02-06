#!/usr/bin/env python3
"""
Simple 3D interpolation of drillhole assay data.

This script:
  1. Reads the collar and sample CSV files.
  2. Merges them on "HoleID".
  3. Computes a 3D coordinate for each sample.
  4. Interpolates a chosen metric (e.g., Zn_pct) using SciPy’s RBF in 3D.
  5. Plots a horizontal slice at a representative depth.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import Rbf

# For 3D plotting of data points (if desired)
from mpl_toolkits.mplot3d import Axes3D

import math

# --- Step 1. Load and Merge Data ---

def load_and_merge(collar_csv, sample_csv):
    """
    Load the collar and sample CSV files and merge them on "HoleID".

    Parameters:
        collar_csv (str): Path to the collar CSV.
        sample_csv (str): Path to the sample CSV.

    Returns:
        pd.DataFrame: Merged DataFrame.
    """
    # Assume the CSV files are comma-separated
    collar_df = pd.read_csv(collar_csv, encoding='utf-8-sig')
    sample_df = pd.read_csv(sample_csv, encoding='utf-8-sig')

    # Strip whitespace from column names
    collar_df.columns = [col.strip() for col in collar_df.columns]
    sample_df.columns = [col.strip() for col in sample_df.columns]

    # Merge on "HoleID". Use inner join so only holes with both collar and sample data are kept.
    merged = pd.merge(sample_df, collar_df, on="HoleID", how="inner")
    return merged

# --- Step 2. Compute 3D Sample Coordinates ---

def compute_sample_coordinates(df):
    """
    Compute 3D coordinates for each assay sample.

    The collar provides the surface (or entry) coordinates (Easting, Northing, Elevation).
    Each sample interval has a 'From_m' and 'To_m' along the drill hole.
    We compute the mid-depth along the drill hole, then use the hole's dip and azimuth
    (from the collar data) to compute horizontal and vertical offsets.

    Parameters:
        df (pd.DataFrame): Merged DataFrame containing collar and sample data.

    Returns:
        np.ndarray: x, y, z coordinates and the target metric (e.g., Zn_pct) as 1D arrays.
    """
    # Compute mid-depth for each sample
    df['MidDepth_m'] = (df['From_m'] + df['To_m']) / 2.0

    # Convert angles from degrees to radians.
    # Assume "Dip" is negative (downward). Use the absolute value for computing distances.
    dip_rad = np.deg2rad(np.abs(df['Dip'].astype(float)))
    az_rad  = np.deg2rad(df['Azimuth'].astype(float))
    
    # Compute horizontal distance along the drill hole (projected onto a horizontal plane)
    horizontal_distance = df['MidDepth_m'] * np.cos(dip_rad)
    # Compute vertical distance (how far down the sample is from the collar elevation)
    vertical_distance   = df['MidDepth_m'] * np.sin(dip_rad)
    
    # Offsets in the horizontal plane.
    # Assume azimuth is measured in degrees clockwise from north.
    east_offset  = horizontal_distance * np.sin(az_rad)
    north_offset = horizontal_distance * np.cos(az_rad)
    
    # Compute final 3D coordinates:
    # x: collar Easting + east_offset
    # y: collar Northing + north_offset
    # z: collar Elevation minus vertical_distance (since vertical_distance is downward)
    x = df['Easting'].astype(float) + east_offset
    y = df['Northing'].astype(float) + north_offset
    z = df['Elevation'].astype(float) - vertical_distance

    # For this example we use "Zn_pct" as the target metric. Adjust as needed.
    target = df['Zn_pct'].astype(float).values

    return x.values, y.values, z.values, target

# --- Step 3. 3D Interpolation using RBF ---

def interpolate_3d(x, y, z, values, function='multiquadric'):
    """
    Interpolate values in 3D using SciPy's RBF interpolation.

    Parameters:
        x, y, z (np.ndarray): 1D arrays of coordinates.
        values (np.ndarray): 1D array of target metric values.
        function (str): Type of RBF function (default 'multiquadric').

    Returns:
        rbf_func: A function that can be called with (xi, yi, zi) coordinates.
    """
    # Create the RBF interpolator in 3D
    rbf_func = Rbf(x, y, z, values, function=function)
    return rbf_func

# --- Step 4. Plot a Horizontal Slice of the Interpolated Data ---

def plot_horizontal_slice(rbf_func, x_range, y_range, z_level, resolution=100):
    """
    Plot a horizontal slice (at a constant z-level) of the 3D interpolation.

    Parameters:
        rbf_func: The RBF interpolator function.
        x_range (tuple): (xmin, xmax)
        y_range (tuple): (ymin, ymax)
        z_level (float): The constant z level at which to evaluate the interpolation.
        resolution (int): Number of grid points in each horizontal direction.
    """
    xi = np.linspace(x_range[0], x_range[1], resolution)
    yi = np.linspace(y_range[0], y_range[1], resolution)
    xi, yi = np.meshgrid(xi, yi)
    
    # Evaluate the interpolator at each (x, y) on the grid at the given z_level
    zi = rbf_func(xi, yi, z_level)
    
    plt.figure(figsize=(8, 6))
    contour = plt.contourf(xi, yi, zi, cmap="viridis", levels=20)
    plt.colorbar(contour, label='Interpolated Zn_pct')
    plt.xlabel("Easting")
    plt.ylabel("Northing")
    plt.title(f"Horizontal Slice at Z = {z_level:.1f}")
    plt.tight_layout()
    plt.show()

# --- Optional: 3D Scatter Plot of Data Points ---

def plot_3d_scatter(x, y, z, values):
    """
    Plot the original 3D data points colored by the target value.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(x, y, z, c=values, cmap='viridis')
    ax.set_xlabel("Easting")
    ax.set_ylabel("Northing")
    ax.set_zlabel("Elevation")
    plt.title("3D Scatter of Sample Data")
    plt.colorbar(sc, label='Zn_pct')
    plt.show()

# --- Main Routine ---

def main():
    # File paths (update these to your actual CSV file locations)
    collar_csv = "FW_Dataset_20240227/MPA_Collar_20240227.csv"
    sample_csv = "FW_Dataset_20240227/MPA_Samples_BD_20240227.csv"
    
    # Load and merge data
    merged_df = load_and_merge(collar_csv, sample_csv)
    print(f"Merged data has {len(merged_df)} rows.")
    
    # Compute 3D coordinates and target metric
    x, y, z, target = compute_sample_coordinates(merged_df)
    print("Computed 3D coordinates for all samples.")
    
    # Create an RBF interpolator in 3D
    rbf_func = interpolate_3d(x, y, z, target, function='multiquadric')
    print("Created 3D RBF interpolator.")
    
    # Plot a horizontal slice of the interpolated data.
    # Determine x, y bounds from the data.
    x_min, x_max = np.min(x), np.max(x)
    y_min, y_max = np.min(y), np.max(y)
    
    # Choose a z level for the horizontal slice.
    # For example, take the median of the sample z coordinates.
    z_level = np.median(z)
    print(f"Plotting horizontal slice at z = {z_level:.1f}")
    
    plot_horizontal_slice(rbf_func, (x_min, x_max), (y_min, y_max), , resolution=150)
    
    # Optionally, plot the original data in 3D.
    plot_3d_scatter(x, y, z, target)

if __name__ == "__main__":
    main()
