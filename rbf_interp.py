import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import Rbf

# For 3D plotting of data points (if desired)
from mpl_toolkits.mplot3d import Axes3D

import math

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


def plot_interpolated_3d(rbf_func, x_range, y_range, z_range, resolution=10):
    """
    Create a 3D grid over the specified ranges, evaluate the RBF interpolator
    on that grid, and plot the result in a 3D scatter plot.

    Parameters:
        rbf_func: The RBF interpolator function.
        x_range (tuple): (xmin, xmax)
        y_range (tuple): (ymin, ymax)
        z_range (tuple): (zmin, zmax)
        resolution (int): Number of points along each axis (default 10, total points = resolution^3)
    """
    # Create a 3D grid
    xi = np.linspace(x_range[0], x_range[1], resolution)
    yi = np.linspace(y_range[0], y_range[1], resolution)
    zi = np.linspace(z_range[0], z_range[1], resolution)
    
    # Create the 3D meshgrid
    XI, YI, ZI = np.meshgrid(xi, yi, zi)
    
    # Evaluate the interpolator on the grid.
    VI = rbf_func(XI, YI, ZI)
    
    # Flatten the arrays for plotting
    XI_flat = XI.flatten()
    YI_flat = YI.flatten()
    ZI_flat = ZI.flatten()
    VI_flat = VI.flatten()
    
    # Create a 3D scatter plot with color representing the interpolated value.
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(XI_flat, YI_flat, ZI_flat, c=VI_flat, cmap='viridis', s=40)
    ax.set_xlabel("Easting")
    ax.set_ylabel("Northing")
    ax.set_zlabel("Elevation")
    plt.title("3D Interpolated Zn_pct")
    plt.colorbar(scatter, label="Interpolated Zn_pct")
    plt.show()

#this function plots the mineral deposit depending on whether the data
# shows a certain threshold of concentration has been passed at the coordinate


def plot_voxel_threshold(rbf_func, x_range, y_range, z_range, threshold, resolution=20):
    """
    Plot voxels representing areas where the interpolated value exceeds a threshold.
    
    A 3D grid is created over the specified ranges. The RBF interpolator is then
    evaluated at the centers of each voxel. Voxels are drawn only if the interpolated
    value exceeds the given threshold. The voxel color is set using the "viridis" colormap,
    so that variations above the threshold are visually apparent.

    Parameters:
        rbf_func: The RBF interpolator function.
        x_range, y_range, z_range (tuple): The (min, max) boundaries of each axis.
        threshold (float): The threshold value for Zn_pct.
        resolution (int): Number of grid points along each axis.
    """
    # Create grid centers for evaluation.
    x_centers = np.linspace(x_range[0], x_range[1], resolution)
    y_centers = np.linspace(y_range[0], y_range[1], resolution)
    z_centers = np.linspace(z_range[0], z_range[1], resolution)
    Xc, Yc, Zc = np.meshgrid(x_centers, y_centers, z_centers, indexing='ij')
    VI = rbf_func(Xc, Yc, Zc)
    print(VI)

    # Create a boolean mask: True where the interpolated value exceeds the threshold.
    mask = VI >= threshold
    print(mask)
    colors = np.empty(mask.shape, dtype=object)
    colors[mask] = 'red'

    ax = plt.figure().add_subplot(projection='3d')
    ax.voxels(mask, facecolors=colors, edgecolor='k')




    # Prepare a facecolors array using a colormap.
    """cmap = plt.get_cmap('viridis')
    norm = plt.Normalize(np.min(VI), np.max(VI))
    # Allocate facecolors with an extra dimension for RGBA (i.e., shape = mask.shape + (4,))
    facecolors = np.empty(mask.shape + (4,), dtype=float)
    
    # For voxels where mask is True, assign color from colormap.
    facecolors[mask] = cmap(norm(VI[mask]))
    # For voxels where mask is False, assign a transparent color.
    facecolors[~mask] = (0, 0, 0, 0)

    # Create grid boundaries for the voxels.
    x_bound = np.linspace(x_range[0], x_range[1], resolution + 1)
    y_bound = np.linspace(y_range[0], y_range[1], resolution + 1)
    z_bound = np.linspace(z_range[0], z_range[1], resolution + 1)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.voxels(x_bound, y_bound, z_bound, mask, facecolors=facecolors, edgecolor='k', alpha=0.5)
    ax.set_xlabel("Easting")
    ax.set_ylabel("Northing")
    ax.set_zlabel("Elevation")
    plt.title(f"Voxel Plot of Mineral Deposit (Threshold = {threshold})")
    plt.show()"""


def main():
    # File paths (update these to your actual CSV file locations)
    collar_csv = "MPA_Collar_20240227.csv"
    sample_csv = "MPA_Samples_BD_20240227.csv"
    
    # Load and merge data
    merged_df = load_and_merge(collar_csv, sample_csv)
    print(f"Merged data has {len(merged_df)} rows.")

    # Subsample the data for testing if necessary.
    if len(merged_df) > 100:
        merged_df = merged_df.sample(n=100, random_state=42)
        print(f"Subsampled data to {len(merged_df)} rows for testing.")

    # Compute 3D coordinates and target metric
    x, y, z, target = compute_sample_coordinates(merged_df)
    print("Computed 3D coordinates for all samples.")
    
    # Debug: print shapes of training arrays
    print("Training data shapes:")
    print("x:", x.shape)
    print("y:", y.shape)
    print("z:", z.shape)
    print("target:", target.shape)
    
    # Create an RBF interpolator in 3D
    rbf_func = interpolate_3d(x, y, z, target, function='multiquadric')
    print("Created 3D RBF interpolator.")
    
    # Determine bounds for the 3D grid from the training data.
    x_range = (np.min(x), np.max(x))
    y_range = (np.min(y), np.max(y))
    z_range = (np.min(z), np.max(z))
    print(f"3D grid bounds:\n x: {x_range}\n y: {y_range}\n z: {z_range}")
    
    # Plot the original data for reference.
    plot_3d_scatter(x, y, z, target)
    
    # Plot the interpolated 3D data on a relatively low resolution grid.
    plot_interpolated_3d(rbf_func, x_range, y_range, z_range, resolution=10)
