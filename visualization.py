import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pyvista as pv

def plot_slice(results, grid, target_name, elevation=1100, method='Kriging'):
    z_idx = np.abs(grid[2] - elevation).argmin()
    
    plt.figure(figsize=(10, 8))
    plt.contourf(grid[0][:,:,z_idx], grid[1][:,:,z_idx], results[:,:,z_idx])
    plt.colorbar(label=target_name)
    plt.title(f'{method} Interpolation at {elevation} m Elevation')
    plt.xlabel('Easting')
    plt.ylabel('Northing')
    plt.show()

def plot_3d_volume(results, grid, target_name):
    grid_pv = pv.StructuredGrid(grid[0], grid[1], grid[2])
    grid_pv[target_name] = results.ravel()
    
    plotter = pv.Plotter()
    plotter.add_volume(grid_pv, cmap='viridis', opacity='sigmoid')
    plotter.show()
