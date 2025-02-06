import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go

class Visualizer:
    def __init__(self, df, grid, values, method_name):
        self.df = df
        self.grid = grid  # Tuple of (x_grid, y_grid, z_grid)
        self.values = values
        self.method_name = method_name
        self.cmap = 'viridis'

    def plot_2d_slices(self):
        """Generate orthogonal slice plots through the 3D volume"""
        x, y, z = self.grid
        mid_x = len(x) // 2
        mid_y = len(y) // 2
        mid_z = len(z) // 2

        fig, ax = plt.subplots(1, 3, figsize=(18, 6))
        
        # XY Slice
        im0 = ax[0].imshow(self.values[:, :, mid_z].T, 
                         extent=[x.min(), x.max(), y.min(), y.max()],
                         origin='lower', cmap=self.cmap)
        self._plot_original_points(ax[0], 'XY')
        fig.colorbar(im0, ax=ax[0], label='Concentration')
        ax[0].set_title(f'{self.method_name} - XY Plane (Z={z[mid_z]:.1f}m)')

        # XZ Slice
        im1 = ax[1].imshow(self.values[mid_y, :, :].T,
                         extent=[x.min(), x.max(), z.min(), z.max()],
                         origin='lower', cmap=self.cmap)
        self._plot_original_points(ax[1], 'XZ')
        fig.colorbar(im1, ax=ax[1], label='Concentration')
        ax[1].set_title(f'{self.method_name} - XZ Plane (Y={y[mid_y]:.1f}m)')

        # YZ Slice
        im2 = ax[2].imshow(self.values[:, mid_x, :].T,
                         extent=[y.min(), y.max(), z.min(), z.max()],
                         origin='lower', cmap=self.cmap)
        self._plot_original_points(ax[2], 'YZ')
        fig.colorbar(im2, ax=ax[2], label='Concentration')
        ax[2].set_title(f'{self.method_name} - YZ Plane (X={x[mid_x]:.1f}m)')

        plt.tight_layout()
        plt.show()

    def plot_3d_volume(self, threshold=0.1):
        """Interactive 3D volume visualization using Plotly"""
        x, y, z = self.grid
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        fig = go.Figure(data=go.Volume(
            x=X.flatten(),
            y=Y.flatten(),
            z=Z.flatten(),
            value=self.values.flatten(),
            isomin=np.nanmin(self.values) + threshold*(np.nanmax(self.values)-np.nanmin(self.values)),
            isomax=np.nanmax(self_values),
            opacity=0.1,
            surface_count=20,
            colorscale=self.cmap,
            colorbar=dict(title='Concentration')
        ))
        
        # Add original data points
        fig.add_trace(go.Scatter3d(
            x=self.df['X'],
            y=self.df['Y'],
            z=self.df['Z'],
            mode='markers',
            marker=dict(
                size=4,
                color=self.df['Ag_g_t'],
                colorscale=self.cmap,
                showscale=True
            ),
            name='Original Samples'
        ))

        fig.update_layout(
            title=f'{self.method_name} 3D Interpolation - Silver (Ag) Distribution',
            scene=dict(
                xaxis_title='X (m)',
                yaxis_title='Y (m)',
                zaxis_title='Z (m)',
                aspectratio=dict(x=1, y=1, z=0.5)
            ),
            width=1200,
            height=800
        )
        fig.show()

    def _plot_original_points(self, ax, plane):
        """Helper to plot original sample locations"""
        if plane == 'XY':
            ax.scatter(self.df['X'], self.df['Y'], c='red', 
                      s=20, edgecolor='black', label='Samples')
        elif plane == 'XZ':
            ax.scatter(self.df['X'], self.df['Z'], c='red',
                      s=20, edgecolor='black', label='Samples')
        elif plane == 'YZ':
            ax.scatter(self.df['Y'], self.df['Z'], c='red',
                      s=20, edgecolor='black', label='Samples')
        ax.legend()
