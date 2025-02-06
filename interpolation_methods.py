import numpy as np
from pykrige.ok3d import OrdinaryKriging3D
from scipy.interpolate import griddata
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

class Interpolator:
    def __init__(self, data):
        self.data = data.dropna()
        self.X = data[['Easting_3D', 'Northing_3D', 'Elevation_3D']].values
        self.scaler = StandardScaler().fit(self.X)
        
    def create_grid(self, resolution=50):
        x_min, x_max = self.X[:,0].min(), self.X[:,0].max()
        y_min, y_max = self.X[:,1].min(), self.X[:,1].max()
        z_min, z_max = self.X[:,2].min(), self.X[:,2].max()
        
        return np.mgrid[x_min:x_max:resolution*1j,
                        y_min:y_max:resolution*1j,
                        z_min:z_max:resolution*1j]

    def kriging_interpolation(self, target='Ag_ppm', grid=None):
        ok = OrdinaryKriging3D(
            self.X[:,0], self.X[:,1], self.X[:,2],
            self.data[target].values,
            variogram_model='spherical',
            nlags=20,
            enable_plotting=False
        )
        
        if grid is None:
            grid = self.create_grid()
            
        z, ss = ok.execute('grid', grid[0], grid[1], grid[2])
        return z.data
        
    def idw_interpolation(self, target='Ag_ppm', grid=None):
        if grid is None:
            grid = self.create_grid()
            
        return griddata(self.X, self.data[target].values,
                       (grid[0], grid[1], grid[2]), method='linear')
        
    def neural_net_interpolation(self, target='Ag_ppm', grid=None):
        X_scaled = self.scaler.transform(self.X)
        y = self.data[target].values
        
        nn = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000)
        nn.fit(X_scaled, y)
        
        if grid is None:
            grid = self.create_grid()
            
        grid_flat = np.vstack([grid[0].ravel(), 
                             grid[1].ravel(), 
                             grid[2].ravel()]).T
        grid_scaled = self.scaler.transform(grid_flat)
        
        return nn.predict(grid_scaled).reshape(grid[0].shape)
