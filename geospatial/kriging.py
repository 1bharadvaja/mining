import numpy as np
import pandas as pd
from pykrige.ok3d import OrdinaryKriging3D
from sklearn.preprocessing import QuantileTransformer
from .utils import calculate_3d_coordinates, create_interpolation_grid

class KrigingInterpolator3D:
    def __init__(self, df: pd.DataFrame):
        self.df = calculate_3d_coordinates(df)
        self.scaler = QuantileTransformer(output_distribution='normal')
        self.grids = None

    def _prepare_data(self, property_name: str) -> tuple:
        valid_df = self.df.dropna(subset=[property_name])
        coords = valid_df[['X', 'Y', 'Z']].values
    
    # Handle small sample sizes for QuantileTransformer
        n_samples = len(coords)
        self.scaler.n_quantiles = min(n_samples, 1000)
    
        scaled_coords = self.scaler.fit_transform(coords)
    
    # Return separated coordinates and values
        return (
            scaled_coords[:, 0],  # x coordinates
            scaled_coords[:, 1],  # y coordinates
            scaled_coords[:, 2],  # z coordinates
            valid_df[property_name].values
        )

    def interpolate(self, property_name: str, grid_size: int = 50) -> np.ndarray:
        x, y, z, values = self._prepare_data(property_name)
        self.grids = create_interpolation_grid(self.df, grid_size)
    
        ok3d = OrdinaryKriging3D(
            x, y, z, values,
            variogram_model='power',
            pseudo_inv=True,
            verbose=False
        )
    
        return ok3d.execute('grid', *self.grids)[0]
