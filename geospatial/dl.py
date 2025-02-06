import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.preprocessing import PowerTransformer
from torch.utils.data import DataLoader, TensorDataset
from .utils import calculate_3d_coordinates, create_interpolation_grid

class MineralNet(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.GELU(),
            nn.Linear(hidden_dim//2, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class DLInterpolator3D:
    def __init__(self, df: pd.DataFrame):
        self.df = calculate_3d_coordinates(df)
        self.scaler = PowerTransformer()
        self.model = None
        self.grids = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _prepare_data(self, property_name: str) -> tuple:
        valid_df = self.df.dropna(subset=[property_name])
        coords = valid_df[['X', 'Y', 'Z']].values
        target = valid_df[property_name].values.reshape(-1, 1)
        
        X = self.scaler.fit_transform(coords)
        y = self.scaler.fit_transform(target)
        
        return (torch.tensor(X, dtype=torch.float32).to(self.device),
                torch.tensor(y, dtype=torch.float32).to(self.device))

    def train(self, property_name: str, epochs: int = 1000):
        X, y = self._prepare_data(property_name)
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        self.model = MineralNet().to(self.device)
        optimizer = optim.AdamW(self.model.parameters(), lr=1e-3)
        criterion = nn.HuberLoss()
        
        for epoch in range(epochs):
            self.model.train()
            for inputs, targets in loader:
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
    def predict(self, grid_size: int = 50) -> np.ndarray:
        self.grids = create_interpolation_grid(self.df, grid_size)
        xx, yy, zz = np.meshgrid(*self.grids, indexing='ij')
        grid_points = np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T
        
        grid_scaled = self.scaler.transform(grid_points)
        tensor_grid = torch.tensor(grid_scaled, 
                                  dtype=torch.float32).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            preds = self.model(tensor_grid)
            
        return preds.cpu().numpy().reshape(len(self.grids[0]), 
                                         len(self.grids[1]),
                                         len(self.grids[2]))
