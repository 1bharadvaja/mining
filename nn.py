import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import Rbf
from skimage import measure
import plotly.graph_objects as go
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


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

class MineralDataset(Dataset):
    """
    PyTorch Dataset for the mineral data.
    Loads the CSV files, computes 3D coordinates, and prepares input/target pairs.
    """
    def __init__(self, collar_csv, sample_csv):
        self.df = load_and_merge(collar_csv, sample_csv)
        # Use all available data (do not subsample)
        self.x, self.y, self.z, self.targets = compute_sample_coordinates(self.df)
        # Stack coordinates into a (N, 3) array.
        self.data = np.column_stack((self.x, self.y, self.z))

        nan_mask = ~np.isnan(self.data).any(axis=1) & ~np.isnan(self.targets)
        if np.sum(~nan_mask) > 0:
            print(f"Removing {np.sum(~nan_mask)} datapoints containing NaN values.")
        self.data = self.data[nan_mask]
        self.targets = self.targets[nan_mask]

          

    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]  # This is an array: [x, y, z]
        target = self.targets[idx]
        # Convert to torch tensors.
        sample = torch.tensor(sample, dtype=torch.float32)
        target = torch.tensor(target, dtype=torch.float32)
        return sample, target

class MLP(nn.Module):
    """
    A simple multilayer perceptron (MLP) to predict Zn_pct from 3D coordinates.
    """
    def __init__(self, input_dim=3, hidden_dim=64, output_dim=1):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x):
        return self.net(x)
def train(model, dataloader, criterion, optimizer, device, num_epochs=100):
    model.train()
    train_losses = []
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            # Reshape targets to have shape (batch_size, 1)
            targets = targets.to(device).view(-1, 1)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * inputs.size(0)
        
        epoch_loss /= len(dataloader.dataset)
        train_losses.append(epoch_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")
    
    return train_losses

def validate(model, dataloader, criterion, device):
    """
    Validate the model on a given DataLoader and return the average loss
    along with predictions and true targets.

    Parameters:
        model (torch.nn.Module): The trained PyTorch model.
        dataloader (torch.utils.data.DataLoader): DataLoader for the validation set.
        criterion (torch.nn.Module): The loss function (e.g., nn.MSELoss()).
        device (torch.device): The device on which to run inference (CPU or GPU).

    Returns:
        avg_loss (float): Average loss over the validation dataset.
        predictions (np.ndarray): The model's predictions as a NumPy array.
        true_targets (np.ndarray): The ground truth targets as a NumPy array.
    """
    model.eval()  # Set the model to evaluation mode
    total_loss = 0.0
    predictions = []
    true_targets = []
    
    # Disable gradient computations for validation.
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            # Ensure targets have the right shape: (batch_size, 1)
            targets = targets.to(device).view(-1, 1)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item() * inputs.size(0)
            
            # Collect predictions and true targets, converting to CPU numpy arrays.
            predictions.append(outputs.cpu().numpy())
            true_targets.append(targets.cpu().numpy())
    
    # Calculate average loss over the entire validation dataset.
    avg_loss = total_loss / len(dataloader.dataset)
    
    # Concatenate all batches into single arrays.
    predictions = np.concatenate(predictions, axis=0)
    true_targets = np.concatenate(true_targets, axis=0)
    
    return avg_loss, predictions, true_targets

def infer(threshold = -15.0):
    # ---------------------------------------------------------
    # 1. Define the 3D Inference Volume
    # ---------------------------------------------------------
    # Replace these boundaries with those relevant to your study area.
    x_range = (950, 1050)    # Easting bounds
    y_range = (1950, 2050)   # Northing bounds
    z_range = (40, 60)       # Elevation bounds (vertical)
    
    # Set the resolution: number of points along each axis.
    resolution = 20  # You can increase this for a finer grid.
    
    # Create grid points along each axis.
    x_vals = np.linspace(x_range[0], x_range[1], resolution)
    y_vals = np.linspace(y_range[0], y_range[1], resolution)
    z_vals = np.linspace(z_range[0], z_range[1], resolution)
    
    # Create a 3D grid of coordinates.
    X, Y, Z = np.meshgrid(x_vals, y_vals, z_vals, indexing='ij')
    
    # Flatten the grid into a list of coordinates of shape (N, 3)
    coords = np.column_stack((X.flatten(), Y.flatten(), Z.flatten()))
    
    # ---------------------------------------------------------
    # 2. Load the Trained Model
    # ---------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP(input_dim=3, hidden_dim=64, output_dim=1).to(device)
    model.load_state_dict(torch.load("mineral_model.pth", map_location=device))
    model.eval()
    
    # ---------------------------------------------------------
    # 3. Perform Inference
    # ---------------------------------------------------------
    # Convert the grid coordinates to a torch tensor.
    inputs = torch.tensor(coords, dtype=torch.float32)
    
    with torch.no_grad():
        # Model outputs will have shape (N, 1); flatten to (N,)
        predictions = model(inputs.to(device)).cpu().numpy().flatten()
    
    # ---------------------------------------------------------
    # 4. Plot the Inferred Data in 3D
    # ---------------------------------------------------------
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create a scatter plot. The color of each point corresponds to the predicted Zn_pct.
    sc = ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
                    c=predictions, cmap='viridis', marker='o')
    pred_3d = predictions.reshape(X.shape)

    
    mask = pred_3d >= threshold
    colors = np.empty(mask.shape, dtype=object)
    colors[mask] = 'red'
    ax = plt.figure().add_subplot(projection='3d')
    ax.voxels(mask, facecolors=colors, edgecolor='k')
    
    ax.set_xlabel("Easting")
    ax.set_ylabel("Northing")
    ax.set_zlabel("Elevation")
    fig.colorbar(sc, ax=ax, label="Predicted Zn_pct")
    plt.title("3D Inference of Mineral Deposit (Predicted Zn_pct)")
    plt.show()



def main():
    # File paths (update these paths as needed)
    collar_csv = "MPA_Collar_20240227.csv"
    sample_csv = "MPA_Samples_BD_20240227.csv"
    
    # Create the dataset and dataloader.
    dataset = MineralDataset(collar_csv, sample_csv)
    print("Total samples:", len(dataset))
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Device configuration.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Training on device:", device)
    
    # Initialize the model, loss function, and optimizer.
    model = MLP(input_dim=3, hidden_dim=64, output_dim=1).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Train the network.
    num_epochs = 200
    losses = train(model, dataloader, criterion, optimizer, device, num_epochs=num_epochs)
    
    # Plot training loss.
    plt.figure(figsize=(8, 6))
    plt.plot(losses, label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training Loss Curve")
    plt.legend()
    plt.show()
    
    # Save the trained model.
    torch.save(model.state_dict(), "mineral_model.pth")
    print("Model saved to mineral_model.pth")
