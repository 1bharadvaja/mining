from data_preprocessing import load_and_merge_data
from interpolation_methods import Interpolator
from visualization import plot_slice, plot_3d_volume

# Configuration
COLLAR_PATH = 'FW_Dataset_20240227/MPA_Collar_20240227.csv'
SAMPLE_PATH = 'FW_Dataset_20240227/MPA_Samples_BD_20240227.csv'
TARGET = 'Ag_ppm'  # Change to Pb_pct or Zn_pct as needed

# Preprocess data
data = load_and_merge_data(COLLAR_PATH, SAMPLE_PATH)

# Initialize interpolator
interpolator = Interpolator(data)
grid = interpolator.create_grid()

# Run different interpolation methods
kriging_results = interpolator.kriging_interpolation(TARGET, grid)
idw_results = interpolator.idw_interpolation(TARGET, grid)
#nn_results = interpolator.neural_net_interpolation(TARGET, grid)

# Visualize results
plot_slice(kriging_results, grid, TARGET, elevation=1100, method='Kriging')
plot_3d_volume(kriging_results, grid, TARGET)
