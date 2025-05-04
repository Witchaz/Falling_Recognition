from src.read_csi_csv import read_csi_csv
from src.preprocessing import smooth_csi, normalize_csi
import sys,os

sys.path.insert(0, '/path/to/Project') # location of src 

# Step 1: Load CSV
df = read_csi_csv('/csi_data/walk/walk_0.csv')

# Assume subcarriers are in columns: sc1, sc2, ..., sc30
subcarrier_columns = [col for col in df.columns if 'sc' in col.lower()]
csi_data = df[subcarrier_columns].values  # numpy array

# Step 2: Smoothing
smoothed_csi = smooth_csi(csi_data, window_size=5)

# Step 3: Normalization
normalized_csi = normalize_csi(smoothed_csi)

print("Preprocessing complete. Normalized CSI shape:", normalized_csi.shape)
