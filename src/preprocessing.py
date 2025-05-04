import numpy as np
from scipy.ndimage import uniform_filter1d
from sklearn.preprocessing import MinMaxScaler

def smooth_csi(data, window_size=5):
    """Apply moving average smoothing to each column."""
    return uniform_filter1d(data, size=window_size, axis=0)

def normalize_csi(data):
    """Apply min-max normalization column-wise."""
    scaler = MinMaxScaler()
    normalized = scaler.fit_transform(data)
    return normalized
