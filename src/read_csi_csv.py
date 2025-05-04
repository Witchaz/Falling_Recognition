import pandas as pd

def read_csi_csv(file_path):
    df = pd.read_csv(file_path)
    print(f"Loaded CSI data with shape: {df.shape}")
    return df
