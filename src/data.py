import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd

def list_of_df_to_array(X_list, Y_list, fill_value=0):
    # Padding ให้ sequence ยาวเท่ากัน
    def pad_df_list(df_list, fill_value=0):
        n_features = df_list[0].shape[1]
        max_len = max(df.shape[0] for df in df_list)
        arr = np.full((len(df_list), max_len, n_features), fill_value, dtype=np.float32)
        for i, df in enumerate(df_list):
            length = df.shape[0]
            arr[i, :length, :] = df.values
        return arr

    X_array = pad_df_list(X_list, fill_value=fill_value)
    Y_array = pad_df_list(Y_list, fill_value=fill_value)
    return X_array, Y_array

class CSIDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

def get_dataloaders(X_train, Y_train, X_val, Y_val, batch_size=32):
    train_dataset = CSIDataset(X_train, Y_train)
    val_dataset = CSIDataset(X_val, Y_val)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    return train_loader, val_loader