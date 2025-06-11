import pandas as pd
import numpy as np

def to_complex(val):
    # ถ้าเป็น string เช่น "1+2j" ให้แปลงเป็น complex
    if isinstance(val, str):
        return complex(val.replace(' ', ''))
    return val


def read_csi_csv(file_path):
    df = pd.read_csv(file_path)
    print(f"Loaded CSI data with shape: {df.shape}")
    return df

def split_csi_csv(df):
    subcarrier_cols = [col for col in df.columns if 'subcarrier' in col]
    # สมมติ df คือ DataFrame ที่มีคอลัมน์ subcarrier_1, subcarrier_2, ...
    for col in df.columns:
        if 'subcarrier' in col:
            # แปลงค่าเป็น complex
            df[col] = df[col].apply(to_complex)
            # สร้างคอลัมน์ใหม่สำหรับ Amp และ Phase
            df[f'Amp_{col.split("_")[-1]}'] = df[col].apply(np.abs)
            df[f'Phase_{col.split("_")[-1]}'] = df[col].apply(np.angle)

    df = df.drop(subcarrier_cols, axis=1)
    return df