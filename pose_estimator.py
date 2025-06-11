import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# 1. เตรียม path
csi_data_dir = 'csi_data'
keypoint_csv_dir = 'keypoint_csv'

# 2. อ่านไฟล์ feature และ label
feature_files = sorted(os.listdir(csi_data_dir))
label_files = sorted(os.listdir(keypoint_csv_dir))

X = []
Y = []

for feat_file, label_file in zip(feature_files, label_files):
    # อ่าน feature (สมมติเป็น .npy หรือ .csv)
    feat_path = os.path.join(csi_data_dir, feat_file)
    if feat_file.endswith('.npy'):
        feat = np.load(feat_path)
    elif feat_file.endswith('.csv'):
        feat = pd.read_csv(feat_path).values
    else:
        continue  # ข้ามไฟล์ที่ไม่รู้จัก
    
    # อ่าน label (สมมติเป็น .csv)
    label_path = os.path.join(keypoint_csv_dir, label_file)
    label = pd.read_csv(label_path).values.flatten()  # สมมติ label เป็น 1 แถว
    
    X.append(feat.flatten())  # flatten ถ้า feature เป็น matrix
    Y.append(label)

X = np.array(X)
Y = np.array(Y)

# 3. แบ่ง train/test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# 4. สร้างและ train โมเดล
model = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42))
model.fit(X_train, Y_train)

# 5. ทำนายและประเมินผล
Y_pred = model.predict(X_test)
mse = mean_squared_error(Y_test, Y_pred, multioutput='raw_values')
print("MSE for each label:", mse)
print("Average MSE:", np.mean(mse))