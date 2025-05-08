import pandas as pd

df = pd.read_csv("./csi_data/walk/walk_10.csv")

# ตรวจว่ามีกี่คอลัมน์
print(df.columns)

# ตรวจว่ามีค่า missing ไหม
print(df.isnull().sum())

# ตรวจว่าแถวใดมีคอลัมน์ไม่ครบ
expected_cols = 65  # timestamp + 30 subcarriers
with open("./csi_data/walk/walk_10.csv") as f:
    for i, line in enumerate(f):
        data_amount = len(line.strip().split(","))
        if data_amount != expected_cols:
            print(f"Line {i+1} has wrong number of columns. Amount : {data_amount}")