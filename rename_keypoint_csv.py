import os
import shutil

CSI_DIR = 'csi_data'
KEYPOINT_DIR = 'keypoint_csv'

# Map keypoint subfolder to csi_data subfolder
keypoint_subfolders = [d for d in os.listdir(KEYPOINT_DIR) if os.path.isdir(os.path.join(KEYPOINT_DIR, d))]
csi_subfolders = [d for d in os.listdir(CSI_DIR) if os.path.isdir(os.path.join(CSI_DIR, d))]

# สร้าง mapping จาก keypoint_subfolder -> csi_subfolder
mapping = {}
for k in keypoint_subfolders:
    for c in csi_subfolders:
        if k.replace('_key_point', '') == c:
            mapping[k] = c
            break

# เปลี่ยนชื่อโฟลเดอร์ย่อย
for old_name, new_name in mapping.items():
    old_path = os.path.join(KEYPOINT_DIR, old_name)
    new_path = os.path.join(KEYPOINT_DIR, new_name)
    if old_path != new_path:
        if os.path.exists(new_path):
            print(f"Warning: {new_path} already exists. Skipping.")
        else:
            os.rename(old_path, new_path)
            print(f"Renamed folder: {old_path} -> {new_path}")

# เปลี่ยนชื่อไฟล์ภายในแต่ละโฟลเดอร์ย่อย
for subfolder in csi_subfolders:
    csi_files = sorted(os.listdir(os.path.join(CSI_DIR, subfolder)))
    kp_path = os.path.join(KEYPOINT_DIR, subfolder)
    if not os.path.exists(kp_path):
        print(f"Warning: {kp_path} does not exist. Skipping.")
        continue
    kp_files = sorted(os.listdir(kp_path))
    if len(csi_files) != len(kp_files):
        print(f"Warning: file count mismatch in {subfolder}. Skipping.")
        continue
    for old_file, new_file in zip(kp_files, csi_files):
        old_file_path = os.path.join(kp_path, old_file)
        new_file_path = os.path.join(kp_path, new_file)
        if old_file_path != new_file_path:
            if os.path.exists(new_file_path):
                print(f"Warning: {new_file_path} already exists. Skipping.")
            else:
                os.rename(old_file_path, new_file_path)
                print(f"Renamed file: {old_file_path} -> {new_file_path}")

print("Done renaming folders and files in keypoint_csv.") 