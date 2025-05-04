
import numpy as np
import matplotlib.pyplot as plt

# สมมุติว่า csi_data คือ ndarray ขนาด [time_steps, 64]
# เช่น: csi_data.shape = (1000, 64)

def plot(csi_data):

    # สร้างแกนเวลา (เช่น row index)
    time = np.arange(csi_data.shape[0])

    # Plot
    plt.figure(figsize=(15, 8))
    for i in range(csi_data.shape[1]):
        plt.plot(time, csi_data[:, i], label=f'Subcarrier {i}')

    plt.xlabel("Time (index)")
    plt.ylabel("Amplitude")
    plt.title("CSI Amplitude over Time (64 Subcarriers)")
    plt.legend(loc='upper right', ncol=4, fontsize=7)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
