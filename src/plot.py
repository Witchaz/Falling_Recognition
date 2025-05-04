
import numpy as np
import matplotlib.pyplot as plt

# สมมุติว่า csi_data คือ ndarray ขนาด [time_steps, 64]
# เช่น: csi_data.shape = (1000, 64)

def plot(csi_data, title="CSI Amplitude over Time (64 Subcarriers)"):
    """Plot CSI to graph
    csi_data => เอากราฟที่แกน x เป็น Amp หรือ Phase ส่วนแกน y จะเป็น เวลา/ลำดับ
    tilte => ชื่อ plot
    """

    # สร้างแกนเวลา (เช่น row index)
    time = np.arange(csi_data.shape[0])

    # Plot
    plt.figure(figsize=(15, 8))
    for i in range(csi_data.shape[1]):
        plt.plot(time, csi_data[:, i], label=f'Subcarrier {i}')

    plt.xlabel("Time (index)")
    plt.ylabel("Amplitude")
    plt.title(title)
    # plt.legend(loc='upper right', ncol=4, fontsize=7)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
