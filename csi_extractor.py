import numpy as np
import matplotlib.pyplot as plt
import csiread

# โหลดไฟล์ที่ได้จาก Nexmon
csi_data = csiread.Nexmon()
csi_data.read()

# แสดง amplitude
amplitude = np.abs(csi_data)
plt.plot(amplitude)
plt.title("CSI Amplitude of Packet 0")
plt.xlabel("Subcarrier")
plt.ylabel("|CSI|")
plt.grid(True)
plt.show()
