import numpy as np

def separate_amp_phase(csi_data: np.ndarray):
    """
    แยก Amplitude และ Phase ออกจาก CSI complex ndarray
    
    Parameters:
        csi_data (np.ndarray): ndarray ที่เป็น complex (เช่น shape: [time, subcarriers])
    
    Returns:
        amplitude (np.ndarray): Amplitude (shape เดียวกับ csi_data)
        phase (np.ndarray): Phase (radians, shape เดียวกับ csi_data)
    """
    amplitude = np.abs(csi_data)
    phase = np.angle(csi_data)  # phase เป็นค่าระหว่าง -π ถึง π
    return amplitude, phase
