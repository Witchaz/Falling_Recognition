import pcap
import dpkt
import keyboard
import pandas as pd
import numpy as np
import os
import sys
import time
import keyboard
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from multiprocessing import Process
from matplotlib.artist import Artist
from scapy.all import ARP, Ether, srp
from datetime import datetime
from threading import Thread

# Global configuration
BANDWIDTH = 20
NSUB = int(BANDWIDTH * 3.2)
SELECTED_MAC = '5c0214fb6552'
SHOW_PACKET_LENGTH = 100
GAP_PACKET_NUM = 20

#Flag
test_mode = 0


# Create CSI data folder
CSI_FOLDER = 'csi_data'
if not os.path.exists(CSI_FOLDER):
    os.makedirs(CSI_FOLDER)
    print(f"Created directory: {CSI_FOLDER}")

def get_mac():
    """
    Scans the network to find MAC addresses.
    Returns the first MAC address found without colons.
    """
    target_ip = "192.168.1.1/16"
    arp = ARP(pdst=target_ip)
    ether = Ether(dst="ff:ff:ff:ff:ff:ff")
    packet = ether / arp
    result = srp(packet, timeout=3, verbose=False)[0]
    
    for sent, received in result:
        print(f"IP: {received.psrc}, MAC: {received.hwsrc}")
    return result[0][1].hwsrc.replace(":","") if result else None

def truncate(num, n):
    """
    Truncates a number to n decimal places.
    """
    integer = int(num * (10 ** n)) / (10 ** n)
    return float(integer)


def append_to_csv(filepath, timestamp, csi_data):
    """
    Appends new CSI data to the CSV file in the csi_data folder.
    """
    # Create a new row with timestamp and CSI data
    
    new_row = [timestamp] + csi_data
    df = pd.DataFrame([new_row], columns=['timestamp'] + [f'subcarrier_{i}' for i in range(NSUB)])
    
    # Append to CSV without writing the header
    df.to_csv(filepath, mode='a', header=False, index=False)

def setup_plot():
    """
    Sets up the matplotlib plot for real-time visualization.
    Returns the figure, axis, line list, and text object.
    """
    plt.ion()
    fig, ax = plt.subplots(figsize=(12, 8))
    
    x = np.arange(0, SHOW_PACKET_LENGTH, 1)
    y_list = [[0 for _ in range(SHOW_PACKET_LENGTH)] for _ in range(NSUB)]
    line_list = []
    
    for y in y_list:
        line, = ax.plot(x, y, alpha=0.5)
        line_list.append(line)
    
    plt.title(f'{SELECTED_MAC}', fontsize=18)
    plt.ylabel('Signal Amplitude', fontsize=16)
    plt.xlabel('Packet', fontsize=16)
    plt.ylim(0, 1500)
    
    txt = ax.text(40, 1600, 'Amp Min-Max Gap: None', fontsize=14)
    
    return fig, ax, line_list, txt, y_list

def process_csi_data(csi, bandwidth):
    """
    Processes raw CSI data into complex numbers and returns complex CSI.
    """
    nsub = int(bandwidth * 3.2)
    
    # Convert CSI bytes to numpy array
    csi_np = np.frombuffer(csi, dtype=np.int16, count=nsub * 2)
    csi_np = csi_np.reshape((1, nsub * 2))
    
    # Convert to complex numbers (real + j*imag)
    csi_cmplx = np.fft.fftshift(csi_np[:1, ::2] + 1.j * csi_np[:1, 1::2], axes=(1,))
    
    return csi_cmplx[0]  # Return as 1D array of complex numbers


def update_plot(line_list, y_list, csi_data, minmax, gap_count, txt, ax):
    """
    Updates the plot with new CSI data and min-max gap information.
    """
    for i, y in enumerate(y_list):
        del y[0]
        new_y = csi_data[i]
        y.append(new_y)
        line_list[i].set_xdata(np.arange(0, SHOW_PACKET_LENGTH, 1))
        line_list[i].set_ydata(y)
        
        # Update min-max values
        if gap_count == 0:
            minmax.append([new_y, new_y])
        else:
            if minmax[i][0] > new_y:
                minmax[i][0] = new_y
            if minmax[i][1] < new_y:
                minmax[i][1] = new_y
    
    # Calculate and display gap
    gap_list = [mm[1] - mm[0] for mm in minmax]
    gap = max(gap_list)
    
    Artist.remove(txt)
    txt = ax.text(40, 1600, f'Amp Min-Max Gap: {gap}', fontsize=14)
    
    return txt, minmax, (gap_count + 1) % GAP_PACKET_NUM

def setup_csv_file():
    """
    Creates a new CSV file with timestamp in the csi_data folder.
    Returns the file path and DataFrame.
    """
    files = os.listdir(CSI_FOLDER)
    for i in files:
        print(i)

    dataset_folder = input('Enter Label : ')
    
    if dataset_folder == "Test":
        global test_mode
        test_mode = 1
        return f"{CSI_FOLDER}/" , None

    file_location = (CSI_FOLDER + "/" + dataset_folder) 
    if not os.path.exists(file_location):
        os.makedirs(file_location)
        print(f"Created directory: {file_location}")

    count = len([name for name in os.listdir(file_location) if os.path.isfile(os.path.join(file_location, name))])
    filename = f"{dataset_folder}_{count}.csv"
    filepath = os.path.join(file_location, filename)
    
    # Create DataFrame with subcarrier columns
    columns = ['timestamp'] + [f'subcarrier_{i}' for i in range(NSUB)]
    df = pd.DataFrame(columns=columns)
    df.to_csv(filepath, index=False)
    
    print(f"Created new CSV file: {filepath}")
    return filepath, df
def sniffing(nicname, mac_address):
    """
    Main function that captures and processes CSI data in real-time.
    """
    print(f'Start Sniffing... @ {nicname}, UDP, Port 5500')
    sniffer = pcap.pcap(name=nicname, promisc=True, immediate=True, timeout_ms=50)
    sniffer.setfilter('udp and port 5500')
    
    # Setup CSV file for recording
    csv_filepath, _ = setup_csv_file()
    print(f"Recording CSI data to {csv_filepath}")
    
    before_ts = 0.0
    fig, ax, line_list, txt, y_list = setup_plot()
    minmax = []
    gap_count = 0
    idx = SHOW_PACKET_LENGTH - 1

    start = time.time()

    for ts, pkt in sniffer:
        # Skip duplicate timestamps
        if int(ts) == int(before_ts):
            cur_ts = truncate(ts, 1)
            bef_ts = truncate(before_ts, 1)
            if cur_ts == bef_ts:
                before_ts = ts
                continue
        
        # Process packet
        eth = dpkt.ethernet.Ethernet(pkt)
        ip = eth.data
        udp = ip.data
        
        # Check MAC address
        mac = udp.data[4:10].hex()
        if mac != mac_address:
            continue
        
        # Extract and process CSI data
        csi = udp.data[18:]
        bandwidth = ip.__hdr__[2][2]
        # Get CSI complex values
        csi_cmplx = process_csi_data(csi, bandwidth)
        # Separate real and imag for saving
        csi_real = csi_cmplx.real
        csi_imag = csi_cmplx.imag

        csi_complex_list = [complex(r, i) for r, i in zip(csi_real, csi_imag)]

        # Optional: combine or save separately
        # หรือเก็บเป็น complex string: [str(val) for val in csi_cmplx]
        # Record to CSV
        if test_mode == 0 :
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
            append_to_csv(csv_filepath, current_time, csi_complex_list)
        
        # Update plot
        idx += 1
        txt, minmax, gap_count = update_plot(line_list, y_list, np.abs(csi_cmplx), minmax, gap_count, txt, ax)

        
        # Update display
        fig.canvas.draw()
        fig.canvas.flush_events()
        before_ts = ts
        
        end = time.time() - start 
        
        if test_mode == 0 and end >= 20 or keyboard.is_pressed('q'):
            print(f"{'-*' * 30} \n Stop Recording....\n record for {end:.2f} seconds \n{'-*' * 30} ")
            exit()


if __name__ == '__main__':
    t = Thread(target=sniffing('wlan0', SELECTED_MAC), daemon=True)
    t.start()
