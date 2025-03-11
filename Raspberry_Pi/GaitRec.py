import bmp581
import time
import board
import busio
import numpy as np
from scipy.signal import butter, filtfilt

# Setup BMP581 sensor
i2c = busio.I2C(board.SCL, board.SDA)
sensor = bmp581.BMP581(i2c, 0x47)

# Butterworth low-pass filter
def butter_lowpass_filter(data, cutoff=20, fs=250, order=2):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

# Calibration step: Measure min/max pressure
print("Get ready for initialization: foot in air")
time.sleep(5)
print("Initializing air pressure...")
time.sleep(1)
xmin = np.mean([sensor.pressure for _ in range(10)])
time.sleep(5)

print("Get ready for initialization: foot on ground")
time.sleep(5)
print("Initializing ground pressure...")
time.sleep(1)
xmax = np.mean([sensor.pressure for _ in range(10)])

print(f"Min force: {xmin}, Max force: {xmax}")

# Collect data
sampling_rate = 250  # Hz
data_buffer = []
start_time = time.time()

print("Recording gait data...")
while time.time() - start_time < 10:  # Record for 10 seconds
    pressure = sensor.pressure
    normalized_force = (pressure - xmin) / (xmax - xmin)  # Normalize
    data_buffer.append(normalized_force)
    time.sleep(1 / sampling_rate)  # Maintain sampling rate

# Apply filtering
filtered_data = butter_lowpass_filter(data_buffer, cutoff=20, fs=sampling_rate)

# Time normalization
normalized_time_series = np.interp(np.linspace(0, len(filtered_data), 101),
                                   np.arange(len(filtered_data)), filtered_data)

# Save data to file
np.savetxt("gait_pressure_data.csv", normalized_time_series, delimiter=",")
print("Data collection complete. Saved to 'gait_pressure_data.csv'.")
