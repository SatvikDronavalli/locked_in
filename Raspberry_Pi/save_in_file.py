import time
import board
import busio
import numpy as np
from adafruit_lsm6ds import ism330dhcx as ism
import bmp581
import math
import pandas as pd

# Initialize I2C and sensors
i2c = busio.I2C(board.SCL, board.SDA)
accel_sensor = ism.ISM330DHCX(i2c, 0x6B)
force_sensor = bmp581.BMP581(i2c, 0x47)
time.sleep(20)
velocity = 0
prev_time = time.time()
strike_detected = False
step_count = 0
cadence = 0
stride_lengths = []
step_timestamps = []

# Thresholds to filter noise and avoid unnecessary integration
THRESHOLD_FORCE = 0.75  # Adjust based on real testing
THRESHOLD_ACCEL = 5  # Ignore acceleration below this value
TIME_STEP = 0.5  # Sampling time step
ZERO_VELOCITY_THRESHOLD = 0.05  # If velocity is very small, set to zero to avoid drift
CADENCE_WINDOW = 10  # Time window (in seconds) to calculate cadence
def write_to_file(df,metric,value,idx):
    if metric not in df.columns:
        df[metric] = range(10)
    else:
        df.loc[idx,metric] = value
    return df

df = pandas.DataFrame(index=range(10))
started = False
idx = 1
readings = 0
starting_pressure = 0
'''
print("get ready for initializing 0")
time.sleep(3)
print("init started")
time.sleep(1)

#==========foot in air=================
xmin = None
sum1 = 0
for i in range(0, 10):
    sum1 += force_sensor.pressure
    time.sleep(0.5)
xmin = round(sum1/10,5)
print("get ready for initializing 1")
time.sleep(3)
print("init started")
time.sleep(1)

#==========foot on ground==============
xmax = None
sum2 = 0
for i in range(0, 10):
    sum2 += force_sensor.pressure
    time.sleep(0.5)
xmax = round(sum2/10,5)'''
xmin = 100
xmax = 102
print(f"min force is {xmin} and max force is {xmax}")
n = lambda x: round((x-xmin)/(xmax-xmin),5)


def integrate_acceleration(acceleration, time_interval):
    """Numerically integrates acceleration to estimate velocity."""
    return acceleration * time_interval

first = False
initialized = False
iterations = 0
# Open the file in append mode, or 'w' for overwriting
with open("output_data.txt", "w") as file:
    while iterations < 10:
        current_time = time.time()
        time_interval = current_time - prev_time
        prev_time = current_time

        # Read accelerometer data
        accel_x, accel_y, accel_z = accel_sensor.acceleration
        accel_x = round(accel_x, 2)
        accel_y = round(accel_y, 2)
        accel_z = round(accel_z, 2)

        if not first:
            file.write("wait\n")
            print("wrote 'wait' in file")
            first = True
            start_net_accel = math.sqrt(accel_x**2 + accel_y**2 + accel_z**2)  
            time.sleep(5)

        # Read force sensor data
        if initialized:
            force_1 = force_2
            n_force_1 = n_force_2
            force_2 = force_sensor.pressure
            n_force_2 = force_sensor.pressure
        if not initialized:
            force_2 = force_sensor.pressure
            n_force_2 = n(force_2)
            initialized = True
            continue

        # Detect foot strike (step event)
        if n_force_2 > n_force_1 + 0.3 and not strike_detected:
            strike_detected = True
            start_time = current_time
            stride_velocity = 0  # Reset integrated velocity for stride length calculation

        # Detect foot off (end of stride)
        elif n_force_2 < n_force_1 - 0.3 and strike_detected:
            strike_detected = False
            stride_time = current_time - start_time
            stride_length = stride_velocity * stride_time  # Integrating velocity over stride duration
            stride_lengths.append(stride_length)
            file.write(f"Stride Length: {stride_length:.2f} meters\n")

        # Integrate acceleration if above noise threshold
        if abs(accel_x) > THRESHOLD_ACCEL:
            velocity += integrate_acceleration(max(math.sqrt(accel_x**2 + accel_y**2 + accel_z**2)-start_net_accel,0), time_interval)

        # Prevent velocity drift from noise
        if abs(velocity) < ZERO_VELOCITY_THRESHOLD:
            velocity = 0

        # Accumulate velocity over stride duration for stride length calculation
        if strike_detected:
            file.write("STRIKE DETECTED\n")
            strike_detected = False
            print("wrote 'strike detected' in file")
            stride_velocity += velocity * time_interval

        # Calculate cadence (steps per minute)
        step_timestamps = [t for t in step_timestamps if current_time - t <= CADENCE_WINDOW]
        if len(step_timestamps) > 1:
            cadence = (len(step_timestamps) / CADENCE_WINDOW) * 60  # Convert to steps per minute
        df = write_to_file(df,"cadence",iterations,iterations)
        file.write(f"Velocity: {velocity:.2f} m/s | Cadence: {cadence:.2f} steps/min\n")
        file.write(f"Acceleration: {accel_sensor.acceleration}\n")
        file.write(f"Normalized Force: {n_force_2}\n")

        time.sleep(TIME_STEP)
        iterations += 1
print('-------------------------------------------------')
print(df["cadence"])
df.to_csv("metrics.csv")
