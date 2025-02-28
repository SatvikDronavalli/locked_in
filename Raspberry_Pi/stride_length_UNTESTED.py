import time
import board
import busio
import numpy as np
from adafruit_lsm6ds import ism330dhcx as ism
import bmp581

i2c = busio.I2C(board.SCL, board.SDA)

accel_sensor = ism.ISM330DHCX(i2c, 0x6B)
force_sensor = bmp581.BMP581(i2c, 0x47)

velocity = 0
stride_lengths = []
prev_time = time.time()
strike_detected = False

THRESHOLD_FORCE = 5  # CHANGE THIS BASED ON WHAT WE FIND IS BEST WHEN TESTING
THRESHOLD_ACCEL = 0.1  # CHANGE THIS BASED ON WHAT WE FIND IS BEST WHEN TESTING
TIME_STEP = 0.05  # CHANGE THIS BASED ON WHAT WE FIND IS BEST WHEN TESTING

def integrate_acceleration(acceleration, time_interval):
    """Numerically integrates acceleration to estimate velocity."""
    return acceleration * time_interval

while True:
    current_time = time.time()
    time_interval = current_time - prev_time
    prev_time = current_time

    accel_x, accel_y, accel_z = accel_sensor.acceleration

    accel_x = round(accel_x, 1)
    accel_y = round(accel_y, 1)
    accel_z = round(accel_z, 1)

    force = force_sensor.pressure

    if force > THRESHOLD_FORCE and not strike_detected:
        strike_detected = True
        initial_velocity = velocity
        start_time = current_time

    elif force < THRESHOLD_FORCE and strike_detected:
        strike_detected = False
        stride_time = current_time - start_time
        stride_length = initial_velocity * stride_time
        stride_lengths.append(stride_length)
        print(f"Stride Length: {stride_length:.2f} meters")

    if abs(accel_x) > THRESHOLD_ACCEL:
        velocity += integrate_acceleration(accel_x, time_interval)

    time.sleep(TIME_STEP)
