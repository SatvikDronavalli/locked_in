// THIS CODE WORKS
import smbus2
import time

# I2C address for ISM330DHCX (default is 0x6A)
I2C_ADDRESS = 0x6B
bus = smbus2.SMBus(1)

# Register Addresses (from ISM330DHCX datasheet)
CTRL1_XL = 0x10  # Accelerometer control
CTRL2_G = 0x11  # Gyroscope control
CTRL3_C = 0x12  # Control register
OUTX_L_A = 0x28  # X-axis acceleration LSB
OUTX_L_G = 0x22  # X-axis gyroscope LSB

# Configure Accelerometer: 104 Hz, ±4g, High-Performance mode
bus.write_byte_data(I2C_ADDRESS, CTRL1_XL, 0b01001000)

# Configure Gyroscope: 104 Hz, ±250 dps, High-Performance mode
bus.write_byte_data(I2C_ADDRESS, CTRL2_G, 0b01001000)

# Enable block data update & auto-increment
bus.write_byte_data(I2C_ADDRESS, CTRL3_C, 0b01000100)

def read_sensor_data(register):
    """ Read two bytes of data from a register and combine them """
    low, high = bus.read_i2c_block_data(I2C_ADDRESS, register, 2)
    value = (high << 8) | low
    if value > 32767:
        value -= 65536  # Convert to signed integer
    return value

def main():
    while True:
        ax = read_sensor_data(OUTX_L_A) * 0.000122  # Convert to m/s² (±4g scale)
        ay = read_sensor_data(OUTX_L_A + 2) * 0.000122
        az = read_sensor_data(OUTX_L_A + 4) * 0.000122

        gx = read_sensor_data(OUTX_L_G) * 0.00875  # Convert to degrees/s (±250 dps)
        gy = read_sensor_data(OUTX_L_G + 2) * 0.00875
        gz = read_sensor_data(OUTX_L_G + 4) * 0.00875

        print(f"Acceleration (m/s²): X: {ax:.2f}, Y: {ay:.2f}, Z: {az:.2f}")
        print(f"Gyroscope (°/s):      X: {gx:.2f}, Y: {gy:.2f}, Z: {gz:.2f}\n")

        time.sleep(0.5)
