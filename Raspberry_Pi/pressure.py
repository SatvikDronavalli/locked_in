// THIS CODE DOES NOT WORK
import smbus2
import time

# I2C address for BMP581
I2C_ADDRESS = 0x47
bus = smbus2.SMBus(1)

# Register Addresses (from BMP581 datasheet)
CMD_RESET = 0x7E      # Reset command register
SOFT_RESET_CMD = 0xB6  # Soft reset command
OSR_CONFIG = 0x36     # Oversampling config
ODR_CONFIG = 0x37     # Output Data Rate & mode
STATUS_REGISTER = 0x28  # Status register (checks if data is ready)
TEMP_XLSB = 0x1D      # Temperature XLSB
PRESS_XLSB = 0x20     # Pressure XLSB

# 🚀 **Step 1: Soft Reset & Configure BMP581**
print("Performing soft reset...")
bus.write_byte_data(I2C_ADDRESS, CMD_RESET, SOFT_RESET_CMD)
time.sleep(0.1)  # Wait for reset to complete

# **Step 2: Configure BMP581**
print("Configuring BMP581...")
bus.write_byte_data(I2C_ADDRESS, OSR_CONFIG, 0x33)  # Enable both pressure & temperature oversampling
bus.write_byte_data(I2C_ADDRESS, ODR_CONFIG, 0x02)  # Set FORCED mode

def trigger_forced_measurement():
    """ Trigger a single pressure & temperature measurement in FORCED mode. """
    bus.write_byte_data(I2C_ADDRESS, ODR_CONFIG, 0x02)  # Re-trigger FORCED mode
    time.sleep(0.01)  # Short delay for measurement

def read_sensor_data(register, length):
    """ Read multiple bytes from a register. """
    return bus.read_i2c_block_data(I2C_ADDRESS, register, length)

def read_pressure_temperature():
    """ Read and convert the pressure and temperature from the BMP581. """
    # Trigger new measurement
    trigger_forced_measurement()

    # Check status register to see if new data is available
    status = bus.read_byte_data(I2C_ADDRESS, STATUS_REGISTER)
    if not (status & 0b00000001):  # Bit 0 = Pressure data ready
        print("⚠️ Warning: Pressure data not ready!")
        return None, None

    # Read 6 bytes (Temperature: 0x1D-0x1F, Pressure: 0x20-0x22)
    data = read_sensor_data(TEMP_XLSB, 6)
    print(f"Raw Data: {data}")

    # Extract temperature (3 bytes)
    raw_temp = (data[2] << 16) | (data[1] << 8) | data[0]
    if raw_temp & (1 << 23):  # Convert 24-bit signed integer
        raw_temp -= (1 << 24)
    temperature = raw_temp / 65536.0  # Convert to Celsius

    # Extract pressure (3 bytes)
    raw_press = (data[5] << 16) | (data[4] << 8) | data[3]

    # Check for invalid pressure data (0x7F, 0x7F, 0x7F)
    if data[3] == 0x7F and data[4] == 0x7F and data[5] == 0x7F:
        print("⚠️ Warning: Received invalid pressure data (0x7F, 0x7F, 0x7F)")
        pressure = None
    else:
        pressure = raw_press / 64.0  # Convert to Pascals

    print(f"Raw Pressure: {raw_press} | Converted Pressure: {pressure} Pa")
    print(f"Raw Temperature: {raw_temp} | Converted Temperature: {temperature:.2f} °C\n")

    return pressure, temperature

# 🚀 **Main Loop**
while True:
    pressure, temperature = read_pressure_temperature()

    if pressure is None:
        print("Pressure data is invalid (sensor returned 0x7F, 0x7F, 0x7F)")
    else:
        print(f"Pressure (Pa): {pressure:.2f}")

    print(f"Temperature (°C): {temperature:.2f}\n")
    time.sleep(0.5)

