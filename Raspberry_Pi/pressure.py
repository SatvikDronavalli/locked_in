import bmp581
import time
import board
import busio
i2c = busio.I2C(board.SCL, board.SDA)
sensor = bmp581.BMP581(i2c,0x47)
while True:
    print(sensor.pressure)66
    time.sleep(0.5)
