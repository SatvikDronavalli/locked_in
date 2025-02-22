import time
import board
import busio
from adafruit_lsm6ds import ism330dhcx as ism
i2c = busio.I2C(board.SCL, board.SDA)
sensor = ism.ISM330DHCX(i2c,0x6B)
while True:
    print("Acceleration: X:%.2f, Y: %.2f, Z: %.2f m/s^2" % (sensor.acceleration))
    print("Gyro X:%.2f, Y: %.2f, Z: %.2f radians/s" % (sensor.gyro))
    print("")
    time.sleep(0.5)
