import bmp581
import time
import board
import busio
i2c = busio.I2C(board.SCL, board.SDA)
sensor = bmp581.BMP581(i2c,0x47)
started = False
idx = 1
readings = 0
starting_pressure = 0
while True:
	if idx <= 5 and not started:
		readings += sensor.pressure
		idx += 1
	elif not started:
		starting_pressure = round(readings/5,4)
		print(f"Standing Pressure = {starting_pressure}")
		started = True
	time.sleep(0.5)

