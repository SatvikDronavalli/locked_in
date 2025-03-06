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
print("get ready for initializing 0")
time.sleep(5)
print("init started")
time.sleep(1)
#==========foot in air=================
xmin = None
sum1 = 0
for i in range(0, 10):
	sum1 += sensor.pressure
	time.sleep(0.5)
xmin = round(sum1/10,5)
print("get ready for initializing 1")
time.sleep(5)
print("init started")
time.sleep(1)
#==========foot on ground==============
xmax = None
sum2 = 0
for i in range(0, 10):
	sum2 += sensor.pressure
	time.sleep(0.5)
xmax = round(sum2/10,5)
print(f"min force is {xmin} and max force is {xmax}")
n = lambda x: round((x-xmin)/(xmax-xmin),5)
while True:
	x = sensor.pressure
	print(n(x))
	time.sleep(0.5)
