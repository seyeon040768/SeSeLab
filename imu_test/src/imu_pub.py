#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from geometry_msgs.msg import Vector3
from imu_test.msg import IMUData
import serial
import math

comport_num = "/dev/ttyUSB0"
comport_baudrate = 115200
# comport_num = '/dev/tty' + input("EBIMU Port: /dev/tty")
# comport_baudrate = input("Baudrate: ")
ser = serial.Serial(port=comport_num,baudrate=comport_baudrate)

try:
	ser = serial.Serial(port=comport_num, baudrate=comport_baudrate)
except:
	print('Serial port error!')


class EbimuPublisher(Node):
	def __init__(self):
		super().__init__('ebimu_publisher')
		qos_profile = QoSProfile(depth=10)

		self.publisher = self.create_publisher(IMUData, 'ebimu_data', qos_profile)
		timer_period = 0.0005
		self.timer = self.create_timer(timer_period, self.timer_callback)
		self.count = 0

	def timer_callback(self):
		msg = IMUData()

		msg.header.stamp = self.get_clock().now().to_msg()

		ser_data = ser.readline().decode('utf-8').split(',')
		print(ser_data)
		if -1 < ser_data[0].find('*'):
			ser_data[0] = ser_data[0].replace('*','')
		ser_data = list(map(float, ser_data))

		if len(ser_data) < 9:
			return
		
		quaternion = ser_data[0:4]
		
		sin_roll = 2.0 * (quaternion[3] * quaternion[1] - quaternion[2] * quaternion[0])
		sin_roll = max(min(sin_roll, 1.0), -1.0)
		yaw = math.atan2(2.0 * (quaternion[3] * quaternion[0] + quaternion[1] * quaternion[2]),
						 1.0 - 2.0 * (quaternion[0] * quaternion[0] + quaternion[1] * quaternion[1]))
		roll = math.asin(sin_roll)
		pitch = math.atan2(2.0 * (quaternion[3] * quaternion[2] + quaternion[0] * quaternion[1]),
						1.0 - 2.0 * (quaternion[1] * quaternion[1] + quaternion[2] * quaternion[2]))
		
		msg.angle.x, msg.angle.y, msg.angle.z = -math.degrees(roll), -math.degrees(pitch), -math.degrees(yaw)
		msg.angle_accel.y, msg.angle_accel.x, msg.angle_accel.z = ser_data[4:7]
		msg.accel.y, msg.accel.x, msg.accel.z = ser_data[7:10]
		self.publisher.publish(msg)



def main(args=None):
	rclpy.init(args=args)

	print("Starting ebimu_publisher..")

	node = EbimuPublisher()

	try:
		rclpy.spin(node)

	finally:
		node.destroy_node()
		rclpy.shutdown()


if __name__ == '__main__':
	main()
