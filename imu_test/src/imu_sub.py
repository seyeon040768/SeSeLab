#!/usr/bin/env python3

import rclpy
import numpy as np
from rclpy.node import Node
from rclpy.qos import QoSProfile
from geometry_msgs.msg import Vector3
from imu_test.msg import IMUData
import matplotlib.pyplot as plt
from dynamic_plot_drawer import DynamicPlotDrawer
from dynamic_axes_drawer import DynamicAxesDrawer
from position_estimation import PositionEstimator

class EbimuSubscriber(Node):
	def __init__(self):
		super().__init__('ebimu_subscriber')
		qos_profile = QoSProfile(depth=10)
		self.subscription = self.create_subscription(
			IMUData, 
			'ebimu_data', 
			self.callback, 
			qos_profile
		)
		self.plot_drawer = DynamicPlotDrawer()
		# self.axes_drawer = DynamicAxesDrawer(axes_length=3.0)
		self.position_estimator = PositionEstimator()
		
		self.create_timer(0.05, self.timer_callback)

	def timer_callback(self):
		plt.pause(0.0001)  # Reduced pause time

	def callback(self, msg):
		imu_data = msg
		angle = np.array([imu_data.angle.x, imu_data.angle.y, imu_data.angle.z])
		angle_accel = np.array([imu_data.angle_accel.x, imu_data.angle_accel.y, imu_data.angle_accel.z])
		accel = np.array([imu_data.accel.x, imu_data.accel.y, imu_data.accel.z])

		# print(f"Euler angles (roll, pitch, yaw): {angle}")
		# print(f"Angular Acceleration (x, y, z): {angle_accel}")
		# print(f"Linear Acceleration (x, y, z): {accel}")

		self.plot_drawer.update(angle, accel, angle_accel)
		# self.axes_drawer.update_axes(angle)

		timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
		self.position_estimator.update(timestamp, accel)

		print("position:", self.position_estimator.position)
		print("velocity:", self.position_estimator.velocity)
		print()



def main(args=None):
	rclpy.init(args=args)

	print("Starting ebimu_subscriber..")

	node = EbimuSubscriber()

	try:
		rclpy.spin(node)

	finally:
		node.destroy_node()
		rclpy.shutdown()


if __name__ == '__main__':
	main()
