import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

class DynamicPlotDrawer:
	def __init__(self):
		# Create one figure with three subplots
		self.fig, (self.ax_angle, self.ax_accel, self.ax_angle_accel) = plt.subplots(3, 1, figsize=(10, 12))
		
		self.max_points = 100
		self.times = np.linspace(0, 10, self.max_points)
		
		# Initialize data arrays
		self.roll = np.full(self.max_points, np.nan)
		self.pitch = np.full(self.max_points, np.nan)
		self.yaw = np.full(self.max_points, np.nan)
		self.accel_x = np.full(self.max_points, np.nan)
		self.accel_y = np.full(self.max_points, np.nan)
		self.accel_z = np.full(self.max_points, np.nan)
		self.angle_accel_x = np.full(self.max_points, np.nan)
		self.angle_accel_y = np.full(self.max_points, np.nan)
		self.angle_accel_z = np.full(self.max_points, np.nan)
		self.length = 0

		# Create lines for all three plots
		self.lines_angle = [
			self.ax_angle.plot(self.times, self.roll, 'b-', label="Roll", animated=True)[0],
			self.ax_angle.plot(self.times, self.pitch, 'g-', label="Pitch", animated=True)[0],
			self.ax_angle.plot(self.times, self.yaw, 'r-', label="Yaw", animated=True)[0],
		]
		self.lines_accel = [
			self.ax_accel.plot(self.times, self.accel_x, 'b-', label="X", animated=True)[0],
			self.ax_accel.plot(self.times, self.accel_y, 'g-', label="Y", animated=True)[0],
			self.ax_accel.plot(self.times, self.accel_z, 'r-', label="Z", animated=True)[0],
		]
		self.lines_angle_accel = [
			self.ax_angle_accel.plot(self.times, self.angle_accel_x, 'b-', label="X", animated=True)[0],
			self.ax_angle_accel.plot(self.times, self.angle_accel_y, 'g-', label="Y", animated=True)[0],
			self.ax_angle_accel.plot(self.times, self.angle_accel_z, 'r-', label="Z", animated=True)[0],
		]

		# Configure plots
		for ax, title, ylabel, ylim in [
			(self.ax_angle, "Euler Angles", "Angle [deg]", (-180, 180)),
			(self.ax_accel, "Linear Acceleration", "Acceleration [m/s²]", (-5, 5)),
			(self.ax_angle_accel, "Angular Acceleration", "Angular Acceleration [deg/s²]", (-2000, 2000))
		]:
			ax.set_xlim(0, 10)
			ax.set_ylim(*ylim)
			ax.set_xlabel("Time [s]")
			ax.set_ylabel(ylabel)
			ax.set_title(title)
			ax.legend()
			ax.grid(True)

		# Adjust layout to prevent overlap
		self.fig.tight_layout()
		
		# Enable blitting
		self.fig.canvas.draw()
		self.bg = self.fig.canvas.copy_from_bbox(self.fig.bbox)
		
		# Create single animation
		self.ani = FuncAnimation(self.fig, self.update_plot, interval=20, blit=True)
		plt.show(block=False)

	def update_plot(self, frame):
		self.fig.canvas.restore_region(self.bg)
		
		# Update angle plot
		for line, data in zip(self.lines_angle, [self.roll, self.pitch, self.yaw]):
			line.set_ydata(data)
			self.ax_angle.draw_artist(line)
		
		# Update acceleration plot
		for line, data in zip(self.lines_accel, [self.accel_x, self.accel_y, self.accel_z]):
			line.set_ydata(data)
			self.ax_accel.draw_artist(line)
		
		# Update angular acceleration plot
		for line, data in zip(self.lines_angle_accel, [self.angle_accel_x, self.angle_accel_y, self.angle_accel_z]):
			line.set_ydata(data)
			self.ax_angle_accel.draw_artist(line)
		
		self.fig.canvas.blit(self.fig.bbox)
		return self.lines_angle + self.lines_accel + self.lines_angle_accel

	def update(self, angles, accels, angle_accels):
		if self.length < self.max_points:
			idx = self.length
			self.length += 1
		else:
			# Shift all data arrays
			self.roll[:-1] = self.roll[1:]
			self.pitch[:-1] = self.pitch[1:]
			self.yaw[:-1] = self.yaw[1:]
			self.accel_x[:-1] = self.accel_x[1:]
			self.accel_y[:-1] = self.accel_y[1:]
			self.accel_z[:-1] = self.accel_z[1:]
			self.angle_accel_x[:-1] = self.angle_accel_x[1:]
			self.angle_accel_y[:-1] = self.angle_accel_y[1:]
			self.angle_accel_z[:-1] = self.angle_accel_z[1:]
			idx = -1

		# Update latest values
		self.roll[idx], self.pitch[idx], self.yaw[idx] = angles
		self.accel_x[idx], self.accel_y[idx], self.accel_z[idx] = accels
		self.angle_accel_x[idx], self.angle_accel_y[idx], self.angle_accel_z[idx] = angle_accels
