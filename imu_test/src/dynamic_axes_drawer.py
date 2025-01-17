#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import time
from utils import get_rotation_matrix, get_translation_matrix
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

class DynamicAxesDrawer:
    def __init__(self, axes_length=1.0):
        self.axes_length = axes_length
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_xlim([-self.axes_length, self.axes_length])
        self.ax.set_ylim([-self.axes_length, self.axes_length]) 
        self.ax.set_zlim([-self.axes_length, self.axes_length])
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')

    def update_axes(self, angles):
        self.ax.cla()
        self.ax.set_xlim([-self.axes_length, self.axes_length])
        self.ax.set_ylim([-self.axes_length, self.axes_length])
        self.ax.set_zlim([-self.axes_length, self.axes_length])
        
        angle_text = f'Roll: {angles[0]:.1f}°\nPitch: {angles[1]:.1f}°\nYaw: {angles[2]:.1f}°'
        self.ax.text2D(0.02, 0.98, angle_text, transform=self.ax.transAxes, fontsize=10)
        
        origin = np.array([0, 0, 0])
        m_rotation = get_rotation_matrix(np.deg2rad(angles)).T

        xyz_axis = np.eye(4) * self.axes_length
        xyz_axis[3, 3] = 1
        xyz_axis = xyz_axis @ m_rotation

        x_axis = xyz_axis[0, 0:3]
        y_axis = xyz_axis[1, 0:3]
        z_axis = xyz_axis[2, 0:3]

        self.ax.plot([origin[0], x_axis[0]], [origin[1], x_axis[1]], [origin[2], x_axis[2]], color='r', label='X-axis')
        self.ax.plot([origin[0], y_axis[0]], [origin[1], y_axis[1]], [origin[2], y_axis[2]], color='g', label='Y-axis')
        self.ax.plot([origin[0], z_axis[0]], [origin[1], z_axis[1]], [origin[2], z_axis[2]], color='b', label='Z-axis')
        self.ax.legend()

        self.ax.quiver(origin[0], origin[1], origin[2], 
                      x_axis[0], x_axis[1], x_axis[2], 
                      color='r', alpha=0.8, lw=3)
        self.ax.quiver(origin[0], origin[1], origin[2],
                      y_axis[0], y_axis[1], y_axis[2],
                      color='g', alpha=0.8, lw=3)
        self.ax.quiver(origin[0], origin[1], origin[2],
                      z_axis[0], z_axis[1], z_axis[2],
                      color='b', alpha=0.8, lw=3)

        x, y, z = 1.0, 0.5, 0.5
        vertices = np.array([
            [-x, -y, -z], [x, -y, -z], [x, y, -z], [-x, y, -z],
            [-x, -y, z], [x, -y, z], [x, y, z], [-x, y, z]
        ])
        
        rotated_vertices = (vertices @ m_rotation[:3, :3])
        
        faces = [
            [rotated_vertices[j] for j in [0, 1, 2, 3]],  # bottom
            [rotated_vertices[j] for j in [4, 5, 6, 7]],  # top
            [rotated_vertices[j] for j in [0, 1, 5, 4]],  # front
            [rotated_vertices[j] for j in [2, 3, 7, 6]],  # back
            [rotated_vertices[j] for j in [0, 3, 7, 4]],  # left
            [rotated_vertices[j] for j in [1, 2, 6, 5]]   # right
        ]
        
        poly = Poly3DCollection(faces, alpha=0.25, facecolor='gray', edgecolor='gray')
        self.ax.add_collection3d(poly)
        
        plt.draw()

