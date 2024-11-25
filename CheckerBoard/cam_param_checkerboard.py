import cv2
import numpy as np
import matplotlib.pyplot as plt

def find_corners_from_checkerboard(image, checkerboard_shape):
    if (len(image.shape) == 3):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    ret, corners = cv2.findChessboardCorners(image, checkerboard_shape, None)

    if ret:
        return corners
    else:
        return None

if __name__ == "__main__":
    pass