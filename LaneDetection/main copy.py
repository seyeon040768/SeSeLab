import numpy as np
import cv2
import matplotlib.pyplot as plt
import time

if __name__ == "__main__":
    data_path = "data/video.mp4"

    cap = cv2.VideoCapture(data_path)
    if not cap.isOpened():
        print("Error: Could not open video file")
        exit()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            exit()
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h = hsv[:, :, 0]
        threshold_value = 200  # 0~255 사이의 값
        _, thresh = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
        _, thresh_h = cv2.threshold(h, 70, 255, cv2.THRESH_BINARY)

        kernel = np.ones((5,5), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        closing_h = cv2.morphologyEx(thresh_h, cv2.MORPH_ERODE, kernel)

        delete_noise = np.clip(opening.astype(np.int32) - closing_h, 0, 255).astype(np.uint8)
        
        result = np.hstack((gray, opening))
        result1 = np.hstack((h, delete_noise))
        result2 = np.hstack((closing_h, dst))

        result = np.vstack((result, result1, result2))

        cv2.imshow('Frame', result)

        time.sleep(0.01)
        
        # Wait for 'q' key to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break