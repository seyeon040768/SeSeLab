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
        s = hsv[:, :, 1]

        threshold_value = 200
        _, thresh = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)

        kernel = np.ones((5,5), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        contour = np.zeros_like(gray)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 1000:
                max_xy = np.max(cnt, axis=0)[0]
                min_xy = np.min(cnt, axis=0)[0]

                width, height = max_xy - min_xy
                ratio = area / height

                if ratio < 25:
                    cv2.drawContours(contour, [cnt], -1, 255, thickness=cv2.FILLED)
        
        #result = cv2.Canny(opening, 150, 255)

        row1 = np.hstack((gray, thresh))
        row2 = np.hstack((opening, contour))
        row3 = np.hstack((h, s))

        result = np.vstack((row1, row2, row3))

        cv2.imshow('Frame', result)

        time.sleep(0.01)
        
        # Wait for 'q' key to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break