import cv2
import numpy as np

def take_photo():
    # Initialize webcam with optimized settings
    cap = cv2.VideoCapture(2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Lower resolution for speed
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer size
    cap.set(cv2.CAP_PROP_FPS, 30)  # Set FPS

    if not cap.isOpened():
        print("Error: Could not open webcam") 
        return None

    # Skip first few frames to let camera adjust
    for _ in range(5):
        cap.grab()

    # Capture frame
    ret, frame = cap.read()
    
    # Release webcam immediately
    cap.release()

    if not ret:
        print("Error: Could not capture frame")
        return None

    return frame

if __name__ == "__main__":
    # Take photo
    image = take_photo()
    
    if image is not None:
        # Use optimized image writing
        cv2.imwrite("left_mac.png", image, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        print("Photo saved as left_mac.png")
    else:
        print("Failed to take photo")
