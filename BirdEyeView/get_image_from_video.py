import cv2

# Open video file
cap = cv2.VideoCapture('road_video.mp4')

# Check if video opened successfully
if not cap.isOpened():
    print("Error: Could not open video file")
    exit()

# Read the first frame
ret, frame = cap.read()

if ret:
    # Save the first frame as an image
    cv2.imwrite('road_image.png', frame)
    print("First frame saved as road_image.png")
else:
    print("Error: Could not read frame from video")

# Release video capture object
cap.release()
