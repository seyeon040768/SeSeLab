import numpy as np
import cv2
import os

def get_image_from_video(video_name: str, frame: int = 0) -> np.ndarray | None:
    """Extracts a single frame from a video file.

    Args:
        video_name (str): Path to the video file to extract frame from
        frame (int, optional): Frame number to extract from video. Defaults to 0.

    Returns:
        np.ndarray | None: The extracted frame as a numpy array, or None if there was an error
    """

    cap = cv2.VideoCapture(video_name)

    if not cap.isOpened():
        print("Error: Could not open video file")
        return None

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
    
    ret, image = cap.read()
    
    cap.release()
    
    if not ret:
        print("Error: Could not read frame")
        return None
        
    return image


if __name__ == "__main__":
    video_dir = "./data/videos"
    video_names = os.listdir(video_dir)
    image_dir = "./data/images"

    for video_path in video_names:
        frame = 0
        image = get_image_from_video(os.path.join(video_dir, video_path), frame)
        image_name = os.path.splitext(video_path)[0] + f"_frame{frame}.png"

        cv2.imwrite(os.path.join(image_dir, image_name), image)
        