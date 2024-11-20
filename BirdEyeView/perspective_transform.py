import cv2
import numpy as np

def get_four_clicks(img: np.ndarray) -> np.ndarray:
    """
    이미지에서 4개의 좌표를 클릭하여 반환합니다.
    param:
        img: 이미지
    return:
        points: 4개의 좌표
    """
    points = []

    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            print(f"Clicked position: ({x}, {y})")
            
            if len(points) == 4:
                cv2.destroyAllWindows()

    if img is None:
        print("Image not found!")
        return None

    cv2.imshow("Image", img)
    cv2.setMouseCallback("Image", click_event)

    cv2.waitKey(0)
    
    return np.float32(points)

height = 1000
width = 500
image_path = "./road_image.png"

img = cv2.imread(image_path)
points = get_four_clicks(img)

dst_points = np.float32([
    [0, 0],
    [width, 0],
    [width, height],
    [0, height]
])

print(points, dst_points)

matrix = cv2.getPerspectiveTransform(points, dst_points)

result = cv2.warpPerspective(img, matrix, (width, height))

cv2.imshow("", result)
cv2.waitKey(0)
cv2.destroyAllWindows()

