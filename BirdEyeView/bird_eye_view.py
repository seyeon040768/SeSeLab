import numpy as np
import cv2
import json
import calibration

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

image_path = "./um_000012.png"
image = cv2.imread(image_path)
image_shape = (image.shape[1], image.shape[0])

rotation_degree = (90, -90, 0)
# translation = (0.06, -0.08, -0.27)
translation = (0.06, -7.631618000000e-02, -2.717806000000e-01)
fov_degree = 85.7
camera_height = 1.65
m_transformation = calibration.get_transformation_matrix(image_shape, np.deg2rad(fov_degree), np.deg2rad(rotation_degree), translation).T

x_range = (6.3, 100)
y_range = (-15, 15)

bev_pixel_interval = (0.05, 0.025)

output_shape = (int((x_range[1] - x_range[0]) / bev_pixel_interval[0]), int((y_range[1] - y_range[0]) / bev_pixel_interval[1]))

print(f"image shape: {image_shape}, output shape: {output_shape}")

map_matrix = np.zeros((output_shape[1], output_shape[0], 2), dtype=np.float32)
for i, x in enumerate(np.arange(x_range[0], x_range[1], bev_pixel_interval[0])):
    for j, y in enumerate(np.arange(y_range[0], y_range[1], bev_pixel_interval[1])):
        world_point = np.array([x, y, -camera_height, 1])
        projected_point = world_point @ m_transformation
        projected_point = projected_point[:2] / projected_point[2]

        map_matrix[j, i] = projected_point

# for i in range(output_shape[0]):
#     for j in range(output_shape[1]):
#         projected_point = map_matrix[j, i]
#         if 0 <= projected_point[0] < image_shape[0] and 0 <= projected_point[1] < image_shape[1]:
#             cv2.circle(image, (int(projected_point[0]), int(projected_point[1])), 1, (0, 0, 255), -1)

# cv2.imshow("image", image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

bev_image = np.zeros((output_shape[1], output_shape[0], 3), dtype=np.uint8)
for i in range(output_shape[0]):
    for j in range(output_shape[1]):
        projected_point = map_matrix[j, i]
        if 0 <= projected_point[0] < image_shape[0] and 0 <= projected_point[1] < image_shape[1]:

            bev_image[j, i] = image[int(projected_point[1]), int(projected_point[0])]

cv2.imwrite("./bev_image.png", bev_image)
