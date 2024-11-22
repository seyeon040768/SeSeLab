import numpy as np
import cv2
import matplotlib.pyplot as plt
import calibration
import bird_eye_view

if __name__ == "__main__":
    image_path = "./um_000012.png"
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_shape = (image.shape[1], image.shape[0])

    rotation_degree = (90, -90, 0)
    translation = (0, 0, 0) # camera 위치 기준
    fov_degree = 85.7
    camera_height = 1.65
    m_transformation = calibration.get_transformation_matrix(image_shape, np.deg2rad(fov_degree), np.deg2rad(rotation_degree), translation).T

    x_range = (6.3, 50)
    y_range = (-10, 10)
    bev_pixel_interval = (0.05, 0.05)
    
    map_matrix = bird_eye_view.get_map_matrix(m_transformation, x_range, y_range, bev_pixel_interval, camera_height)
    bev_image = bird_eye_view.make_bev_image(image, map_matrix)
    hsl_image = cv2.cvtColor(bev_image, cv2.COLOR_RGB2HLS)

    green_channel = bev_image[:, :, 1]
    lightness_channel = hsl_image[:, :, 1]

    blur = cv2.GaussianBlur(lightness_channel, (5, 5), 0)

    edge = cv2.Canny(blur, 50, 150)

    lines = cv2.HoughLinesP(
        edge,
        rho=1,
        theta=np.pi/180,
        threshold=50,
        minLineLength=50,
        maxLineGap=10
    )

    line_image = np.zeros_like(bev_image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    result = cv2.addWeighted(bev_image, 0.8, line_image, 1, 0)

    cv2.imshow("lines", result)

    cv2.waitKey()
    cv2.destroyAllWindows()

    
