import numpy as np
import cv2

from calibration import get_transformation_matrix
from bird_eye_view import get_map_matrix, make_bev_image

if __name__ == "__main__":
    cap = cv2.VideoCapture(2, cv2.CAP_DSHOW)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        exit()


    ret, image = cap.read()
    if not ret:
        print("Error: Could not read frame") 

    image_shape = (image.shape[1], image.shape[0])
    # cv2.imwrite("dgu_0.485_737fov.png", image)
    print(image_shape)

    axis_rotation_degree = (90, -90, 0)
    translation = (0, 0, 0)
    rotation_degree = (0, 0, 0)
    fov_degree = 73.7
    camera_height = 0.67

    axis_rotation_radian = np.deg2rad(axis_rotation_degree)
    rotation_radian = np.deg2rad(rotation_degree)
    fov_radian = np.deg2rad(fov_degree)
    m_transformation = get_transformation_matrix(image_shape, fov_radian, axis_rotation_radian, translation, rotation_radian).T

    example = np.array([[0.1, 0.05, -camera_height, 1],
                        [0.1, -0.0, -camera_height, 1],
                        [0.3, 0, -camera_height, 1]])
    
    projected_points = example @ m_transformation
    projected_points = projected_points[:, :2] / projected_points[:, 2:3]
    projected_points = projected_points.astype(np.int32)

    # Draw points on image for visualization
    # vis_image = image.copy()
    # for point in projected_points:
    #     cv2.circle(vis_image, tuple(point), 5, (0, 255, 0), -1)
        
    # cv2.imshow("Projected Points", vis_image)
    # cv2.waitKey(0)  # Changed from 1 to 0 to wait for key press
    # print("Projected points:", projected_points)

    # exit()

    x_range = (1.5, 12)
    y_range = (-3, 3)
    bev_pixel_interval = (0.02, 0.01)

    map_matrix = get_map_matrix(m_transformation, x_range, y_range, bev_pixel_interval, camera_height)
    
    projected_points = map_matrix.reshape(-1, 2)

    valid_mask = (0 <= projected_points[:, 0]) & (projected_points[:, 0] <= image_shape[0] - 1) & \
                 (0 <= projected_points[:, 1]) & (projected_points[:, 1] <= image_shape[1] - 1)
    valid_points = projected_points[valid_mask]

    # Define the codec and create VideoWriter object
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # out_original = cv2.VideoWriter('original.mp4', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    while True:
        ret, image = cap.read()
        if not ret:
            print("Error: Could not read frame") 
            break

        bev_image, valid_area = make_bev_image(image, map_matrix, valid_points, valid_mask)

        # Write the frames
        # out_original.write(image)

        cv2.imshow("Original", image)
        cv2.imshow("BEV", bev_image)
            
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release everything
    cap.release()
    # out_original.release()
    cv2.destroyAllWindows()
    exit()
    

    

    
    
    