import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from read_data import read_oxts_from_directory, read_images_from_directory

def latlon_to_xy(lat, lon, lat0, lon0):
    """
    위도, 경도 정보를 기준 좌표(lat0, lon0)로부터의 상대적인 x, y 좌표(미터 단위)로 변환합니다.
    
    공식:
      - x = R * (Δlon in rad) * cos(lat0 in rad)
      - y = R * (Δlat in rad)
      
    여기서 R은 지구의 반지름(약 6,378,137 m)입니다.
    """
    R = 6378137.0  # 지구의 평균 반지름 (미터)

    # 위도, 경도를 라디안 단위로 변환
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    lat0_rad = np.radians(lat0)
    lon0_rad = np.radians(lon0)
    
    # 위도와 경도의 차이 (라디안)
    dlat = lat_rad - lat0_rad
    dlon = lon_rad - lon0_rad
    
    # 동서(x) 방향 거리: 경도 차이에 기준 위도의 cos 값을 곱해 보정
    x = R * dlon * np.cos(lat0_rad)
    # 남북(y) 방향 거리: 위도 차이에 지구 반지름을 곱함
    y = R * dlat
    
    return np.array([x, y])

def rotate_2d(vector, angle):
    """
    2D 벡터를 주어진 각도만큼 회전시킵니다.
    
    Args:
        vector: 회전할 2D 벡터 (numpy array)
        angle: 회전 각도 (라디안)
    
    Returns:
        회전된 2D 벡터 (numpy array)
    """
    rotation_matrix = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)]
    ])
    return np.dot(rotation_matrix, vector)

if __name__ == "__main__":
    data = read_oxts_from_directory(r"dataset\oxts\data", r"dataset\oxts\timestamps.txt")
    images = read_images_from_directory(r"dataset\image_00\data")

    # 처음 위치는 (0, 0)으로 설정
    x_coords = [0.0]
    y_coords = [0.0]

    x_pred_coords = [0.0]
    y_pred_coords = [0.0]

    init_lat, init_lon, init_alt, init_roll, init_pitch, init_yaw, init_v, init_ax, init_ay, init_az, init_t = data[0]

    dir = np.array([np.cos(init_yaw), np.sin(init_yaw)])
    prev_t = init_t
    prev_v = init_v * dir # 속력을 이용해 속도 벡터 생성
    prev_a = rotate_2d(np.array([init_ax, init_ay]), init_yaw) # 로컬 가속도를 일관된 x, y 좌표계로 회전
    prev_loc = np.array([0.0, 0.0])

    for lat, lon, alt, roll, pitch, yaw, velo, ax, ay, az, t in data[1:]:
        loc = latlon_to_xy(lat, lon, init_lat, init_lon)

        deltatime = t - prev_t

        # 가속도 -> 속도 -> 변위
        dir = np.array([np.cos(yaw), np.sin(yaw)])
        a = rotate_2d(np.array([ax, ay]), yaw)
        avg_a = (prev_a + a) / 2
        v = prev_v + avg_a * deltatime
        avg_v = (prev_v + v) / 2
        s = avg_v * deltatime

        loc_pred = prev_loc + s

        x_coords.append(loc[0])
        y_coords.append(loc[1])

        x_pred_coords.append(loc_pred[0])
        y_pred_coords.append(loc_pred[1])

        prev_t = t
        prev_a = a
        prev_v = v
        prev_loc = loc_pred

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(19.2, 10.8))
    
    def animate(frame):
        ax1.clear()
        ax2.clear()
        
        ax1.imshow(images[frame])
        ax1.set_title('Camera View', fontsize=30)
        ax1.axis('off')
        
        ax2.plot([-x for x in x_coords[:frame+1]], [-y for y in y_coords[:frame+1]], 'b-', label='Ground Truth')
        ax2.plot([-x for x in x_pred_coords[:frame+1]], [-y for y in y_pred_coords[:frame+1]], 'r--', label='Predicted')
        ax2.set_xlabel('X', fontsize=28)
        ax2.set_ylabel('Y', fontsize=28)
        ax2.set_title('Vehicle Trajectory with IMU', fontsize=30)
        ax2.legend(fontsize=26)
        # ax2.axis('equal')
        ax2.tick_params(axis='both', which='major', labelsize=26)
    
    anim = animation.FuncAnimation(fig, animate, frames=len(images), 
                                 interval=100,
                                 repeat=False)
    # plt.tight_layout()
    
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=10, metadata=dict(artist='Seyeon Lee'), bitrate=1800)
    anim.save('vehicle_trajectory_with_imu.mp4', writer=writer)
    
    plt.show()