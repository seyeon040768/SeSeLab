import cv2
import numpy as np
import matplotlib.pyplot as plt

# 컬러 이미지 불러오기
left = cv2.imread('left_stair.jpg', cv2.IMREAD_COLOR)
right = cv2.imread('right_stair.jpg', cv2.IMREAD_COLOR)

# BGR 채널 분리
stereo = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=64,
    blockSize=9,
    P1=8 * 3 * 9 ** 2,
    P2=32 * 3 * 9 ** 2,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
)

# 시차 맵 계산
disparity = stereo.compute(left, right)

# 시각화
cv2.imshow("Disparity", disparity)
cv2.waitKey(0)
cv2.destroyAllWindows()