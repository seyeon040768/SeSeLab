import numpy as np
import matplotlib.pyplot as plt

# 예제: 직사각형 영역 내 무작위 점 생성
np.random.seed(0)
num_points = 100
# x 좌표: 0 ~ 10, y 좌표: 0 ~ 5 사이의 값
x = np.array([0, 0, 0, 1, 2, 2, 2, 3, 3, 3, 4, 4])
y = np.array([0, 0, 1, 2, 3, 3, 4, 5, 5, 5, 6, 6])
points = np.column_stack((x, y))

# 데이터 중심화 (평균 제거)
mean = np.mean(points, axis=0)
centered_points = points - mean

# 공분산 행렬 계산
cov_matrix = np.cov(centered_points.T)

# 고유값 및 고유벡터 계산
eigvals, eigvecs = np.linalg.eig(cov_matrix)

# 고유값 기준으로 내림차순 정렬
order = np.argsort(eigvals)[::-1]
eigvals = eigvals[order]
eigvecs = eigvecs[:, order]

# 결과 출력
print("평균:", mean)
print("공분산 행렬:\n", cov_matrix)
print("고유값:", eigvals)
print("고유벡터:\n", eigvecs)

# 시각화: 점들과 주성분 방향 표시
plt.figure(figsize=(8, 6))
plt.scatter(points[:, 0], points[:, 1], alpha=0.5, label='Data Points')
plt.xlabel("x")
plt.ylabel("y")
plt.title("PCA: 데이터의 주요 방향")

# 데이터의 중심을 표시
plt.scatter(mean[0], mean[1], color='black', label='Mean')

# 첫 번째 주성분 (최대 분산 방향)
pc1 = eigvecs[:, 0]  # 첫번째 주성분
# 길이 조정 (고유값의 제곱근 활용)
scale = np.sqrt(eigvals[0])
plt.quiver(mean[0], mean[1], pc1[0]*scale, pc1[1]*scale, 
           color='red', scale_units='xy', scale=1, label='Principal Component 1')

plt.legend()
plt.grid(True)
plt.show()
