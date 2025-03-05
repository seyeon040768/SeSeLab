import cv2

# 입력 및 출력 영상 파일 경로
input_video_path = 'dataset/school.mp4'
output_video_path = 'dataset/school_25.mp4'

# 가로와 세로 배율 설정 (원하는 값으로 조정)
fx = 0.2
fy = 0.2

# 입력 영상 열기
cap = cv2.VideoCapture(input_video_path)
if not cap.isOpened():
    print("영상을 열 수 없습니다.")
    exit()

# 원본 영상의 프레임 정보 읽기
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# 변환 후 영상의 크기 계산
new_width = int(frame_width * fx)
new_height = int(frame_height * fy)

# VideoWriter 객체 생성 (코덱: mp4v)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (new_width, new_height))

while True:
    ret, frame = cap.read()
    if not ret:
        break  # 영상의 끝에 도달하면 종료
    
    # 프레임 크기 조정 (cv2.resize)
    resized_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    
    # 변환된 프레임을 출력 영상에 기록
    out.write(resized_frame)

# 자원 해제
cap.release()
out.release()
print("영상이 저장되었습니다:", output_video_path)
