import numpy as np
import cv2
# numpy ver 1.22.1      행렬같은 다차원 배열을 쉽게 처리할 수 있도록 하는 라이브러리
# opencv ver 4.5.5      이미지 관련 라이브러리


# 1. Filtering
# 2. Warping을 통한 Bird Eye View 변환
# 3. binary 이미지로 변환
# 4. Histogram으로 Window의 시작점을 정함
# 5. Window Sliding Algorithm을 통한 차선 추출
# 6. 위 단계에서 얻은 차선 추출 정보로 차선 표시


# 1. Filtering
def filltering(img_color):
    img_hls = cv2.cvtColor(img_color, cv2.COLOR_BGR2HLS)
    # BGR to HLS 색공간 변환

    w_low = np.array([20, 160, 20])
    w_up = np.array([255, 255, 255])
    # 흰색 차선의 임계값

    y_low = np.array([0, 85, 81])
    y_up = np.array([190, 255, 255])
    # 노란색 차선의 임계값

    w_mask = cv2.inRange(img_hls, w_low, w_up)
    # 흰색 차선 마스크 적용
    y_mask = cv2.inRange(img_hls, y_low, y_up)
    # 노란색 차선 마스크 적용

    mask = cv2.bitwise_or(y_mask, w_mask)
    # mask 영역의 두 이미지를 합침
    img_res = cv2.bitwise_and(img_color, img_color, mask=mask)
    # mask 영역에서 공통으로 겹치는 부분 저장

    return img_res

# 2. Warping을 통한 Bird Eye View 변환
def warpPer(img_src):
    h, w = img_color.shape[:2]

    # 좌상, 좌하, 우상, 우하
    pts1 = np.array([[340-50, 235], [0, h], [340+50, 235], [w, h]], dtype=np.float32)
    # 원본 이미지 좌표
    pts2 = np.array([[0, 0], [0, h], [w, 0], [w, h]], dtype=np.float32)
    # Bird Eye View로 변환할 좌표

    M = cv2.getPerspectiveTransform(pts1, pts2)
    # 변환 행렬을 구함 (원근법)
    img_res = cv2.warpPerspective(img_src, M, (w, h))
    # 변환행렬값 적용

    return img_res


# 3. binary 이미지로 변환
def to_binary(img_src):
    img_bgr = cv2.cvtColor(img_src, cv2.COLOR_HLS2BGR)
    # 입력 이미지를 HLS에서 BGR로 변환
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    # BGR을 GRAY로 변환
    img_blur = cv2.GaussianBlur(img_gray, (0,0), 1)
    # 이미지의 이진화를 위한 가우시안블러 처리 (sigma = 2)

    ret, img_binary = cv2.threshold(img_blur, 180, 255, cv2.THRESH_BINARY)
    # 이미지 이진화     두 번째 임계값 = 0~255

    return img_binary


# 4. Histogram
def image_histogram(img_src):
    hist = np.sum(img_src[img_src.shape[0]//2:, :], axis=0)
    # x축에 대한 histogram 계산
    center = np.int(hist.shape[0]/2)
    # histogram을 양쪽으로 나눔
    left_lane = np.argmax(hist[:center])
    # 왼쪽 차선 (hist의 ~ center)
    right_lane = np.argmax(hist[center:]) + center
    # 오른쪽 차선 (hist의 center ~)
    
    return left_lane, right_lane


cap = cv2.VideoCapture("video.mp4")
# VideoCapture 비디오를 캡처하기 위한 클래스
while cap.isOpened():
    ret, img_color = cap.read()
    if not ret:
        print("이미지 로드 실패 또는 종료")
        break
    
    img_filter = filltering(img_color)
    img_Warping = warpPer(img_filter)
    img_binary = to_binary(img_Warping)
    print(image_histogram(img_binary))


    cv2.imshow("img_color", img_color)
    cv2.imshow("img_filter", img_filter)
    cv2.imshow("img_Warping", img_Warping)
    cv2.imshow("img_binary", img_binary)

    key = cv2.waitKey(25)
    if(key == 27):
        break
cap.release()
cv2.destroyAllWindows()