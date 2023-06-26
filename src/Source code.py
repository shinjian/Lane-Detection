import numpy as np
import cv2
# numpy ver 1.22.1      행렬같은 다차원 배열을 쉽게 처리할 수 있도록 하는 라이브러리
# opencv ver 4.5.5      이미지 관련 라이브러리

# 1. Warpping을 통한 Bird Eye View 변환
# 2. ROI 관심 영역 지정
# 3. Filtering 적용
# 4. binary 이미지로 변환
# 5. Histogram으로 Window의 시작점을 정함
# 6. Sliding Window Algorithm을 통한 차선 추출
# 7. 위 단계에서 얻은 차선 추출 정보로 차선 표시



# 1. Warpping을 통한 Bird Eye View 변환
def Warpping(img_src):
    # 왼쪽 차선 함수 y = (88/107)x

    # 좌상, 좌하, 우상, 우하
    
    pts1 = np.array([[305, 285], [210, 360], [347, 285], [489, 360]], dtype=np.float32)
    # 원본 이미지 좌표
    pts2 = np.array([[80, 0], [80, 360], [560, 0], [560, 360]], dtype=np.float32)
    # Bird Eye View로 변환할 좌표

    M = cv2.getPerspectiveTransform(pts1, pts2)
    # 변환 행렬을 구함 (원근법)
    img_res = cv2.warpPerspective(img_src, M, (640, 360))
    # 변환행렬값 적용

    return img_res, pts1, pts2


# 2. ROI 관심 영역 지정
def region_of_interest(img_src):
    # 왼쪽 차선에 대한 함수 y = (88/107)x

    # 왼쪽 아래 - 왼쪽 위 - 오른쪽 위 - 오른쪽 아래
    pts1 = np.array([[0, 360], [0, 0], [230, 0], [230, 360],
                    [410, 360], [410, 0],[640, 0], [640, 360]])
                    # 왼쪽 차선, 오른쪽 차선
    
    mask = np.zeros_like(img_src)
    # zeros_like = img_src 행렬값을 0으로 설정
    color = (255,) * img_src.shape[2]
    # img_src.shape[2] = 이미지 채널

    cv2.fillPoly(mask, np.int32([pts1]), color)
    # 위에서 정한 pts1의 좌표를 이용하여 마스크 이미지 생성
    img_roi = cv2.bitwise_and(img_src, mask)
    # bitwise_and = 이미지 합치는 함수

    return img_roi


# 3. Filtering
def filtering(img_src):
    img_hls = cv2.cvtColor(img_src, cv2.COLOR_BGR2HLS)
    # BGR to HLS 색공간 변환
    # HLS = Hue 색조, Lightness 명도, Saturation 채도

    # 흰색 차선의 임계값
    # w_low = np.array([20, 0, 43])
    w_low = np.array([20, 150, 20])
    w_up = np.array([255, 255, 255])

    # 노란색 차선의 임계값
    y_low = np.array([0, 85, 81])
    y_up = np.array([190, 255, 255])

    w_mask = cv2.inRange(img_hls, w_low, w_up)
    # 흰색 차선 마스크 적용
    y_mask = cv2.inRange(img_hls, y_low, y_up)
    # 노란색 차선 마스크 적용

    mask = cv2.bitwise_or(y_mask, w_mask)
    # mask 영역의 두 이미지를 합침
    img_res = cv2.bitwise_and(img_src, img_src, mask=mask)
    # mask 영역에서 공통으로 겹치는 부분 저장

    return img_res


# 4. binary 이미지로 변환
def to_binary(img_src):
    img_bgr = cv2.cvtColor(img_src, cv2.COLOR_HLS2BGR)
    # 입력 이미지를 HLS에서 BGR로 변환
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    # BGR을 GRAY로 변환
    img_blur = cv2.GaussianBlur(img_gray, (0,0), 1)
    # 이미지의 이진화, 노이즈 제거를 위한 가우시안블러 처리 (sigma = 2)

    ret, img_binary = cv2.threshold(img_blur, 180, 255, cv2.THRESH_BINARY)
    # 이미지 이진화     두 번째 임계값 = 0~255   80
    # 픽셀값이 180 이상이면 255(흰색)으로 처리

    return img_binary


# 5. Histogram으로 Window의 시작점을 정함
def image_histogram(img_src):
    hist = np.sum(img_src[img_src.shape[0]//2:, :], axis=0)
    # x축에 대한 histogram 계산

    center = int(hist.shape[0]/2)
    # histogram을 양쪽으로 나눔

    left_lane = np.argmax(hist[:center])
    # 왼쪽 차선 (hist의 ~ center)       argmax = 최댓값의 index 반환

    right_lane = np.argmax(hist[center:]) + center
    # 오른쪽 차선 (hist의 center ~)     argmax = 최댓값의 index 반환
    
    return left_lane, right_lane


# 6. Sliding Window Algorithm을 통한 차선 추출
def sliding_window(img_src, left_current, right_current):
    nonzero = np.nonzero(img_src)
    # img_src의 0이 아닌 값들의 index 반환
    nonzero_x = np.array(nonzero[1])
    # nonzero의 x축 중 0이 아닌 값들의 index 반환
    nonzero_y = np.array(nonzero[0])
    # nonzero의 y축 중 0이 아닌 값들의 index 반환

    left_lane, right_lane = [], []

    green_color = (0, 255, 0)
    cnt_win = 6
    # Sliding window의 개수
    margin = 60
    pixels = 40
    # Window 최소 픽셀 값
    win_height = int(360 / cnt_win)
    # Window의 y값
    min_lane_pts = 10

    img_res = np.dstack((img_src, img_src, img_src)) * 255
    # Window를 그리기 위한 색공간 변환

    # 위에서 정한 cnt_win 윈도우 개수 만큼 반복
    for w in range(cnt_win):
        win_h1 = 360 - (w+1) * win_height
        win_h2 = 360 - w * win_height
        # 윈도우의 y값
        
        l_win_w1 = left_current - margin
        l_win_w2 = left_current + margin
        # 왼쪽 윈도우의 x값

        r_win_w1 = right_current - margin
        r_win_w2 = right_current + margin
        # 오른쪽 윈도우의 x값

        img_src = cv2.rectangle(img_res, (l_win_w1, win_h1), (l_win_w2, win_h2), green_color, 1)
        # rectangle = 위에서 설정한 값들을 바탕으로 왼쪽 사각형 Window를 그리는 함수
        img_src = cv2.rectangle(img_res, (r_win_w1, win_h1), (r_win_w2, win_h2), green_color, 1)
        # rectangle = 위에서 설정한 값들을 바탕으로 오른쪽 사각형 Window를 그리는 함수

        l_win_lane = ((nonzero_y >= win_h1) & (nonzero_y < win_h2) & (nonzero_x >= l_win_w1) & (nonzero_x < l_win_w2)).nonzero()[0]
        r_win_lane = ((nonzero_y >= win_h1) & (nonzero_y < win_h2) & (nonzero_x >= r_win_w1) & (nonzero_x < r_win_w2)).nonzero()[0]
        # 위에서 설정한 nonzero_x|y, win을 바탕으로 각 window 내부의 값들 중 0이 아닌 것의 index를 반환

        left_lane.append(l_win_lane)
        right_lane.append(r_win_lane)

        # l_win_lane(Window 내부에 있는 예상 차선 픽셀 값) 배열의 길이가 위에서 정한 최소 픽셀값 보다 크다면
        if len(l_win_lane) > pixels:
            left_current = int(np.mean(nonzero_x[l_win_lane]))
            # left_lane의 값을 새롭게 재정의
        
        # r_win_lane(Window 내부에 있는 예상 차선 픽셀 값) 배열의 길이가 위에서 정한 최소 픽셀값 보다 크다면
        if len(r_win_lane) > pixels:
            right_current = int(np.mean(nonzero_x[r_win_lane]))
            # right_lane의 값을 새롭게 재정의


    left_lane = np.concatenate(left_lane)
    right_lane = np.concatenate(right_lane)
    # 배열을 1차원으로 합치는 함수

    leftx = nonzero_x[left_lane]
    lefty = nonzero_y[left_lane]
    rightx = nonzero_x[right_lane]
    righty = nonzero_y[right_lane]

    left_fit, right_fit = None, None
    ret = None

    if len(leftx) >= min_lane_pts and len(rightx) >= min_lane_pts:
        # 최소자승법을 이용하여 각 값들을 제일 잘 표현할 수 있는 2차 함수 구함
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        y = np.linspace(0, img_src.shape[0] - 1, img_src.shape[0])
        # 2차 함수의 y값 지정 (linspace)        범위 0~359
        
        # y = a*x^2  +  b*x  + c
        left_fit_x = left_fit[0] * y ** 2 + left_fit[1] * y + left_fit[2]
        right_fit_x = right_fit[0] * y ** 2 + right_fit[1] * y + right_fit[2]
        # 위에서 정한 y값마다 2차 함수를 통해 x값 구함

        ltx = np.trunc(left_fit_x)
        rtx = np.trunc(right_fit_x)
        # trunc = 소수점 부분을 버리는 함수
        
        ret = [ltx, rtx, y]

    img_res[nonzero_y[left_lane], nonzero_x[left_lane]] = [255, 0, 0]
    img_res[nonzero_y[right_lane], nonzero_x[right_lane]] = [0, 0, 255]

    return img_res, ret


def draw_lane(img_src, img_warped, pts1, pts2, ret):
    img_warped = np.zeros_like(img_warped).astype(np.uint8)

    pts_left = np.array([np.transpose(np.vstack([ret[0], ret[2]]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([ret[1], ret[2]])))])
    pts = np.hstack((pts_left, pts_right))
    # vstack = 배열을 세로로 결합       transpose = 행과 열 바꾸기(전치행렬)
    # hstack = 배열을 옆으로 결합 시키는 함수

    mean = np.mean((ret[0], ret[1]), axis=0)
    # mean = 평균값 구하는 함수
    pts_mean = np.array([np.flipud(np.transpose(np.vstack([mean, ret[2]])))])
    # vstack = 배열을 세로로 결합       transpose = 행과 열 바꾸기(전치행렬)

    color = (120, 255, 105)
    cv2.fillPoly(img_warped, np.int_([pts]), color)
    cv2.fillPoly(img_warped, np.int_([pts_mean]), color)
    # 위 정보들을 바탕으로 차선 해당 부분 채움

    M = cv2.getPerspectiveTransform(pts2, pts1)
    img_res = cv2.warpPerspective(img_warped, M, (640, 360))
    # 1번 내용과 동일

    img_res = cv2.addWeighted(img_src, 1, img_res, 0.3, 0)
    # 원본 영상(img_src)에 차선 마킹 img_res를 1:0.3 비율로 합침

    return img_res


temp_ret = []
cap = cv2.VideoCapture("video.mp4")
# VideoCapture 비디오를 캡처하기 위한 클래스
while cap.isOpened():
    ret, img_color = cap.read()
    if not ret:
        print("이미지 로드 실패 또는 종료")
        break

    img_Warpping, pts1, pts2 = Warpping(img_color)
    img_roi = region_of_interest(img_Warpping)
    img_filter = filtering(img_roi)
    img_bin = to_binary(img_filter)
    left_lane, right_lane = image_histogram(img_bin)
    img_window, ret1 = sliding_window(img_bin, left_lane, right_lane)

    if ret1:
        img_draw = draw_lane(img_color, img_window, pts1, pts2, ret1)
        temp_ret = ret1
    else:
        img_draw = draw_lane(img_color, img_window, pts1, pts2, temp_ret)


    cv2.imshow("Color", img_color)
    cv2.imshow("Sliding window", img_window)
    cv2.imshow("Lane draw", img_draw)


    key = cv2.waitKey(25)
    # 25ms 간격으로 키입력 대기
    if(key == 27):      # 키입력이 27(esc) = 종료
        break
cap.release()
cv2.destroyAllWindows()     # 모든 윈도우 창을 제거