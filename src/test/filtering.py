import numpy as np
import cv2

def filltering(img_color):
    img_hls = cv2.cvtColor(img_color, cv2.COLOR_BGR2HLS)
    # BGR to HLS 색공간 변환

    lower = np.array([20, 170, 20])
    upper = np.array([255, 255, 255])

    yellow_low = np.array([0, 85, 81])
    yellow_up = np.array([190, 255, 255])

    yellow_mask = cv2.inRange(img_hls, yellow_low, yellow_up)
    white_mask = cv2.inRange(img_hls, lower, upper)
    mask = cv2.bitwise_or(yellow_mask, white_mask)
    img_res = cv2.bitwise_and(img_color, img_color, mask = mask)

    return img_res

img = cv2.imread("1.jpg",cv2.IMREAD_COLOR)

img = filltering(img)

cv2.imshow('img', img)
cv2.imwrite("aaaaaaaaaaaaaa.png", img)

cv2.waitKey() # 키보드 입력이 들어올 때까지 창을 유지
cv2.destroyAllWindows() # 모든 윈도우 창을 제거