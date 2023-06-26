import cv2
from matplotlib import pyplot as plt
import numpy as np


img_color = cv2.imread("3.jpg")

h, w = img_color.shape[:2]
margin = 25

# 왼쪽 차선 함수 y = (88/107)x
# 오른쪽 차선 함수 y = -(88/107)x + 11
left_low = [210, 360]
left_high = [305, 285]
right_low = [489, 360]
right_high = [347, 285]
# 293 294
# 361 294
# 210 360
# 488 360
cv2.line(img_color, (333, 285), (347, 285), 2)

# 좌상, 좌하, 우상, 우하

pts1 = np.array([left_high, left_low, right_high, right_low], dtype=np.float32)
# 원본 이미지 좌표
pts2 = np.array([[80, 0], [80, h], [w-80, 0], [w-80, h]], dtype=np.float32)
# Bird Eye View로 변환할 좌표

M = cv2.getPerspectiveTransform(pts1, pts2)
# 변환 행렬을 구함 (원근법)
img_res = cv2.warpPerspective(img_color, M, (w, h))
# 변환행렬값 적용





plt.subplot(121),plt.imshow(img_color),plt.title('image')
plt.subplot(122),plt.imshow(img_res),plt.title('Perspective')
plt.show()


cv2.waitKey() # 키보드 입력이 들어올 때까지 창을 유지
cv2.destroyAllWindows() # 모든 윈도우 창을 제거