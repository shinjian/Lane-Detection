# 3. 차선 ROI(관심영역) 지정
def region_of_interest(img_src):
    h, w = 270, 340     # w = 250 <-> 430
    # 양쪽 차선의 중점(w)
    vertices = np.array([[(w-320, 360), (w-40, h-60), (w+40, h-60), (w+320, 360)]], dtype=np.int32)
    # 꼭짓점 = 넘파이 행렬      (x1, y1), (x2, y2)      (x3, y3), (x4, y4)

    mask = np.zeros_like(img_src)
    color = 255

    cv2.fillPoly(mask, vertices, color)
    img_roi = cv2.bitwise_and(img_src, mask)

    return img_roi