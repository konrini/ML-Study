import numpy as np
import cv2


src = cv2.imread('opencv_workshop/ch05/lenna.bmp')
# 관심 영역 표시
rects = cv2.selectROIs('select multiple rois', src)

# 첫 번째 영역은 임계값 기준 이진화 작업
x, y, w, h = rects[0]
src_g = cv2.cvtColor(src[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
ret, thresh1 = cv2.threshold(src_g, 150, 255, cv2.THRESH_BINARY)
src[y:y+h, x:x+w] = cv2.cvtColor(thresh1, cv2.COLOR_GRAY2BGR)

# 두 번째 영역은 반전
x, y, w, h = rects[1]
src[y:y+h, x:x+w] = ~src[y:y+h, x:x+w]

# 결과값 출력
cv2.imshow('result', src)
cv2.waitKey()
cv2.destroyAllWindows()
