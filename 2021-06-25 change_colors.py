import cv2
import numpy as np
import matplotlib.pyplot as plt


heart = cv2.imread('C:/Users/user/vision/data/Heart10.jpg')

# 그레이스케일로 색상공간 변환
heart_gray = cv2.cvtColor(heart, cv2.COLOR_RGB2GRAY)

# 임계값으로 적합한 값 고르기
# hist = cv2.calcHist([heart_gray], [0], None, [256], [0, 256])
# print(plt.bar(range(256), hist.flatten()))

# 128을 임계값으로 이진화
_, frame = cv2.threshold(heart_gray, 128, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

# 잡음을 줄이기 위한 닫기 연산 (팽창 후 침식)
frame = cv2.morphologyEx(frame, cv2.MORPH_CLOSE, None, iterations=1)

# 경계값 찾기
contours, hierarchy = cv2.findContours(frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
colorful_heart = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

idx = 0
while idx >= 0:
    c = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
    cv2.drawContours(colorful_heart, contours, idx, c, -1, cv2.LINE_8, hierarchy)
    idx = hierarchy[0, idx, 0]

cv2.imshow('heart', heart)
cv2.imshow('frame', frame)
cv2.imshow('colorful_heart', colorful_heart)

cv2.waitKey()
cv2.destroyAllWindows()