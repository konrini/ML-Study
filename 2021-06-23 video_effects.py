#!/usr/bin/env python
# coding: utf-8

# In[3]:


import cv2
import numpy as np


filepath = './out/change.mp4'
fourcc = cv2.VideoWriter_fourcc(*'XVID') 
fps = 20
frame_size = (512, 512)

out = cv2.VideoWriter(filepath, fourcc, fps, frame_size)

# 회전하면서 나가기
lena = cv2.imread('data/lena.jpg')
for i in range(20):
    M = cv2.getRotationMatrix2D((256, 256), 18*i, 1-0.052*i)
    dst = cv2.warpAffine(lena, M, dsize=(512, 512))
    out.write(dst)

# 회전하면서 들어오기
apple = cv2.imread('data/apple.jpg')
for i in range(1, 20):
    M = cv2.getRotationMatrix2D((256, 256), 20*i-20, 0.052*i)
    dst = cv2.warpAffine(apple, M, dsize=(512, 512))
    out.write(dst)

# 겹치면서 나가고 들어오기
src1 = cv2.imread('data/apple.jpg')
src2 = cv2.imread('data/banana.jpg')
for i in range(20):
    dst = cv2.addWeighted(src1, 0.95-0.05*i, src2, 0.01+0.052*i, 0.0)
    out.write(dst)

# 작아지면서 나가기
banana = cv2.imread('data/banana.jpg')
for i in range(20):
    M = cv2.getRotationMatrix2D((256, 256), 0,  1-0.052*i)
    dst = cv2.warpAffine(banana, M, dsize=(512, 512))
    out.write(dst)

# 커지면서 들어오기
baboon = cv2.imread('data/baboon.jpg')
for i in range(20):
    M = cv2.getRotationMatrix2D((256, 256), 0, 0.01+0.052*i)
    dst = cv2.warpAffine(baboon, M, dsize=(512, 512))
    out.write(dst)
    
out.release()

# 비디오 재생
cap = cv2.VideoCapture('./out/change.mp4')
while True:
    retval, frame = cap.read()
    if not retval:
        break
    cv2.imshow('frame', frame)
    key = cv2.waitKey(50)
    if key == 32:  # 스페이스바
        key = cv2.waitKey()
        if key == 32:
            cv2.imshow('frame', frame)
    if key == 27:  # Esc키
        break

cap.release()
cv2.destroyAllWindows()

