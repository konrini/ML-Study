import cv2
import numpy as np


filepath = './out/countdown.mp4'
fourcc = cv2.VideoWriter_fourcc(*'XVID') 
fps = 1.0
frame_size = (512, 512)

out = cv2.VideoWriter(filepath, fourcc, fps, frame_size)

for i in range(5, 0, -1):
    
    # 빈 화면 설정
    img = np.zeros(shape=(512, 512, 3), dtype=np.uint8)+50*i
    
    # 텍스트 내용
    text = str(i)
    
    # 폰트 꾸미기
    fontFace = [cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_COMPLEX,
                cv2.FONT_HERSHEY_DUPLEX, cv2.FONT_HERSHEY_TRIPLEX,
                cv2.FONT_ITALIC]
    
    # 폰트 크기
    fontScale = 15
    
    # 위치 재설정
    retval, baseline = cv2.getTextSize(text, fontFace[i-1], fontScale, thickness=5)    
    Cx = 255 - retval[0]//2
    Cy = 255 + retval[1]//2
    org = Cx, Cy
    
    # 도형 꾸미기
    if i == 5:
        cv2.circle(img, (256, 256), 215, (155, 255, 0), -1)
    elif i == 4:
        cv2.drawMarker(img, (256, 256), (0, 0, 255), cv2.MARKER_DIAMOND, 450, 3)
    elif i == 3:
        cv2.rectangle(img, (50,50), (450,450), (0,0,255), 2)
    elif i == 2:
        cv2.circle(img, (256, 256), 215, (155, 255, 0))
    elif i == 1:
        cv2.drawMarker(img, (256, 275), (255, 255, 255), cv2.MARKER_TRIANGLE_DOWN, 400, 3)
    
    # 텍스트 작성
    cv2.putText(img, text, org, fontFace[i-1], fontScale, color=[5*i,15*i,255-50*i], thickness=5)
    
    # 좌우 반전
    if i % 2 == 0:
        img = ~img
    
    # 비디오 저장    
    out.write(img)
    
out.release()
cv2.destroyAllWindows()


# 비디오 영상 실행
cap = cv2.VideoCapture('./out/countdown.mp4')
while True:
    retval, frame = cap.read()
    if not retval:
        break
    cv2.imshow('frame', frame)
    key = cv2.waitKey(1000)
    if key == 27:  # Esc키
        break

cap.release()
cv2.destroyAllWindows()