import cv2
import numpy as np

# 打开视频
cap = cv2.VideoCapture('video/C0003.mp4')

# 初始化帧
ret, frame1 = cap.read()
ret, frame2 = cap.read()

while True:
    # 读取新的一帧
    ret, frame3 = cap.read()
    if not ret:
        break

    # 转换为灰度图
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    gray3 = cv2.cvtColor(frame3, cv2.COLOR_BGR2GRAY)

    # 计算两个帧差
    diff1 = cv2.absdiff(gray1, gray2)
    diff2 = cv2.absdiff(gray2, gray3)

    # 二值化处理
    _, thresh1 = cv2.threshold(diff1, 20, 255, cv2.THRESH_BINARY)
    _, thresh2 = cv2.threshold(diff2, 20, 255, cv2.THRESH_BINARY)

    # 取交集
    motion = cv2.bitwise_and(thresh1, thresh2)

    # 形态学处理，去除噪声
    kernel = np.ones((5, 5), np.uint8)
    motion = cv2.morphologyEx(motion, cv2.MORPH_CLOSE, kernel)
    
    # 显示结果
    cv2.imshow('Motion Detection', motion)

    # 更新帧
    frame1 = frame2
    frame2 = frame3

    # 按下 'q' 键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
