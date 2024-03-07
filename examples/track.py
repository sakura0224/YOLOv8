from collections import defaultdict

import cv2
import numpy as np
from ultralytics import YOLO

# 加载 YOLOv8 模型
model = YOLO('yolov8x.pt')

# 打开视频文件
video_path = "after_process.mp4"
cap = cv2.VideoCapture(video_path)

# 获取视频的帧率和尺寸
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_rate = int(cap.get(cv2.CAP_PROP_FPS))

# 定义视频编码和创建 VideoWriter 对象
output_path = "track.mp4"
fourcc = cv2.VideoWriter.fourcc(*'mp4v')  # 或者 'XVID'，根据需要选择编码
out = cv2.VideoWriter(output_path, fourcc, frame_rate, (frame_width, frame_height))

# 存储跟踪历史
track_history = defaultdict(lambda: [])
pedestrian_ids = set()

# 遍历视频帧
while cap.isOpened():
    # 从视频中读取一帧
    success, frame = cap.read()

    if success:
        annotated_frame = frame.copy()  # 复制原始帧用于注释

        # 在帧上运行 YOLOv8 跟踪，保持帧间跟踪
        results = model.track(frame, persist=True, line_width=1, classes=[0, 1, 25])

        # 如果没有检测到对象，跳过当前循环
        if results[0].boxes is None or results[0].boxes.id is None or results[0].boxes.cls is None:
            out.write(frame)  # 直接写入原始帧
            continue

        # 获取边界框和跟踪 ID
        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        class_labels = results[0].boxes.cls.int().cpu().tolist()  # 获取类别标签

        for track_id, class_label in zip(track_ids, class_labels):
            if class_label == 0:  # 假设类别 0 是行人
                pedestrian_ids.add(track_id)

        # 在帧上可视化结果
        annotated_frame = results[0].plot()

        # 绘制轨迹
        for box, track_id in zip(boxes, track_ids):
            x, y, w, h = box
            track = track_history[track_id]
            track.append((float(x), float(y)))  # x, y 中心点
            if len(track) > 20:  # 保留最近 20 帧的轨迹
                track.pop(0)

            # 绘制跟踪线
            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(annotated_frame, [points], isClosed=False, color=(0, 255, 255), thickness=2)

        pedestrian_count = len(pedestrian_ids)
        cv2.putText(annotated_frame, f"Number of Pedestrians: {pedestrian_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # 将处理过的帧写入输出视频
        out.write(annotated_frame)

        # 显示带注释的帧
        cv2.imshow("YOLOv8 Tracking", annotated_frame)

        # 如果按下 'q'，则退出循环
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # 如果到达视频末尾，则退出循环
        break

# 释放视频捕获对象并关闭显示窗口
cap.release()
out.release()  # 关闭视频写入对象
cv2.destroyAllWindows()
print("Number of Pedestrians:", len(pedestrian_ids))
