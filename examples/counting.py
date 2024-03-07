import cv2
from ultralytics import YOLO
from ultralytics.solutions import object_counter

model = YOLO("yolov8x.pt")
cap = cv2.VideoCapture("after_process.mp4")
assert cap.isOpened(), "Error reading video file"
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

line_points = [(1200, 100), (1250, 100), (1250, 900), (1200, 900)]  # line or region points
classes_to_count = [0, 1, 25]  # person and car classes for count

# Video writer
video_writer = cv2.VideoWriter("object_counting_output.avi",
                               cv2.VideoWriter.fourcc(*'mp4v'),
                               fps,
                               (w, h))

# Init Object Counter
counter = object_counter.ObjectCounter()
counter.set_args(view_img=True,
                 reg_pts=line_points,
                 classes_names=model.names,
                 draw_tracks=True)

while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break
    tracks = model.track(im0, persist=True, show=False,
                         classes=classes_to_count)

    im0 = counter.start_counting(im0, tracks)
    video_writer.write(im0)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
video_writer.release()
cv2.destroyAllWindows()
