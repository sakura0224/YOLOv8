import cv2
from ultralytics import YOLO
from ultralytics.solutions import object_counter


class YOLOTracker:
    def __init__(self, input_path, output_path, model_path='models/yolov8x.pt'):
        self.input_path = input_path
        self.output_path = output_path
        self.model_path = model_path
        self.model = YOLO(self.model_path)

    def track_and_count(self):
        cap = cv2.VideoCapture(self.input_path)
        assert cap.isOpened(), "Error reading· video file"
        w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

        line_points = [(1280, 1080), (1280, 0)]
        classes_to_count = [0]

        video_writer = cv2.VideoWriter(self.output_path, cv2.VideoWriter.fourcc(*'avc1'), fps, (w, h))
        counter = object_counter.ObjectCounter()
        counter.set_args(view_img=True, view_in_counts=False, reg_pts=line_points, classes_names=self.model.names, draw_tracks=True)

        while cap.isOpened():
            success, im0 = cap.read()
            if not success:
                break

            tracks = self.model.track(im0, persist=True, show=False, classes=classes_to_count)
            im0 = counter.start_counting(im0, tracks)
            video_writer.write(im0)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        video_writer.release()
        cv2.destroyAllWindows()

        return self.output_path
