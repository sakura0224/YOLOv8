import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
from tqdm import tqdm


class VideoSharpening:
    def __init__(self, input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path

    @staticmethod
    def _sharpen(src):
        blur = cv2.GaussianBlur(src, (5, 5), 1)
        return cv2.addWeighted(src, 2, blur, -1, 0)

    def _process_frame_and_save(self, i):
        cap = cv2.VideoCapture(self.input_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        cap.release()
        if ret:
            sharp_frame = self._sharpen(frame)
            return i, sharp_frame
        return i, None

    def process_video(self):
        cap = cv2.VideoCapture(self.input_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        if not os.path.exists('frames'):
            os.makedirs('frames')

        # Process frames in parallel and store results
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self._process_frame_and_save, i)
                       for i in range(frame_count)]
            results = [future.result() for future in
                       tqdm(as_completed(futures), total=frame_count, desc="Processing Frames")]

        # Filter out None results and sort by frame index
        processed_frames = [r for r in results if r[1] is not None]
        processed_frames.sort(key=lambda x: x[0])

        if processed_frames:
            # Check if the frame is not None before accessing its shape
            if processed_frames[0][1] is not None:
                height, width, _ = processed_frames[0][1].shape
                fourcc = cv2.VideoWriter.fourcc(*'avc1')
                out = cv2.VideoWriter(
                    self.output_path, fourcc, 25.0, (width, height))

                # Write frames in correct order
                for _, frame in tqdm(processed_frames, total=len(processed_frames), desc="Processing Video"):
                    # Ensure frame is not None before writing
                    if frame is not None:
                        out.write(frame)
                    else:
                        print(f"Skipped a None frame")

                out.release()

        folder_path = 'frames'
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)

        return self.output_path
