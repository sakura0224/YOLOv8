import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
from tqdm import tqdm


def someblur(src, blursize=5):
    dst = cv2.GaussianBlur(src, (blursize, blursize), 1)
    return dst


def sharpen(src):
    blur = someblur(src, 5)
    dst = cv2.addWeighted(src, 2, blur, -1, 0)
    return dst


def process_frame_and_save(i, path):
    pac = cv2.VideoCapture(path)
    pac.set(cv2.CAP_PROP_POS_FRAMES, i)
    ret, img = pac.read()
    pac.release()
    if ret:
        imgsharp = sharpen(img)
        cv2.imwrite(f'frames/frame_{i}.png', imgsharp)
        return i, imgsharp  # 返回一个包含帧索引和处理后帧的元组
    return None


if __name__ == '__main__':
    video_path = 'video/test.mp4'
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if not os.path.exists('frames'):
        os.makedirs('frames')

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_frame_and_save, i, video_path) for i in range(frame_count)]
        results = []
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing Frames"):
            result = future.result()
            if result is not None:
                results.append(result)

    # 根据帧索引对结果排序
    results.sort(key=lambda x: x[0])
    frames = [frame for i, frame in results]  # 提取排序后的帧

    output_video = 'after_process.mp4'
    if frames:
        height, width, _ = frames[0].shape
        fourcc = cv2.VideoWriter.fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video, fourcc, 25.0, (width, height))

        for frame in frames:
            out.write(frame)

        out.release()
