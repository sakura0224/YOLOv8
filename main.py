import os

import gradio as gr

from classes.VideoSharpening import VideoSharpening  # 导入封装好的视频锐化类
from classes.YOLOTracker import YOLOTracker  # 导入封装好的YOLO追踪计数类


def process_video(input_video, process_type):
    temp_video_path = input_video

    # 中间视频路径，用于当执行两个步骤时
    intermediate_video_path = "run/intermediate_video.mp4"
    output_video_path = "run/processed_video.mp4"

    # 根据所选的处理类型执行操作
    if process_type == "Sharpen":
        sharpener = VideoSharpening(temp_video_path, output_video_path)
        output_video_path = sharpener.process_video()
    elif process_type == "Track and Count":
        tracker = YOLOTracker(temp_video_path, output_video_path)
        output_video_path = tracker.track_and_count()
    elif process_type == "Run All":
        # 首先执行 VideoSharpening
        sharpener = VideoSharpening(temp_video_path, intermediate_video_path)
        intermediate_video_path = sharpener.process_video()

        # 然后用中间结果作为 YOLOTracker 的输入
        tracker = YOLOTracker(intermediate_video_path, output_video_path)
        output_video_path = tracker.track_and_count()

    # 清理临时文件
    if os.path.exists(temp_video_path):
        os.remove(temp_video_path)
    if process_type == "Run All" and os.path.exists(intermediate_video_path):
        os.remove(intermediate_video_path)

    return output_video_path


# 创建 Gradio 界面
with gr.Blocks() as demo:
    gr.Markdown("### Video Processing with Sharpening and YOLO Tracking")

    with gr.Row():
        file_input = gr.File(label="Upload A Video")
        process_type = gr.Radio(["Sharpen", "Track and Count", "Run All"], label="Select Process Type")
        submit_button = gr.Button("Process Video")

    with gr.Row():
        with gr.Column(scale=1):
            video_input = gr.Video(label="Original Video")
        with gr.Column(scale=1):
            video_output = gr.Video(label="Processed Video")

    file_input.change(fn=lambda x: x, inputs=file_input, outputs=video_input)
    submit_button.click(fn=process_video, inputs=[file_input, process_type], outputs=video_output)

demo.launch()
