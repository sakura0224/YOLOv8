import os
import shutil

file_path = 'video_with_noise.mp4'  # 替换为要删除的文件路径
if os.path.exists(file_path):  # 确保文件存在
    os.remove(file_path)
    print(f"文件 '{file_path}' 已删除")
else:
    print(f"文件 '{file_path}' 不存在")

file_path = 'output_video.mp4'  # 替换为要删除的文件路径

if os.path.exists(file_path):  # 确保文件存在
    os.remove(file_path)
    print(f"文件 '{file_path}' 已删除")
else:
    print(f"文件 '{file_path}' 不存在")

folder_path = 'frames'  # 替换为要删除的文件夹路径
if os.path.exists(folder_path):  # 确保文件夹存在
    shutil.rmtree(folder_path)
    print(f"文件夹 '{folder_path}' 及其内容已删除")
else:
    print(f"文件夹 '{folder_path}' 不存在")

folder_path = 'runs'  # 替换为要删除的文件夹路径
if os.path.exists(folder_path):  # 确保文件夹存在
    shutil.rmtree(folder_path)
    print(f"文件夹 '{folder_path}' 及其内容已删除")
else:
    print(f"文件夹 '{folder_path}' 不存在")
