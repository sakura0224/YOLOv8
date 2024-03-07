import numpy as np
import simpleaudio as sa


def error_alarm():
    # 设置警笛声音的参数
    duration = 2.0  # 警笛声音持续的时间
    fs = 44100  # 采样率

    # 生成时间数组
    t = np.linspace(0, duration, int(fs * duration), False)

    # 设置两个频率
    freq1 = 1200  # 警笛声音的第一个频率
    freq2 = 700  # 警笛声音的第二个频率

    # 设置每个频率的持续时间
    freq_duration = 0.425  # 每个频率的持续时间，0.425秒

    # 生成交替频率的正弦波
    note1 = np.sin(2 * np.pi * freq1 * t[:int(fs * freq_duration)])
    note2 = np.sin(2 * np.pi * freq2 * t[:int(fs * freq_duration)])

    # 交替合成两个频率
    siren = np.array([])
    for i in range(int(duration / freq_duration)):
        if i % 2 == 0:
            siren = np.concatenate((siren, note1))
        else:
            siren = np.concatenate((siren, note2))

    # 归一化到16位范围
    audio = siren * (2 ** 15 - 1) / np.max(np.abs(siren))
    audio = audio.astype(np.int16)

    # 播放警笛声音
    play_obj = sa.play_buffer(audio, 1, 2, fs)
    play_obj.wait_done()


# 调用函数以播放声音
error_alarm()
