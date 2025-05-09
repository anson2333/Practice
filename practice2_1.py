from practice1 import conv1d
from practice1 import signal_system_conv1d
import numpy as np
import soundfile as sf  # 仅用于读取/保存WAV文件

# 读取音频文件（假设为单声道）
audio, sample_rate = sf.read('input/6492-68353-0019.wav')  # 返回音频数据和采样率
audio = audio.astype(np.float32)  # 统一为float32类型

# 手动实现音频归一化（避免依赖库函数）
def normalize(x):
    max_val = max(abs(np.max(x)), abs(np.min(x)))
    return x / max_val if max_val > 0 else x

audio = normalize(audio)

def reverb_kernel(duration=0.3, decay=0.5, sample_rate=44100):
    """生成指数衰减的混响核"""
    length = int(duration * sample_rate)
    t = np.linspace(0, duration, length)
    kernel = np.exp(-t * decay) * np.sin(2 * np.pi * 5 * t)  # 带震荡的衰减
    return normalize(kernel)


def bass_boost_kernel(cutoff=200, sample_rate=44100):
    """简易低通滤波器核"""
    freq_ratio = cutoff / sample_rate
    n = int(5 / freq_ratio)  # 核长度与截止频率相关
    t = np.linspace(-n//2, n//2, n)
    kernel = np.sinc(2 * freq_ratio * t)  # sinc函数实现低通
    return normalize(kernel)



def sharpen_kernel():
    """高频增强核（边缘检测变体）"""
    return normalize(np.array([-0.5, 1, -0.5]))



# 添加高斯白噪声
noise = np.random.normal(0, 0.05, len(audio))
noisy_audio = audio + noise

# 去噪核（均值滤波）
denoise_kernel = np.ones(5)/5  # 5点移动平均
clean_audio = conv1d(noisy_audio, denoise_kernel)
clean_audio_1 = signal_system_conv1d(noisy_audio, denoise_kernel)

reverb_audio = conv1d(audio, reverb_kernel())
bass_audio = conv1d(audio, bass_boost_kernel())
sharp_audio = conv1d(audio, sharpen_kernel())

reverb_audio_1 = signal_system_conv1d(audio, reverb_kernel())
bass_audio_1 = signal_system_conv1d(audio, bass_boost_kernel())
sharp_audio_1 = signal_system_conv1d(audio, sharpen_kernel())

# 保存处理后的音频
sf.write('generate2.1/noise_audio.wav', noisy_audio, sample_rate)
sf.write('generate2.1/reverb_1.wav', reverb_audio, sample_rate)
sf.write('generate2.1/bass_1.wav', bass_audio, sample_rate)
sf.write('generate2.1/sharp_1.wav', sharp_audio, sample_rate)
sf.write('generate2.1/clean_1.wav', clean_audio, sample_rate)
sf.write('generate2.2/reverb_2.wav', reverb_audio_1, sample_rate)
sf.write('generate2.2/bass_2.wav', bass_audio_1, sample_rate)
sf.write('generate2.2/sharp_2.wav', sharp_audio_1, sample_rate)
sf.write('generate2.2/clean_2.wav', clean_audio_1, sample_rate)