import librosa
import matplotlib.pyplot as plt

# 加载原始音频和处理后的音频
y_orig, sr = librosa.load("input/14-208-0048.wav", sr=None)
y_reverb, _ = librosa.load("generate2.1/reverb_1.wav", sr=sr)
y_sharp, _ = librosa.load("generate2.1/sharp_1.wav", sr=sr)
y_denoise, _ = librosa.load("generate2.1/clean_1.wav", sr=sr)
y_lowpass, _ = librosa.load("generate2.1/bass_1.wav", sr=sr)
#
# # 绘制波形对比
# plt.figure(figsize=(14, 8))
# plt.subplot(5, 1, 1)
# librosa.display.waveshow(y_orig, sr=sr, color="blue", alpha=0.7, label="Original")
# plt.legend()
#
# plt.subplot(5, 1, 2)
# librosa.display.waveshow(y_reverb, sr=sr, color="green", alpha=0.7, label="Reverb")
# plt.legend()
#
# plt.subplot(5, 1, 3)
# librosa.display.waveshow(y_sharp, sr=sr, color="red", alpha=0.7, label="Sharpened")
# plt.legend()
#
# plt.subplot(5, 1, 4)
# librosa.display.waveshow(y_denoise, sr=sr, color="purple", alpha=0.7, label="Denoised")
# plt.legend()
#
# plt.subplot(5, 1, 5)
# librosa.display.waveshow(y_lowpass, sr=sr, color="orange", alpha=0.7, label="Low-pass")
# plt.legend()
#
# plt.tight_layout()
# plt.show()
#
# import numpy as np
#
# def plot_spectrogram(y, sr, title):
#     D = librosa.stft(y)
#     S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
#     plt.figure(figsize=(10, 4))
#     librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='log')
#     plt.colorbar(format='%+2.0f dB')
#     plt.title(title)
#     plt.show()
#
# plot_spectrogram(y_orig, sr, "Original")
# plot_spectrogram(y_reverb, sr, "Reverb")
# plot_spectrogram(y_sharp, sr, "Sharpened")
# plot_spectrogram(y_denoise, sr, "Denoised")
# plot_spectrogram(y_lowpass, sr, "Low-pass")
#
#
# def plot_frequency_response(y, sr, label):
#     n_fft = 2048
#     Y = librosa.stft(y, n_fft=n_fft)
#     Y_db = librosa.amplitude_to_db(np.abs(Y), ref=np.max)
#     Y_avg = np.mean(Y_db, axis=1)  # 取时间平均
#     freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
#     plt.semilogx(freqs, Y_avg, label=label)
#     plt.show()
#
# plt.figure(figsize=(10, 5))
# plot_frequency_response(y_orig, sr, "Original")
# plot_frequency_response(y_reverb, sr, "Reverb")
# plot_frequency_response(y_sharp, sr, "Sharpened")
# plot_frequency_response(y_denoise, sr, "Denoised")
# plot_frequency_response(y_lowpass, sr, "Low-pass")
# plt.xlabel("Frequency (Hz)")
# plt.ylabel("Amplitude (dB)")
# plt.legend()
# plt.grid()
# import numpy as np
# import librosa.display
# import matplotlib.pyplot as plt
#
# def plot_spectrogram(y, sr, title, ax):
#     D = librosa.stft(y)
#     S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
#     img = librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='log', ax=ax)
#     ax.set_title(title)
#     return img
#
# # 创建子图布局
# fig, axes = plt.subplots(5, 1, figsize=(14, 12))
#
# # 绘制原始音频频谱图
# img1 = plot_spectrogram(y_orig, sr, "Original", axes[0])
#
# # 绘制混响音频频谱图
# img2 = plot_spectrogram(y_reverb, sr, "Reverb", axes[1])
#
# # 绘制锐化音频频谱图
# img3 = plot_spectrogram(y_sharp, sr, "Sharpened", axes[2])
#
# # 绘制去噪音频频谱图
# img4 = plot_spectrogram(y_denoise, sr, "Denoised", axes[3])
#
# # 绘制低通滤波音频频谱图
# img5 = plot_spectrogram(y_lowpass, sr, "Low-pass", axes[4])
#
# # 添加颜色条
# cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
# fig.colorbar(img5, cax=cbar_ax, format='%+2.0f dB')
#
# # 调整布局
# plt.tight_layout(rect=[0, 0, 0.9, 1])  # 留出右侧空间给颜色条
# plt.show()

import numpy as np
import librosa.display
import matplotlib.pyplot as plt

def plot_frequency_response(y, sr, label):
    n_fft = 2048
    Y = librosa.stft(y, n_fft=n_fft)
    Y_db = librosa.amplitude_to_db(np.abs(Y), ref=np.max)
    Y_avg = np.mean(Y_db, axis=1)  # 取时间平均
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    plt.semilogx(freqs, Y_avg, label=label)  # 在同一图中绘制曲线

# 创建图形
plt.figure(figsize=(10, 5))

# 绘制所有频率响应曲线
plot_frequency_response(y_orig, sr, "Original")
plot_frequency_response(y_reverb, sr, "Reverb")
plot_frequency_response(y_sharp, sr, "Sharpened")
plot_frequency_response(y_denoise, sr, "Denoised")
plot_frequency_response(y_lowpass, sr, "Low-pass")

# 添加标签、图例和网格
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude (dB)")
plt.legend()
plt.grid()

# 显示图形
plt.show()