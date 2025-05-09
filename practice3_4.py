import cv2
import numpy as np
import matplotlib.pyplot as plt

def fft_analysis(img):
    """计算图像的FFT，返回振幅谱和相位谱"""
    dft = np.fft.fft2(img)
    dft_shift = np.fft.fftshift(dft)  # 低频移到中心
    amplitude = np.abs(dft_shift)     # 振幅谱
    phase = np.angle(dft_shift)       # 相位谱
    return amplitude, phase

def fft_reconstruct(amplitude, phase):
    """从振幅谱和相位谱重建图像"""
    combined = amplitude * np.exp(1j * phase)  # 振幅 * e^(j·相位)
    idft_shift = np.fft.ifftshift(combined)   # 移回原始FFT排列
    img_reconstructed = np.fft.ifft2(idft_shift)
    img_reconstructed = np.abs(img_reconstructed)  # 取模（去除虚部）
    return img_reconstructed

def normalize(img):
    """归一化到0-255"""
    img_normalized = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    return img_normalized.astype(np.uint8)

# 读取图像
img = cv2.imread('lena.jpg', 0)  # 灰度模式
if img is None:
    raise FileNotFoundError("请替换为你的图像路径！")

# 计算FFT、振幅谱和相位谱
amplitude, phase = fft_analysis(img)

# 实验1：仅保留振幅谱（相位=0）
reconstructed_amplitude_only = fft_reconstruct(amplitude, np.zeros_like(phase))
reconstructed_amplitude_only = normalize(reconstructed_amplitude_only)

# 实验2：仅保留相位谱（振幅=1）
reconstructed_phase_only = fft_reconstruct(np.ones_like(amplitude), phase)
reconstructed_phase_only = normalize(reconstructed_phase_only)

# 显示结果
plt.figure(figsize=(12, 6))
plt.subplot(131), plt.imshow(img, cmap='gray')
plt.title('Original Image'), plt.axis('off')

plt.subplot(132), plt.imshow(reconstructed_amplitude_only, cmap='gray')
plt.title('Amplitude Only\n(Phase=0)'), plt.axis('off')

plt.subplot(133), plt.imshow(reconstructed_phase_only, cmap='gray')
plt.title('Phase Only\n(Amplitude=1)'), plt.axis('off')

plt.tight_layout()
plt.show()