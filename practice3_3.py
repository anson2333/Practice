import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# 读取图片并转为灰度
img = Image.open("lena.jpg").convert("L")
img_array = np.array(img)

# 计算2D傅里叶变换
fft_img = np.fft.fft2(img_array)  # 2D FFT
fft_shifted = np.fft.fftshift(fft_img)  # 低频移到中心

# 计算振幅谱和相位谱
magnitude_spectrum = 20 * np.log(np.abs(fft_shifted) + 1e-10)  # 对数尺度
phase_spectrum = np.angle(fft_shifted)  # 相位谱

plt.figure(figsize=(12, 6))

plt.subplot(131), plt.imshow(img_array, cmap='gray'), plt.title("Original Image")
plt.subplot(132), plt.imshow(magnitude_spectrum, cmap='gray'), plt.title("Magnitude Spectrum (log)")
plt.subplot(133), plt.imshow(phase_spectrum, cmap='gray'), plt.title("Phase Spectrum")

plt.tight_layout()
plt.show()

# 仅用振幅谱重建（随机相位）
random_phase = np.exp(1j * np.random.uniform(0, 2*np.pi, img_array.shape))
recon_mag_only = np.abs(fft_img) * random_phase  # 原始振幅 + 随机相位
recon_img_mag = np.fft.ifft2(recon_mag_only).real

# 仅用相位谱重建（振幅=1）
recon_phase_only = np.exp(1j * np.angle(fft_img))  # 原始相位 + 单位振幅
recon_img_phase = np.fft.ifft2(recon_phase_only).real

# 可视化
plt.figure(figsize=(12, 4))
plt.subplot(131), plt.imshow(img_array, cmap='gray'), plt.title("Original")
plt.subplot(132), plt.imshow(recon_img_mag, cmap='gray'), plt.title("Recon (Mag Only)")
plt.subplot(133), plt.imshow(recon_img_phase, cmap='gray'), plt.title("Recon (Phase Only)")
plt.show()

rows, cols = img_array.shape
crow, ccol = rows // 2, cols // 2  # 中心点

# 创建低通掩码（保留中心低频）
mask_lowpass = np.zeros((rows, cols), np.uint8)
mask_lowpass[crow-30:crow+30, ccol-30:ccol+30] = 1  # 保留中心 60x60 区域

# 应用滤波
fft_filtered = fft_shifted * mask_lowpass
img_lowpass = np.fft.ifft2(np.fft.ifftshift(fft_filtered)).real

mask_highpass = 1 - mask_lowpass  # 去除中心低频
fft_filtered_high = fft_shifted * mask_highpass
img_highpass = np.fft.ifft2(np.fft.ifftshift(fft_filtered_high)).real

plt.figure(figsize=(12, 4))
plt.subplot(131), plt.imshow(img_array, cmap='gray'), plt.title("Original")
plt.subplot(132), plt.imshow(img_lowpass, cmap='gray'), plt.title("Lowpass (Blur)")
plt.subplot(133), plt.imshow(img_highpass, cmap='gray'), plt.title("Highpass (Edges)")
plt.show()