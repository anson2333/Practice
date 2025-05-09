import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from mpl_toolkits.mplot3d import Axes3D

# 读取图像并转为灰度
img = np.array(Image.open('lena.jpg').convert('L'))

# 傅里叶变换
fft = np.fft.fft2(img)  # 二维FFT
fft_shift = np.fft.fftshift(fft)  # 中心化
amplitude = np.abs(fft_shift)
phase = np.angle(fft_shift)

# 创建坐标网格
rows, cols = img.shape
x = np.arange(-cols//2, cols//2)
y = np.arange(-rows//2, rows//2)
X, Y = np.meshgrid(x, y)

# 设置全局字体
plt.rcParams.update({'font.size': 12})

# 1. 振幅谱3D曲面图
fig = plt.figure(figsize=(15, 6))

# 振幅谱3D图
ax1 = fig.add_subplot(121, projection='3d')
surf = ax1.plot_surface(X, Y, np.log(1+amplitude),
                      cmap='viridis',
                      rstride=5,
                      cstride=5)
ax1.set_title('Amplitude Spectrum (3D Surface)')
ax1.set_xlabel('Frequency (u)')
ax1.set_ylabel('Frequency (v)')
ax1.set_zlabel('log(Amplitude)')
fig.colorbar(surf, ax=ax1, shrink=0.5)

# 相位谱3D曲面图
ax2 = fig.add_subplot(122, projection='3d')
surf2 = ax2.plot_surface(X, Y, phase,
                        cmap='hsv',
                        rstride=5,
                        cstride=5)
ax2.set_title('Phase Spectrum (3D Surface)')
ax2.set_xlabel('Frequency (u)')
ax2.set_ylabel('Frequency (v)')
ax2.set_zlabel('Phase (radians)')
fig.colorbar(surf2, ax=ax2, shrink=0.5)

plt.tight_layout()
plt.show()

# 2. 二维等高线图
plt.figure(figsize=(15, 6))

# 振幅谱等高线
plt.subplot(121)
contour = plt.contour(X, Y, np.log(1+amplitude), 15, cmap='hot')
plt.clabel(contour, inline=True, fontsize=8)
plt.title('Amplitude Spectrum (Contour)')
plt.xlabel('Frequency (u)')
plt.ylabel('Frequency (v)')
plt.colorbar()

# 相位谱等高线
plt.subplot(122)
phase_contour = plt.contour(X, Y, phase, 15, cmap='twilight')
plt.clabel(phase_contour, inline=True, fontsize=8)
plt.title('Phase Spectrum (Contour)')
plt.xlabel('Frequency (u)')
plt.ylabel('Frequency (v)')
plt.colorbar()

plt.tight_layout()
plt.show()

#2. 振幅谱与相位谱重要性实验
# 实验1：仅用振幅谱重建
recon_amp = np.fft.ifft2(np.fft.ifftshift(amplitude * np.exp(0j)))  # 相位设为0

# 实验2：仅用相位谱重建
recon_phase = np.fft.ifft2(np.fft.ifftshift(1 * np.exp(1j * phase)))  # 振幅设为1

# 可视化对比
plt.figure(figsize=(10, 5))
plt.subplot(131), plt.imshow(img, cmap='gray'), plt.title('Original')
plt.subplot(132), plt.imshow(np.abs(recon_amp), cmap='gray'), plt.title('Amplitude Only')
plt.subplot(133), plt.imshow(np.abs(recon_phase), cmap='gray'), plt.title('Phase Only')
plt.show()

# 结论：
#
# 相位谱更重要：仅用相位谱重建的图像能保留边缘结构，而仅用振幅谱的图像几乎无结构信息。
#
# 原因：相位谱决定了信号中各频率分量的时间关系（即结构），而振幅谱仅影响强度。


# 3. 频域滤波实现效果
# 设计高通滤波器（保留高频）
rows, cols = img.shape
crow, ccol = rows//2, cols//2
mask = np.ones((rows, cols))
mask[crow-30:crow+30, ccol-30:ccol+30] = 0  # 去除中心低频区域

# 滤波并反变换
filtered = fft_shift * mask
img_highpass = np.abs(np.fft.ifft2(np.fft.ifftshift(filtered)))

# 显示结果
plt.imshow(img_highpass, cmap='gray')
plt.title('High-pass Filtered')
plt.show()

# 效果：高通滤波后得到边缘增强结果（类似卷积中的锐化效果）。