import torch
import torch.nn.functional as F
import numpy as np

#********利用pytorch库实现离散时间一维卷积
def signal_system_conv1d(signal, kernel):
    """
    使用PyTorch实现信号与系统定义的一维卷积
    :param signal: 输入信号列表或NumPy数组 [s1, s2, ..., sn]
    :param kernel: 卷积核列表或NumPy数组 [k1, k2, ..., km]
    :return: 卷积结果列表
    """
    # 转换为PyTorch张量
    signal_tensor = torch.tensor(signal, dtype=torch.float32).view(1, 1, -1)  # shape: [1, 1, N]

    # 反转卷积核（信号与系统要求）并复制以避免负步长
    flipped_kernel = torch.tensor(kernel[::-1].copy(), dtype=torch.float32).view(1, 1, -1)  # shape: [1, 1, M]

    # 执行卷积（padding='full'保证输出长度为N+M-1）
    result = F.conv1d(signal_tensor, flipped_kernel, padding=len(kernel) - 1)

    return result.squeeze().tolist()


# 测试
signal = [1, 2, 3, 4, 5]
kernel = [1, 0, -1]
print("PyTorch一维卷积:", signal_system_conv1d(signal, kernel))

#*******利用pytorch库实现离散时间二维卷积
def signal_system_conv2d(image, kernel):
    """
    使用PyTorch实现信号与系统定义的二维卷积
    :param image: 二维输入信号 (列表的列表)
    :param kernel: 二维卷积核 (列表的列表)
    :return: 卷积结果矩阵
    """
    # 转换为PyTorch张量
    image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # shape: [1, 1, H, W]

    # 反转卷积核（信号与系统要求：水平和垂直都反转）
    flipped_kernel = torch.tensor([row[::-1] for row in kernel[::-1]],
                                  dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # shape: [1, 1, KH, KW]

    # 计算padding大小（保证输出尺寸为H+KH-1, W+KW-1）
    padding = (len(kernel) - 1, len(kernel[0]) - 1)

    # 执行卷积
    result = F.conv2d(image_tensor, flipped_kernel, padding=padding)

    return result.squeeze().tolist()


# 测试
image = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]

kernel = [
    [1, 0, -1],
    [1, 0, -1],
    [1, 0, -1]
]

print("PyTorch二维卷积:")
for row in signal_system_conv2d(image, kernel):
    print(row)
"""
输出:
[-6.0, -9.0, -12.0, 6.0, 3.0]
[-9.0, -12.0, -15.0, 6.0, 3.0]
[-12.0, -15.0, -18.0, 6.0, 3.0]
[12.0, 15.0, 18.0, -6.0, -3.0]
[7.0, 8.0, 9.0, -4.0, -5.0]
"""

#*******手动实现离散时间一维卷积
def conv1d(signal, kernel):
    """
    一维离散卷积实现
    :param signal: 输入信号 (NumPy array)
    :param kernel: 卷积核 (NumPy array)
    :return: 卷积结果 (NumPy array)
    """
    # 反转卷积核
    kernel = kernel[::-1]
    m, n = len(kernel), len(signal)
    # 输出长度 = 输入长度 + 核长度 - 1
    output_length = n + m - 1
    result = np.zeros(output_length)

    # 零填充输入信号
    padded_signal = np.concatenate([np.zeros(m - 1), signal, np.zeros(m - 1)])

    # 滑动窗口计算卷积
    for i in range(output_length):
        window = padded_signal[i:i + m]
        # 点积计算
        result[i] = np.sum(window * kernel)

    return result
# 测试手写的一维卷积
signal_1d = [1, 2, 3, 4, 5]
kernel_1d = [1, 0, -1]
print("一维卷积结果:", conv1d(signal_1d, kernel_1d))
# 应输出: [-1, -2, -2, -2, 4, -5]

def conv2d(image, kernel):
    """
    二维离散卷积实现 (适用于Python列表)
    :param image: 二维输入信号 (列表的列表)
    :param kernel: 二维卷积核 (列表的列表)
    :return: 卷积结果 (列表的列表)
    """
    # 获取图像和卷积核尺寸
    img_h, img_w = len(image), len(image[0])
    ker_h, ker_w = len(kernel), len(kernel[0])

    # 反转卷积核 (水平和垂直方向)
    kernel = [row[::-1] for row in kernel][::-1]

    # 计算输出尺寸
    out_h = img_h + ker_h - 1
    out_w = img_w + ker_w - 1

    # 创建输出矩阵并初始化
    result = [[0 for _ in range(out_w)] for _ in range(out_h)]

    # 零填充输入图像
    padded_img = []
    pad_h = ker_h - 1
    pad_w = ker_w - 1

    # 顶部填充
    padded_img.extend([[0] * (img_w + 2 * pad_w) for _ in range(pad_h)])

    # 中间行 (左右填充)
    for row in image:
        padded_row = [0] * pad_w + row + [0] * pad_w
        padded_img.append(padded_row)

    # 底部填充
    padded_img.extend([[0] * (img_w + 2 * pad_w) for _ in range(pad_h)])

    # 执行卷积
    for i in range(out_h):
        for j in range(out_w):
            # 提取当前窗口
            window = [
                row[j:j + ker_w]
                for row in padded_img[i:i + ker_h]
            ]

            # 计算点积
            conv_sum = 0
            for x in range(ker_h):
                for y in range(ker_w):
                    conv_sum += window[x][y] * kernel[x][y]
            result[i][j] = conv_sum

    return result

# 测试二维卷积
image_2d = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]

kernel_2d = [
    [1, 0, -1],
    [1, 0, -1],
    [1, 0, -1]
]

print("二维卷积结果:")

for row in conv2d(image_2d, kernel_2d):
    print(row)
"""
应输出类似:
[-6, -9, -12, 6, 3]
[-9, -12, -15, 6, 3]
[-12, -15, -18, 6, 3]
[12, 15, 18, -6, -3]
[7, 8, 9, -4, -5]
"""


