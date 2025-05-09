from PIL import Image
import matplotlib.pyplot as plt
from practice1 import conv2d
from practice1 import signal_system_conv2d  #未添加
import random
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # Avoids OpenMP conflict

# 读取图像并转为灰度图（简化处理）
def load_image(path):
    img = Image.open(path).convert('L')  # 转为灰度
    return img

# 图像显示与保存
def show_save(img, filename):
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.show()

# 加载测试图像
original_img = load_image('lena.jpg')
width, height = original_img.size

# 将PIL图像转为二维列表
def image_to_matrix(img):
    return [[img.getpixel((x, y)) for x in range(width)] for y in range(height)]

# 将矩阵转回PIL图像
def matrix_to_image(matrix):
    img = Image.new('L', (len(matrix[0]), len(matrix)))
    for y in range(len(matrix)):
        for x in range(len(matrix[0])):
            # Clip values to [0, 255] and convert to int
            pixel_value = int(max(0, min(255, matrix[y][x])))
            img.putpixel((x, y), pixel_value)
    return img


def gaussian_kernel(size=3, sigma=1.0):
    """生成高斯核"""
    kernel = [[0] * size for _ in range(size)]
    center = size // 2
    total = 0
    for i in range(size):
        for j in range(size):
            x, y = i - center, j - center
            kernel[i][j] = (1 / (2 * 3.1416 * sigma ** 2)) * 2.718 ** (-(x ** 2 + y ** 2) / (2 * sigma ** 2))
            total += kernel[i][j]
    # 归一化
    return [[val / total for val in row] for row in kernel]



# 高斯模糊测试
img_matrix = image_to_matrix(original_img)
blurred = conv2d(img_matrix, gaussian_kernel(5, 1.5))
show_save(matrix_to_image(blurred), 'blurred.jpg')

#效果：
#高斯核通过加权平均模糊图像，权重服从二维正态分布，中心像素影响最大，能有效抑制高频噪声。


def sobel_kernel():
    """Sobel边缘检测核"""
    return [
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ]

# 使用两个方向核检测边缘
sobel_x = conv2d(img_matrix, sobel_kernel())
sobel_y = conv2d(img_matrix, [list(row) for row in zip(*sobel_kernel())])  # 转置得y方向核

# 合并两个方向结果
edge_matrix = [[min(255, int((x**2 + y**2)**0.5)) for x, y in zip(row_x, row_y)]
              for row_x, row_y in zip(sobel_x, sobel_y)]
show_save(matrix_to_image(edge_matrix), 'edges.jpg')
# 效果分析：
# Sobel核通过突出水平/垂直方向的像素值突变，正负权重组合能增强边缘响应。

def sharpen_kernel():
    """图像锐化核"""
    return [
        [ 0, -1,  0],
        [-1,  5, -1],
        [ 0, -1,  0]
    ]

sharpened = conv2d(img_matrix, sharpen_kernel())
show_save(matrix_to_image(sharpened), 'sharpened.jpg')

# 效果分析：
# 中心正权重（5）增强原像素，周围负权重（-1）抑制邻域像素，通过差分放大高频细节。

# 添加椒盐噪声
def add_noise(matrix, prob=0.05):
    noisy = [row.copy() for row in matrix]
    for i in range(len(noisy)):
        for j in range(len(noisy[0])):
            if random.random() < prob:
                noisy[i][j] = 0 if random.random() < 0.5 else 255
    return noisy


noisy_img = add_noise(img_matrix)
show_save(matrix_to_image(noisy_img), 'noisy.jpg')


# 中值滤波去噪（非线性滤波，需手动实现）
def median_filter(matrix, size=3):
    h, w = len(matrix), len(matrix[0])
    padded = [[0] * (w + size - 1) for _ in range(h + size - 1)]
    for i in range(h):
        for j in range(w):
            padded[i + size // 2][j + size // 2] = matrix[i][j]

    result = [[0] * w for _ in range(h)]
    for i in range(h):
        for j in range(w):
            window = []
            for x in range(size):
                for y in range(size):
                    window.append(padded[i + x][j + y])
            window.sort()
            result[i][j] = window[size * size // 2]
    return result


denoised = median_filter(noisy_img)
show_save(matrix_to_image(denoised), 'denoised.jpg')

# 效果分析：
# 椒盐噪声：随机像素变为黑白极值
# 中值滤波：取邻域中值，有效消除孤立噪声点


# 加载测试图像
original_img = load_image('lena.jpg')
img_matrix = image_to_matrix(original_img)

# 创建大图
plt.figure(figsize=(15, 10))

# 显示原始图像
plt.subplot(2, 3, 1)
plt.imshow(original_img, cmap='gray')
plt.title('Original')
plt.axis('off')

# 高斯模糊
blurred = conv2d(img_matrix, gaussian_kernel(5, 1.5))
plt.subplot(2, 3, 2)
plt.imshow(matrix_to_image(blurred), cmap='gray')
plt.title('Blurred')
plt.axis('off')

# Sobel边缘检测
sobel_x = conv2d(img_matrix, sobel_kernel())
sobel_y = conv2d(img_matrix, [list(row) for row in zip(*sobel_kernel())])
edge_matrix = [[min(255, int((x**2 + y**2)**0.5)) for x, y in zip(row_x, row_y)]
              for row_x, row_y in zip(sobel_x, sobel_y)]
plt.subplot(2, 3, 3)
plt.imshow(matrix_to_image(edge_matrix), cmap='gray')
plt.title('Edges')
plt.axis('off')

# 图像锐化
sharpened = conv2d(img_matrix, sharpen_kernel())
plt.subplot(2, 3, 4)
plt.imshow(matrix_to_image(sharpened), cmap='gray')
plt.title('Sharpened')
plt.axis('off')

# 添加椒盐噪声
noisy_img = add_noise(img_matrix)
plt.subplot(2, 3, 5)
plt.imshow(matrix_to_image(noisy_img), cmap='gray')
plt.title('Noisy')
plt.axis('off')

# 中值滤波去噪
denoised = median_filter(noisy_img)
plt.subplot(2, 3, 6)
plt.imshow(matrix_to_image(denoised), cmap='gray')
plt.title('Denoised')
plt.axis('off')

# 调整布局
plt.tight_layout()
plt.savefig('combined_results.jpg', bbox_inches='tight', pad_inches=0)
plt.show()