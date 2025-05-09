import numpy as np
import timeit

# 1. 定义输入和卷积核
input_signal = np.random.randn(256, 256)
kernel = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])

# 2. 补零填充函数
def pad_for_conv(signal, kernel):
    h, w = signal.shape
    kh, kw = kernel.shape
    padded = np.zeros((h + kh - 1, w + kw - 1))
    padded[kh//2 : kh//2+h, kw//2 : kw//2+w] = signal
    return padded

padded_input = pad_for_conv(input_signal, kernel)

# 3. 时域卷积实现
def time_domain_conv(signal, kernel):
    kh, kw = kernel.shape
    h, w = signal.shape
    output = np.zeros((h - kh + 1, w - kw + 1))
    for i in range(h - kh + 1):
        for j in range(w - kw + 1):
            output[i, j] = np.sum(signal[i:i+kh, j:j+kw] * kernel)
    return output

# 4. 频域卷积实现
def freq_domain_conv(signal, kernel):
    fh = signal.shape[0] + kernel.shape[0] - 1
    fw = signal.shape[1] + kernel.shape[1] - 1
    signal_pad = np.pad(signal, ((0, fh - signal.shape[0]), (0, fw - signal.shape[1])))
    kernel_pad = np.pad(kernel, ((0, fh - kernel.shape[0]), (0, fw - kernel.shape[1])))
    fft_signal = np.fft.fft2(signal_pad)
    fft_kernel = np.fft.fft2(kernel_pad)
    result = np.fft.ifft2(fft_signal * fft_kernel).real
    return result[kernel.shape[0]-1:, kernel.shape[1]-1:][:signal.shape[0]-kernel.shape[0]+1, :signal.shape[1]-kernel.shape[1]+1]

# 5. 计时对比
def compare_speed():
    # 时域卷积计时（运行10次）
    time_domain_time = timeit.timeit(
        lambda: time_domain_conv(padded_input, kernel), 
        number=10
    ) / 10

    # 频域卷积计时（运行10次）
    freq_domain_time = timeit.timeit(
        lambda: freq_domain_conv(padded_input, kernel), 
        number=10
    ) / 10

    print("\n===== 卷积方法耗时对比 =====")
    print(f"时域卷积平均耗时: {time_domain_time:.6f} 秒")
    print(f"频域卷积平均耗时: {freq_domain_time:.6f} 秒")
    print(f"频域加速比: {time_domain_time/freq_domain_time:.1f} 倍")

# 执行对比
compare_speed()