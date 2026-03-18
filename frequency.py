import numpy as np

# 傅里叶变换得频谱图
def frequency_analysis(lf_data):
    count = len(lf_data)
    avg_img_gray = None
    
    # --- 1. 计算像素空间的平均图 ---
    for (u, v), img_tensor in lf_data.items():
        # 转为 numpy 并处理维度
        img_np = img_tensor.detach().cpu().numpy()
        if img_np.ndim == 4: # [1, C, H, W] -> [C, H, W]
            img_np = img_np[0]
            
        # 标准灰度转换 (0.299R + 0.587G + 0.114B)
        if img_np.shape[0] == 3:
            img_gray = 0.299 * img_np[0] + 0.587 * img_np[1] + 0.114 * img_np[2]
        else:
            img_gray = img_np[0] # 本身就是灰度图

        # 累加像素
        if avg_img_gray is None:
            avg_img_gray = img_gray.astype(np.float64)
        else:
            avg_img_gray += img_gray

    # 得到像素均值图
    avg_img_gray /= count

    # 使用NumPy的fft2函数对图像进行快速傅里叶变换，得到频域表示
    fft = np.fft.fft2(avg_img_gray)

    # 使用fftshift将变换后的零频率分量移动到频谱中心，以便于观察
    fft_shift = np.fft.fftshift(fft)

    # 计算功率谱：通过取绝对值（幅度）得到频率的幅度谱（即功率谱）
    magnitude = np.abs(fft_shift)

    # 创建角度网络
    h, w = img_gray.shape
    y, x = np.ogrid[-h//2:h//2, -w//2:w//2]
    angle = np.arctan2(y, x) * 180 / np.pi
    
    # 分区统计
    energy_by_angle = {}
    angles = [0, 45, 90, 135]
    tolerance = 22.5

    for base_angle in angles:
        # 创建掩码
        angle_mask = ((angle >= base_angle - tolerance) &
                      (angle <= base_angle + tolerance) | \
                     ((angle >= base_angle - 180 - tolerance)) &
                      (angle <= base_angle - 180 + tolerance))

        energy_by_angle[base_angle] = np.sum(magnitude[angle_mask])        

    # 计算总能量
    total_energy = sum(energy_by_angle.values())
    
    # 计算每个方向能量占总能量的百分比
    percentage_by_angle = {}
    for angle, energy in energy_by_angle.items():
        percentage_by_angle[angle] = (energy / total_energy) if total_energy != 0 else 0
    
    # 按照angles顺序返回百分比列表
    percentage_list = [percentage_by_angle[angle] for angle in angles]

    ratio_list = []
    ratio_list.append(percentage_list[1])
    ratio_list.append(percentage_list[3])
    ratio_list.append(percentage_list[0])
    ratio_list.append(percentage_list[2])
    
    return ratio_list