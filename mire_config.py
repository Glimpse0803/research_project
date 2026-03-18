# mire_config.py

CONFIG = {
    # --- 网络结构参数 ---
    'k_channels': 64,  # 通道数 k=64
    'input_size': (32, 32),  # 输入尺寸 32x32 (经过4层上采样 -> 512x512)
    'output_size': (512, 512),  # 目标输出尺寸

    # --- MIRE 搜索阶段 (Search) ---
    'search_epochs': 2000,  # 每层搜索迭代次数
    'search_lr': 0.005,  # 搜索学习率 dd-0.005

    # --- 全局微调阶段 (Finetune) ---
    'finetune_epochs': 20000,  # 最终训练迭代次数
    'finetune_lr': 0.005,  # 初始学习率 (线性衰减至 1/10)

    # --- 正则化参数 ---
    'reg_noise_std': 0.005,  # 输入噪声扰动标准差
    'reg_noise_decay': 500,  # 噪声衰减周期

    # --- 文件路径 ---
    'noise_file': 'fixed_noise_k64_32x32.pt',  # 固定噪声文件
    'img_path': 'data/boxes_5_5.png',
    'img_dir': 'data/boxes',

    # --- 卷积层通道数调整 (控制参数量) ---
    'branch_channels': 140

}
