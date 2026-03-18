# 模型验证
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import numbers
import pywt
import scipy
import skimage.color as color
from skimage.restoration import (denoise_wavelet, estimate_sigma)
from skimage import data, img_as_float
from skimage.util import random_noise
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
import os
from PIL import Image
import PIL
import warnings
warnings.filterwarnings('ignore')
import torch
import torch.optim
from torch.autograd import Variable
import random
from torchvision import transforms
import cv2
from skimage.metrics import peak_signal_noise_ratio as psnr_metric

from include import *



def seed_everything(seed):
    torch.manual_seed(seed)       # Current CPU
    torch.cuda.manual_seed(seed)  # Current GPU
    np.random.seed(seed)          # Numpy module
    random.seed(seed)             # Python random module
    torch.backends.cudnn.benchmark = False    # Close optimization
    torch.backends.cudnn.deterministic = True # Close optimization
    torch.cuda.manual_seed_all(seed) # All GPU (Optional)

def load_model(model_path, model_class, **kwargs):
    """
    加载已保存的模型
    
    参数:
        model_path: 模型文件路径
        model_class: 模型类
        **kwargs: 模型初始化参数
    
    返回:
        加载的模型
    """
    print(f"正在加载模型: {model_path}")
    
    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        print(f"警告: 模型文件不存在: {model_path}")
        print("将创建随机初始化的模型作为示例")
        model = model_class(**kwargs)
        return model
    
    try:
        # 实例化模型
        model = model_class(**kwargs)
        
        # 加载模型权重
        checkpoint = torch.load(model_path, map_location=device)
        
        # 根据保存方式加载
        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
                print("从checkpoint字典加载模型权重")
            elif 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                print("从checkpoint字典加载模型权重")
            else:
                # 尝试直接加载
                model.load_state_dict(checkpoint)
                print("直接加载模型权重")
        else:
            model.load_state_dict(checkpoint)
            print("直接加载模型权重")
        
        print("模型加载成功!")
        
    except Exception as e:
        print(f"模型加载失败: {e}")
        print("将使用随机初始化的模型")
        model = model_class(**kwargs)
    
    # 设置为评估模式
    model.eval()
    
    return model

def main():
    seed_everything(42)

    # 加载输入噪声文件
    noise_data = torch.load("./fixed_noise_k64_32x32.pt")

    # Tensor
    net_input = Variable(noise_data).type(dtype)

    # 检查是否有GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    