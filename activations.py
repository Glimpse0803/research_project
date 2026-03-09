#activations.py
import torch
import torch.nn as nn
import math
import numpy as np


'''sinc'''
class SincActivation(nn.Module):
    def __init__(self, alpha_init=1.0):
        super().__init__()
        self.alpha_init = nn.Parameter(torch.tensor(alpha_init))

    def forward(self, x):
        # sin(alpha*x)/(alpha*x)
        scaled_x = self.alpha_init * x / torch.pi
        return torch.sinc(scaled_x)


'''
class MultiScaleSincActivation(nn.Module):
    """
    多尺度Sinc激活函数
    将输入通道分成多组，每组使用不同缩放因子的sinc函数
    使用torch.sinc实现：sinc(x) = sin(πx)/(πx)
    """

    def __init__(self, channels, alpha_inits=[0.5, 1.0, 2.0], trainable=True):
        super().__init__()
        self.channels = channels
        self.num_groups = len(alpha_inits)
        # 计算每组通道数
        base_channels = channels // self.num_groups
        remainder = channels % self.num_groups
        self.group_channels = [base_channels] * self.num_groups
        # 将余数分配到前面的组
        for i in range(remainder):
            self.group_channels[i] += 1
        # 创建可训练参数或缓冲区
        if trainable:
            self.alpha_inits = nn.Parameter(torch.tensor(alpha_inits, dtype=torch.float32))
        else:
            self.register_buffer('alpha_inits', torch.tensor(alpha_inits, dtype=torch.float32))

    def forward(self, x):
        # 将输入在通道维度上分成多组
        start_idx = 0
        groups = []
        for group_size in self.group_channels:
            end_idx = start_idx + group_size
            groups.append(x[:, start_idx:end_idx, :, :])
            start_idx = end_idx

        processed_groups = []
        for i, group in enumerate(groups):
            alpha_init = self.alpha_inits[i]
            scaled_x = alpha_init * group / torch.pi
            sinc_result = torch.sinc(scaled_x)
            processed_groups.append(sinc_result)
        # 在通道维度上重新连接
        return torch.cat(processed_groups, dim=1)
'''

'''
class RaisedCosineActivation(nn.Module):
    """
    升余弦激活函数
    频域特性: 比sinc更快的衰减，更好的频域局部化
    """

    def __init__(self, beta=0.35, scale=1.0, learnable=True):
        super().__init__()
        if learnable:
            self.beta = nn.Parameter(torch.tensor(beta, dtype=torch.float32))
            self.scale = nn.Parameter(torch.tensor(scale, dtype=torch.float32))
        else:
            self.register_buffer('beta', torch.tensor(beta, dtype=torch.float32))
            self.register_buffer('scale', torch.tensor(scale, dtype=torch.float32))



    def forward(self, x):
        """安全版 升余弦激活函数"""
        x_scaled = self.scale * x
        result = torch.zeros_like(x_scaled)

        abs_x = torch.abs(x_scaled)
        one = torch.tensor(1.0, device=x.device)
        beta = torch.clamp(self.beta, 1e-6, 1.0 - 1e-6)  # 避免 0 或 1

        # 主瓣区域
        mask1 = abs_x <= (1 - beta) / 2
        result[mask1] = 1.0

        # 过渡区域
        mask2 = (abs_x > (1 - beta) / 2) & (abs_x <= (1 + beta) / 2)
        t = abs_x[mask2]
        result[mask2] = 0.5 * (1 + torch.cos(math.pi / beta * (t - (1 - beta) / 2)))

        # 限幅和 NaN 清理
        result = torch.clamp(result, 0.0, 1.0)
        result = torch.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)

        return result
'''
'''
class RRCActivation(nn.Module):
    """
    根升余弦激活函数
    频域特性: 升余弦的平方根，更好的频域特性
    """

    def __init__(self, beta=0.35, scale=1.0, learnable=True):
        super().__init__()
        if learnable:
            self.beta = nn.Parameter(torch.tensor(beta, dtype=torch.float32))
            self.scale = nn.Parameter(torch.tensor(scale, dtype=torch.float32))
        else:
            self.register_buffer('beta', torch.tensor(beta, dtype=torch.float32))
            self.register_buffer('scale', torch.tensor(scale, dtype=torch.float32))



    def forward(self, x):
        """安全版 根升余弦激活函数"""
        x_scaled = self.scale * x
        result = torch.zeros_like(x_scaled)

        beta = torch.clamp(self.beta, 1e-6, 1.0 - 1e-6)
        four_bx = 4 * beta * x_scaled

        # 分母防止除零
        denom = math.pi * x_scaled * (1 - four_bx ** 2)
        safe_denom = torch.where(torch.abs(denom) < 1e-6, torch.ones_like(denom), denom)

        term1 = torch.sin(math.pi * x_scaled * (1 - beta))
        term2 = four_bx * torch.cos(math.pi * x_scaled * (1 + beta))
        result = (term1 + term2) / safe_denom

        # 极限值区域（|4βx| ≈ 1）
        mask_special = torch.abs(4 * beta * x_scaled) >= 0.999
        if mask_special.any():
            special_val = (beta / math.sqrt(2)) * (
                    (1 + 2 / math.pi) * math.sin(math.pi / (4 * beta)) +
                    (1 - 2 / math.pi) * math.cos(math.pi / (4 * beta))
            )
            result[mask_special] = special_val

        # 限幅、NaN清理
        result = torch.clamp(result, -5.0, 5.0)
        result = torch.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)

        return result
'''

'''gabor'''
class ComplexGaborWaveletActivation(nn.Module):
    """Complex Gabor Wavelet activation: exp(j**x - ||*x²)"""

    def __init__(self, alpha_init=3.0, gamma_init=3.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha_init))
        self.gamma = nn.Parameter(torch.tensor(gamma_init))

    def forward(self, x):
        # Real part of complex Gabor wavelet
        real_part = torch.exp(-torch.abs(self.gamma) * x * x) * torch.cos(self.alpha * x)
        return real_part


'''高斯'''


class GaussianActivation(nn.Module):
    """Gaussian activation:exp(-|r|*x²)"""

    def __init__(self, gamma_init=2.0):
        super().__init__()
        self.gamma = nn.Parameter(torch.tensor(gamma_init))

    def forward(self, x):
        return torch.exp(-torch.abs(self.gamma) * x * x)


'''Sinusoid'''


class SinusoidActivation(nn.Module):
    '''sin(α*x + β)'''

    def __init__(self, alpha_init=0.5, beta_init=0.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha_init))
        self.beta = nn.Parameter(torch.tensor(beta_init))

    def forward(self, x):
        return torch.sin(self.alpha * x + self.beta)


'''gelu'''
class GELUActivation(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(1.0 / torch.pi))
            * (x + 0.044715 * torch.pow(x, 3))
        ))


'''relu'''
'''
class ReLUActivation(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x)
'''

# 激活函数类
class ActivationFactory:
    """激活函数类，方便创建和管理"""

    @staticmethod
    def create(activation_type, **kwargs):
        """
        创建激活函数实例

        Args:
            activation_type: 激活函数类型
            **kwargs: 激活函数参数

        Returns:
            nn.Module: 激活函数实例
        """
        activations = {
            'sinc': SincActivation,
            # 'relu': ReLUActivation,
            'gelu': GELUActivation,
            'gabor': ComplexGaborWaveletActivation,
            'gaussian': GaussianActivation,
            'sinusoid': SinusoidActivation
        }

        if activation_type not in activations:
            raise ValueError(f"未知的激活函数类型: {activation_type}")

        return activations[activation_type](**kwargs)

    @staticmethod
    def get_available_activations():
        """获取所有可用的激活函数类型"""
        return ['sinc', 'gelu', 'gabor', 'gaussian', 'sinusoid']




Sinc = SincActivation
# RC = RaisedCosineActivation
# Relu = ReLUActivation
Gelu = GELUActivation
# RRC = RRCActivation
Gabor = ComplexGaborWaveletActivation
Gaussian = GaussianActivation
Sinusoid = SinusoidActivation

def get_activation_instance(name):
    # 每次调用返回一个新的实例，确保参数独立
    # if name == 'relu': return ReLUActivation()
    if name == 'gelu': return GELUActivation()
    if name == 'sinc': return SincActivation(alpha_init=1.0)
    if name == 'gaussian': return GaussianActivation(gamma_init=2.0)
    if name == 'gabor': return ComplexGaborWaveletActivation(alpha_init=3.0, gamma_init=3.0)
    if name == 'sinusoid': return SinusoidActivation(alpha_init=0.5)
    raise ValueError(f"Unknown activation: {name}")

ACT_DICT = ['gelu', 'sinc', 'gaussian', 'gabor', 'sinusoid']