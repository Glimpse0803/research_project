#helpers.py
import torch
import torch.nn as nn
import torchvision
import sys

import numpy as np
from PIL import Image
import PIL

from torch.autograd import Variable

import matplotlib.pyplot as plt

from conv_layer import asy_dir_conv_block
from conv_layer import asy_dir_conv_block_nodia
from frequency import frequency_analysis
from conv_layer import DirectionalConv2d


def load_and_crop(imgname, target_width=512, target_height=512):
    '''
    imgname: string of image location
    load an image, and center-crop if the image is large enough, else return none
    '''
    img = Image.open(imgname)
    width, height = img.size
    if width <= target_width or height <= target_height:
        return None

    # 计算中心裁剪区域
    left = (width - target_width) / 2
    top = (height - target_height) / 2
    right = (width + target_width) / 2
    bottom = (height + target_height) / 2

    return img.crop((left, top, right, bottom))


def save_np_img(img, filename):
    if (img.shape[0] == 1):
        plt.imshow(np.clip(img[0], 0, 1), cmap='Greys', interpolation='nearest')
    else:
        plt.imshow(np.clip(img.transpose(1, 2, 0), 0, 1))
    plt.axis('off')
    plt.savefig(filename, bbox_inches='tight')
    plt.close()


def np_to_tensor(img_np):
    '''Converts image in numpy.array to torch.Tensor.

    From C x W x H [0..1] to  C x W x H [0..1]
    '''
    return torch.from_numpy(img_np)


def np_to_var(img_np, dtype=torch.cuda.FloatTensor):
    '''Converts image in numpy.array to torch.Variable.

    From C x W x H [0..1] to  1 x C x W x H [0..1]
    '''
    return Variable(np_to_tensor(img_np)[None, :])


def var_to_np(img_var):
    '''Converts an image in torch.Variable format to np.array.

    From 1 x C x W x H [0..1] to  C x W x H [0..1]
    '''
    return img_var.data.cpu().numpy()[0]


def pil_to_np(img_PIL):
    '''Converts image in PIL format to np.array.

    From W x H x C [0...255] to C x W x H [0..1]
    '''
    ar = np.array(img_PIL)

    if len(ar.shape) == 3:
        ar = ar.transpose(2, 0, 1)
    else:
        ar = ar[None, ...]

    return ar.astype(np.float32) / 255.  # 归一化到[0,1]


def rgb2ycbcr(img):
    # out = color.rgb2ycbcr( img.transpose(1, 2, 0) )
    # return out.transpose(2,0,1)/256.
    r, g, b = img[0], img[1], img[2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = 0.5 - 0.168736 * r - 0.331264 * g + 0.5 * b
    cr = 0.5 + 0.5 * r - 0.418588 * g - 0.081312 * b
    return np.array([y, cb, cr])


def ycbcr2rgb(img):
    # out = color.ycbcr2rgb( 256.*img.transpose(1, 2, 0) )
    # return (out.transpose(2,0,1) - np.min(out))/(np.max(out)-np.min(out))
    y, cb, cr = img[0], img[1], img[2]
    r = y + 1.402 * (cr - 0.5)
    g = y - 0.344136 * (cb - 0.5) - 0.714136 * (cr - 0.5)
    b = y + 1.772 * (cb - 0.5)
    return np.array([r, g, b])


def mse(x_hat, x_true, maxv=1.):
    x_hat = x_hat.flatten()
    x_true = x_true.flatten()
    mse = np.mean(np.square(x_hat - x_true))
    energy = np.mean(np.square(x_true))
    return mse / energy


def psnr(x_hat, x_true, maxv=1.):
    x_hat = x_hat.flatten()
    x_true = x_true.flatten()
    mse = np.mean(np.square(x_hat - x_true))
    psnr_ = 10. * np.log(maxv ** 2 / mse) / np.log(10.)
    return psnr_


def num_param(net):
    s = sum([np.prod(list(p.size())) for p in net.parameters()]);
    return s
    # print('Number of params: %d' % s)


def rgb2gray(rgb):
    r, g, b = rgb[0, :, :], rgb[1, :, :], rgb[2, :, :]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return np.array([gray])


# 为对数绘图保存数据
def savemtx_for_logplot(A, filename="exp.dat"):
    # 在对数空间选择索引点
    ind = sorted(list(set([int(i) for i in np.geomspace(1, len(A[0]) - 1, num=700)])))
    A = [[a[i] for i in ind] for a in A]
    X = np.array([ind] + A)
    np.savetxt(filename, X.T, delimiter=' ')


# 参数统计
def count_model_parameters(model):
    total_params = 0
    effective_params = 0
    
    # 记录已经处理过的参数对象 ID，防止在复杂网络结构中重复计数
    seen_params = set()

    # 遍历所有子模块
    for name, module in model.named_modules():
        # 如果是 DirectionalConv2d，执行特殊逻辑
        if isinstance(module, DirectionalConv2d):
            # 处理卷积权重 (weight)
            p = module.conv.weight
            if id(p) not in seen_params:
                p_total = p.numel()
                # 有效参数 = 激活的 mask 点数 * 输入通道 * 输出通道
                p_effective = module.effective_param_count()
                
                total_params += p_total
                effective_params += p_effective
                seen_params.add(id(p))
            
            # 处理卷积偏置 (bias) bias=False
            if module.conv.bias is not None:
                b = module.conv.bias
                if id(b) not in seen_params:
                    total_params += b.numel()
                    effective_params += b.numel()
                    seen_params.add(id(b))
                    
        # 对于非 DirectionalConv2d 的叶子节点模块（如普通的 Conv2d, Linear, BatchNorm）
        # 我们只处理那些不再包含子模块的层，防止重复统计
        elif len(list(module.children())) == 0:
            for p_name, p in module.named_parameters(recurse=False):
                if id(p) not in seen_params:
                    total_params += p.numel()
                    effective_params += p.numel()
                    seen_params.add(id(p))

    print("-" * 30)
    print(f"总注册参数量 (Total):     {total_params:,}")
    print(f"有效计算参数量 (Effective): {effective_params:,}")
    print(f"减少的冗余参数:           {total_params - effective_params:,}")
    print("-" * 30)
    
    return total_params, effective_params
