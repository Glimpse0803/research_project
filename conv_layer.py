import torch
import torch.nn as nn
import torch.nn.functional as F  
from mire_config import CONFIG

# 4支路并行
class DirectionalConv2d(nn.Module):
    def __init__(self, in_f, out_f, direction, dilation=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_f, out_f,
            kernel_size=3,
            stride=1,
            padding='same',
            dilation=dilation,
            bias=False
        )
        if direction == '45':
            mask = torch.tensor([
                [0, 0, 1],
                [0, 1, 0],
                [1, 0, 0]
            ])
        elif direction == '135':
            mask = torch.tensor([
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1]
            ])
        elif direction == '0':
            mask = torch.tensor([
                [1, 0, 1],
                [0, 1, 0],
                [1, 0, 1]
            ])


        self.register_buffer(
            'mask',
            mask.view(1, 1, 3, 3)
        )

    def forward(self, x):
        weight = self.conv.weight * self.mask
        return F.conv2d(
            x,
            weight,
            self.conv.bias,
            stride=self.conv.stride,
            padding=self.conv.padding,
            dilation=self.conv.dilation
        )

    def effective_param_count(self):
        # mask: (1,1,3,3)
        active = int(self.mask.sum().item())
        return active * self.conv.in_channels * self.conv.out_channels

def asy_dir_conv_block(in_f, out_f, channel_list, direction1, direction2, dilation=1):
    '''
    参数:
        in_f: 输入通道数
        out_f: 输出通道数
        channel_list: 各方向比例
        direction: 卷积方向
    '''
    layers = nn.ModuleDict()

    # 支路1: 45
    branch1_layers = []
    # branch1_layers.append(nn.ReflectionPad2d((1, 1, 1, 1)))
    branch1_layers.append(DirectionalConv2d(in_f, channel_list[0], direction1, dilation=1))
    # branch1_layers.append(nn.BatchNorm2d(channel_list[0], affine=bn_affine))
    branch1_layers.append(nn.BatchNorm2d(channel_list[0]))
    branch1_layers.append(nn.ReLU(inplace=True))
    layers['branch1'] = nn.Sequential(*branch1_layers)

    # 支路2: 135
    branch2_layers = []
    # branch2_layers.append(nn.ReflectionPad2d((1, 1, 1, 1)))
    branch2_layers.append(DirectionalConv2d(in_f, channel_list[1], direction2, dilation=1))
    # branch2_layers.append(nn.BatchNorm2d(channel_list[1], affine=bn_affine))
    branch2_layers.append(nn.BatchNorm2d(channel_list[1]))
    branch2_layers.append(nn.ReLU(inplace=True))
    layers['branch2'] = nn.Sequential(*branch2_layers)

    # 支路3: 0
    branch3_layers = []
    branch3_layers.append(nn.Conv2d(in_f, channel_list[2],
                                    kernel_size=(1,3),
                                    stride=1,
                                    padding='same',
                                    dilation=(1, dilation),
                                    bias=False))

    # branch3_layers.append(nn.BatchNorm2d(channel_list[2], affine=bn_affine))
    branch3_layers.append(nn.BatchNorm2d(channel_list[2]))
    branch3_layers.append(nn.ReLU(inplace=True))
    layers['branch3'] = nn.Sequential(*branch3_layers)

    # 支路4: 90
    branch4_layers = []
    branch4_layers.append(nn.Conv2d(in_f, channel_list[3],
                                    kernel_size=(3,1),
                                    stride=1,
                                    padding='same',
                                    dilation=(dilation, 1),
                                    bias=False))
    # branch4_layers.append(nn.BatchNorm2d(channel_list[3], affine=bn_affine))
    branch4_layers.append(nn.BatchNorm2d(channel_list[3]))
    branch4_layers.append(nn.ReLU(inplace=True))
    layers['branch4'] = nn.Sequential(*branch4_layers)

    class DualDirectionalConv(nn.Module):
        def __init__(self, layers, out_f):
            super(DualDirectionalConv, self).__init__()
            # 并行五支路层
            self.branch1 = layers['branch1']
            self.branch2 = layers['branch2']
            self.branch3 = layers['branch3']
            self.branch4 = layers['branch4']
            # Conv(1×1)层
            self.fusion_conv = nn.Conv2d(CONFIG['branch_channels'], out_f,
                                        kernel_size=(1, 1),
                                        stride=1,
                                        padding=0,
                                        bias=False)
        def forward(self, x):
            # 并行支路
            out1 = self.branch1(x)
            out2 = self.branch2(x)
            out3 = self.branch3(x)
            out4 = self.branch4(x)
            # concat连接，通道数变branch_channels
            concat_out = torch.cat([out1, out2, out3, out4], dim=1)
            # Conv(1×1)对支路特征进行fusion
            fused_out = self.fusion_conv(concat_out)
            return fused_out

    return DualDirectionalConv(layers, out_f)
