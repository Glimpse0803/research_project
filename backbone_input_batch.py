import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import os
import matplotlib.pyplot as plt
from PIL import Image
from collections import Counter

from torch.amp import autocast  # 新增，用于半精度训练

# 引入辅助模块
from helpers import np_to_var, pil_to_np, psnr, count_model_parameters
from mire_config import CONFIG
from activations import ACT_DICT, get_activation_instance


###############################################################################
################### 参考性能（可训练参数量 -- PSNR）, 81张图，boxes ###############
# 103704 -- 36.08dB
# 164049 -- 37.87dB
# 237894 -- 38.50dB
# 325239 -- 39.60dB
# 540429 -- 40.57dB
# 809619 -- 41.38dB
# 1132809 -- 41.79dB
###############################################################################

dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

def spiral_order(angular_resolution):
    position_list = []
    left, right, up, down = 1, angular_resolution, 1, angular_resolution
    while left < right and up < down:
        for x in range(left, right):
            position_list.append([up, x])
        for y in range(up, down):
            position_list.append([y, right])
        for x in range(right, left, -1):
            position_list.append([down, x])
        for y in range(down, up, -1):
            position_list.append([y, left])
        left += 1
        right -= 1
        up += 1
        down -= 1
    if angular_resolution % 2:
        position_list.append([angular_resolution // 2 + 1, angular_resolution // 2 + 1])
    return position_list

# ==========================================
# 0. 基础组件
# ==========================================
def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def conv_layer(in_f, out_f, kernel_size, dilation, stride=1, pad='reflection'):
    padder = None
    to_pad = int((kernel_size - 1) / 2)
    if pad == 'reflection':
        padder = nn.ReflectionPad2d(to_pad)
        to_pad = 0
    convolver = nn.Conv2d(in_f, out_f, kernel_size, stride, padding='same', bias=False, dilation=dilation)   # Jinglei: 1. 这里padding直接用“same"即可   5.这里添加了dilation，分别为[1,2,2,2,1]
    torch.nn.init.kaiming_normal_(convolver.weight, mode='fan_in', nonlinearity='relu')
    # layers = filter(lambda x: x is not None, [padder, convolver])
    layers = filter(lambda x: x is not None, [convolver])
    return nn.Sequential(*layers)


# ==========================================
# 2. Hybrid Block
# ==========================================


class ViewInputOffset(nn.Module):
    def __init__(self, channels=64, u_views=9, v_views=9, view_kernel_w=3, view_kernel_h=3):
        super().__init__()

        self.channels = channels
        self.view_kernel_w = view_kernel_w
        self.view_kernel_h = view_kernel_h

        # 用 Embedding 替代 ParameterList
        self.u_embed = nn.Embedding(u_views, channels * view_kernel_w * view_kernel_h)
        self.v_embed = nn.Embedding(v_views, channels * view_kernel_w * view_kernel_h)

        nn.init.normal_(self.u_embed.weight, std=0.01)
        nn.init.normal_(self.v_embed.weight, std=0.01)

    def forward(self, base_input, u_batch, v_batch):
        # 如果传入的是 int，说明是单图推断，将其转为 Tensor
        if isinstance(u_batch, int):
            u_batch = torch.tensor([u_batch], device=base_input.device)
        if isinstance(v_batch, int):
            v_batch = torch.tensor([v_batch], device=base_input.device)

        B = u_batch.shape[0]

        # [B, C*k*k]
        u_offset = self.u_embed(u_batch)
        v_offset = self.v_embed(v_batch)

        offset = u_offset + v_offset

        # reshape → [B, C, k, k]
        offset = offset.view(B, self.channels, self.view_kernel_w, self.view_kernel_h)

        # 上采样
        offset = F.interpolate(offset, size=(32, 32), mode='bicubic')

        base_input = base_input.expand(B, -1, -1, -1)

        return base_input + offset

class HybridBlock(nn.Module):
    def __init__(self, in_channels, out_channels=64, upsample=True, dilation=1):
        super().__init__()

        self.desc_conv = conv_layer(in_channels, out_channels, 3, dilation)
        self.desc_bn = nn.BatchNorm2d(out_channels) #nn.BatchNorm2d(self.desc_ch, affine=True)
        # self.batch_norm = nn.BatchNorm2d(out_channels)
        # 依然使用 SA
        self.desc_act = nn.GELU() #ImprovedSelfAttentionAct(channels=self.desc_ch)

        self.upsample = upsample
        if upsample:
            self.up_layer = nn.Upsample(scale_factor=2, mode='bicubic', align_corners=False)    # 4. Jinglei: upsampling方法用bicubic，更加准确

    # conv -> up -> bn -> act(GELU)
    def forward(self, x):
        # up batch act
        out_put = self.desc_conv(x)
        if self.upsample: out_put = self.up_layer(out_put)
        out_put = self.desc_bn(out_put)                     # 3. Jinglei: 这里，我简单使用了 concat - bn - act的形式，后面我们根据需要调整
        out_put = self.desc_act(out_put)
        return out_put


# ==========================================
# 3. Decoders
# ==========================================
class GrowingDecoder(nn.Module):
    def __init__(self, view_offset_layer, fixed_blocks, candidate_block, out_channels=3, target_size=(512, 512)):
        super().__init__()
        self.view_offset_layer = view_offset_layer # 18个 Tensor 仓库
        self.fixed_blocks = nn.ModuleList(fixed_blocks)
        self.candidate_block = candidate_block
        self.target_size = target_size
        curr_ch = CONFIG['k_channels']
        self.adapter = nn.Sequential(
            conv_layer(curr_ch, out_channels, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, net_input, u_batch, v_batch):
        # 融合视角 Tensor
        out = self.view_offset_layer(net_input, u_batch, v_batch)
        for blk in self.fixed_blocks:
            out = blk(out)
        out = self.candidate_block(out)
        out = self.adapter(out)
        if out.shape[2:] != self.target_size:
            out = F.interpolate(out, size=self.target_size, mode='bicubic', align_corners=False)  # 4. Jinglei: upsampling方法用bicubic，更加准确
        return out


class FinalDecoder(nn.Module):
    def __init__(self, view_offset_layer, blocks, last_channels, out_channels=3):
        super().__init__()
        self.view_offset_layer = view_offset_layer # 18个 Tensor 仓库
        self.main_body = nn.ModuleList(blocks)
        self.output_head = nn.Sequential(
            conv_layer(last_channels, out_channels, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, net_input, u_idx, v_idx):
        # 融合视角 Tensor
        out = self.view_offset_layer(net_input, u_idx, v_idx)

        for blk in self.main_body:
            out = blk(out)
        return self.output_head(out)


# ==========================================
# 4. Data Loading (原版)
# ==========================================
def load_lf_images(base_path, h=9, w=9):
    """
    原版加载函数：读取图片 -> Tensor -> 放入字典 -> 驻留显存
    """
    images = {}
    print(f"Loading Light Field from {base_path}...")
    count = 0
    for u in range(h):
        for v in range(w):
            fname = f"lf_{u + 1}_{v + 1}.png"
            path = os.path.join(base_path, fname)
            if not os.path.exists(path):
                fname = f"boxes_{u}_{v}.png"
                path = os.path.join(base_path, fname)

            if os.path.exists(path):
                img = Image.open(path).convert('RGB')
                img_np = pil_to_np(img)
                if img_np.shape[1] > 512: img_np = img_np[:, :512, :512]
                # 转为 CUDA Tensor 并存入字典
                images[(u, v)] = np_to_var(img_np).type(dtype)
                count += 1
    print(f"Loaded {count} images to VRAM dictionary.")
    return images


# ==============================
# 3. Batch构造
# ==============================
def build_batch(batch_keys, lf_data, device):
    u_batch, v_batch, target_batch = [], [], []

    for (u, v) in batch_keys:
        u_batch.append(u)
        v_batch.append(v)
        target_batch.append(lf_data[(u, v)])

    u_batch = torch.tensor(u_batch, device=device)
    v_batch = torch.tensor(v_batch, device=device)
    target_batch = torch.cat(target_batch, dim=0)

    return u_batch, v_batch, target_batch

# ==========================================
# 5. Main Procedure (Batch 20 + SA + Full Unfreeze)
# ==========================================
def main_final_optimized():
    scene_name = "boxes"
    data_path = "data/boxes"
    save_dir = f'outputs/sa_batch20_fullunfreeze_{scene_name}'
    os.makedirs(save_dir, exist_ok=True)

    k_channels = CONFIG['k_channels']
    balanced_schedule = [(600, 2400), (800, 2200), (600, 2400), (1000, 2500), (800, 3200)]
    # balanced_schedule = [(10, 20), (10, 20), (10, 20), (10, 20), (10, 20)]                      # Jinglei: 这里需要调整回去
    upsample_configs = [True, True, True, True, False]
    dilations = [1,2,2,2,1]                                                                     # Jinglei: 这里是我设置的各层dilation

    # === [设置 Batch Size] ===
    BATCH_SIZE = 27

    # 加载数据 (字典模式)
    lf_data = load_lf_images(data_path)
    view_keys = list(lf_data.keys())  # 所有的 (u, v)
    TOTAL_VIEWS = len(view_keys)
    # 修改取图逻辑：将所有 key 转换为列表，保证顺序固定 (不使用 spiral_order)
    sequential_keys = list(lf_data.keys())

    # 获取目标尺寸
    first_target = lf_data[view_keys[0]]
    target_h, target_w = first_target.shape[2], first_target.shape[3]

    print(f"Total Views: {TOTAL_VIEWS}, Training Batch Size: {BATCH_SIZE}")

    noise_file = CONFIG['noise_file']
    if os.path.exists(noise_file):
        net_input = torch.load(noise_file).type(dtype)
    else:
        net_input = torch.zeros(1, k_channels, 32, 32).type(dtype)
        net_input.data.uniform_() * 0.1
    net_input_saved = net_input.data.clone()

    fixed_blocks = []
    last_head = None
    mse = nn.MSELoss()
    view_offset_layer = ViewInputOffset(channels=CONFIG['k_channels'], u_views=9, v_views=9, view_kernel_w=CONFIG['view_kernel_w'], view_kernel_h=CONFIG['view_kernel_h']).type(dtype)
    # 初始优化器1：只包含 18 个视角 Tensor
    # optimizer_view = torch.optim.Adam([{'params': view_offset_layer.parameters(), 'lr': CONFIG['search_lr']}], betas=(0.9, 0.99), eps=1e-14)
    
    # ---------------- Phase 1: Growing ----------------
    print(f"\n{'=' * 20} Phase 1: Layer-wise Growing (SA + Batch 20) {'=' * 20}")

    for layer_idx, (upsample, (n_search, n_refine),dilation) in enumerate(zip(upsample_configs, balanced_schedule, dilations)):
        print(f"\n[Layer {layer_idx + 1}] Search: {n_search} | Refine: {n_refine}")

        for blk in fixed_blocks:
            for p in blk.parameters(): p.requires_grad = True
            blk.desc_bn.train()
            # blk.batch_norm.train()

        cand = HybridBlock(k_channels, k_channels, upsample=upsample, dilation=dilation)
        net = GrowingDecoder(view_offset_layer, fixed_blocks, cand, 3, (target_h, target_w)).type(dtype)

        optimizer = torch.optim.Adam(net.parameters(), lr=CONFIG['search_lr'])
        net.train()

        # === Phase 1.1 Search ===
        for i in range(n_search):
            if CONFIG['reg_noise_std'] > 0:
                cur_std = CONFIG['reg_noise_std'] if i % 500 else 0
                ni = net_input_saved + (torch.randn_like(net_input_saved) * cur_std)
            else:
                ni = net_input_saved

            # accum_loss = 0.0

            # 随机采样 BATCH_SIZE 个索引
            # batch_indices = torch.randperm(TOTAL_VIEWS)[:BATCH_SIZE]
            # batch_keys = [view_keys[idx] for idx in batch_indices]
            # u_batch, v_batch, target_batch = build_batch(batch_keys, lf_data, net_input.device)

            for i in range(0, TOTAL_VIEWS, BATCH_SIZE):
                # 顺序截取 batch_size 个样本
                batch_keys = sequential_keys[i : i + BATCH_SIZE]

                # print(batch_keys[0])

                u_batch, v_batch, target_batch = build_batch(batch_keys, lf_data, net_input.device)

                optimizer.zero_grad()

                with autocast('cuda', dtype=torch.bfloat16):
                    out = net(net_input_saved, u_batch, v_batch)
                    loss = mse(out, target_batch)

                loss.backward()
                optimizer.step()

            if i % 100 == 0:
                print(f"    Search Step {i}/{n_search} - Avg Batch Loss: {loss:.6f}")

        # === Phase 1.2 Refine ===
        for pg in optimizer.param_groups: pg['lr'] = CONFIG['search_lr'] * 0.8
        for i in range(n_refine):
            # accum_loss = 0.0

            for i in range(0, TOTAL_VIEWS, BATCH_SIZE):
                # 顺序截取 batch_size 个样本
                batch_keys = sequential_keys[i : i + BATCH_SIZE]

                # print(batch_keys[0])

                u_batch, v_batch, target_batch = build_batch(batch_keys, lf_data, net_input.device)

                optimizer.zero_grad()

                with autocast('cuda', dtype=torch.bfloat16):
                    out = net(net_input_saved, u_batch, v_batch)
                    loss = mse(out, target_batch)

                loss.backward()
                optimizer.step()

            if i % 500 == 0:
                print(f"    Refine Step {i}/{n_refine} - Avg Batch Loss: {loss:.6f}")

        fixed_blocks.append(net.candidate_block)
        last_head = copy.deepcopy(net.adapter)

        # 简单验证
        # with torch.no_grad():
        #     mid_key = view_keys[TOTAL_VIEWS // 2]
        #     out = net(net_input_saved, mid_key[0], mid_key[1])
        #     p = psnr(out.cpu().numpy()[0], lf_data[mid_key].cpu().numpy()[0])
        #     print(f"  > Layer {layer_idx + 1} Done. Mid-View PSNR: {p:.2f} dB")

        with torch.no_grad():
            psnr_list = []
            for (u, v) in view_keys:
                target = lf_data[(u, v)]
                with autocast('cuda',dtype=torch.bfloat16):  # ← 加上这个
                    out = net(net_input_saved, u, v)
                p = psnr(out.float().cpu().numpy()[0], target.cpu().numpy()[0])
                psnr_list.append(p)
            avg_psnr = np.mean(psnr_list)                                          # Jinglei: PSNR 平均可以用List来算
            print(f"  > Layer {layer_idx + 1} Done. - Est. PSNR: {avg_psnr:.2f} dB")

    # ---------------- Phase 2: Global Finetuning (Full Unfreeze) ----------------
    print(f"\n{'=' * 20} Phase 2: Global Finetuning (Full Unfreeze & Batch 20) {'=' * 20}")

    final_net = FinalDecoder(view_offset_layer, fixed_blocks, k_channels, 3).type(dtype)
    if last_head:
        final_net.output_head[0][-1].weight.data = last_head[0][-1].weight.data.clone()
        if last_head[0][-1].bias is not None:
            final_net.output_head[0][-1].bias.data = last_head[0][-1].bias.data.clone()

    # 解冻所有参数 (Full Unfreeze)
    # 不再调用 fix_descriptors，而是全部打开
    for p in final_net.parameters():
        p.requires_grad = True

    # 开启训练模式 (允许 BN 更新统计量)
    final_net.train()

    # 4. 差异化学习率
    backbone_params = []
    view_offset = []
    for blk in final_net.main_body:
        backbone_params += list(blk.desc_conv.parameters())
        backbone_params += list(blk.desc_bn.parameters())
        backbone_params += list(blk.desc_act.parameters())  # SA 的参数 (如PReLU)
    backbone_params += list(final_net.output_head.parameters())
    view_offset += list(final_net.view_offset_layer.parameters()) 
    

    main_lr = 0.005                                                                      # Jinglei: 我稍微试了大一点的学习率，可以调小为0.003
    optimizer_mod = torch.optim.Adam([
        {'params': backbone_params, 'lr': main_lr},  # Backbone 微调
        {'params': view_offset, 'lr': main_lr}],
        betas=(0.9, 0.99),                                                               # Jinglei: 这里我细化了一下，对结果影响不大
        eps=1e-14,
    )

    TOTAL_STEPS = 500*5*10 

    # 统计参数
    total, effective = count_model_parameters(final_net)

    print(f"  > Training for {TOTAL_STEPS} steps...")

    best_avg_psnr = 0.0
    best_state = None
    history_psnr = []

    # lf_order = spiral_order(9) 
    # lf_order = [(u-1, v-1) for u, v in lf_order]                                                          # Jinglei: 采用螺旋扫描方式

    # 修改取图逻辑：将所有 key 转换为列表，保证顺序固定 (不使用 spiral_order)
    # sequential_keys = list(lf_data.keys())

    for step in range(TOTAL_STEPS):
        if step in [500*5*3, 500*5*6, 500*5*9]:#[25000, 45000, 65000]:                   # Jinglei: 我把TOTAL_STEPS写成a*b*c的形式，可以调整c控制阶段
            for pg in optimizer_mod.param_groups: pg['lr'] *= 0.5
            print(f"    LR Decay at step {step}")

        for i in range(0, TOTAL_VIEWS, BATCH_SIZE):
            # batch_keys = lf_order[i:i+BATCH_SIZE]
            # 顺序截取 batch_size 个样本
            batch_keys = sequential_keys[i : i + BATCH_SIZE]

            # print(batch_keys[0])

            u_batch, v_batch, target_batch = build_batch(batch_keys, lf_data, net_input.device)

            optimizer_mod.zero_grad()

            with autocast('cuda', dtype=torch.bfloat16):
                out = final_net(net_input_saved, u_batch, v_batch)
                loss = mse(out, target_batch)

            loss.backward()
            torch.nn.utils.clip_grad_value_(final_net.parameters(), 0.5)
            optimizer_mod.step()


        if step % 2500 == 0:
            with torch.no_grad():
                psnr_list = []
                for (u, v) in view_keys:
                    target = lf_data[(u, v)]
                    with autocast('cuda',dtype=torch.bfloat16):  # ← 加上这个
                        out = final_net(net_input_saved, u, v)
                    p = psnr(out.float().cpu().numpy()[0], target.cpu().numpy()[0])
                    psnr_list.append(p)
                avg_psnr = np.mean(psnr_list)                                          # Jinglei: PSNR 平均可以用List来算
                print(f"  Step {step}/{TOTAL_STEPS} - Est. PSNR: {avg_psnr:.2f} dB")
            if avg_psnr > best_avg_psnr:
                best_avg_psnr = avg_psnr
                best_state = copy.deepcopy(final_net.state_dict())

    print(f"Training Done. Best Estimated Batch PSNR: {best_avg_psnr:.2f} dB")


if __name__ == "__main__":
    seed_everything(42)
    main_final_optimized()