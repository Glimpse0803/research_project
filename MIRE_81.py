import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import os
import matplotlib.pyplot as plt
from PIL import Image
from collections import Counter
import time

from torch.amp import autocast

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
    convolver = nn.Conv2d(in_f, out_f, kernel_size, stride, padding='same', bias=False, dilation=dilation)
    torch.nn.init.kaiming_normal_(convolver.weight, mode='fan_in', nonlinearity='relu')
    return nn.Sequential(convolver)


# ==========================================
# 1. ViewInputOffset (视角偏移层，保持不变)
# ==========================================
class ViewInputOffset(nn.Module):
    def __init__(self, channels=64, u_views=9, v_views=9, view_kernel_w=3, view_kernel_h=3):
        super().__init__()
        self.u_offsets = nn.ParameterList([
            nn.Parameter(torch.randn(1, channels, view_kernel_w, view_kernel_h) * 0.01) for _ in range(u_views)
        ])
        self.v_offsets = nn.ParameterList([
            nn.Parameter(torch.randn(1, channels, view_kernel_w, view_kernel_h) * 0.01) for _ in range(v_views)
        ])

    def forward(self, base_input, u_idx, v_idx):
        offset = self.u_offsets[u_idx] + self.v_offsets[v_idx]
        return base_input + F.interpolate(offset, size=(32, 32), mode='bicubic', align_corners=False)


# ==========================================
# 2. HybridBlock (接受固定激活函数实例)
# ==========================================
class HybridBlock(nn.Module):
    def __init__(self, in_channels, out_channels=64, upsample=True, dilation=1, act_instance=None):
        super().__init__()
        self.desc_conv = conv_layer(in_channels, out_channels, 3, dilation)
        self.desc_bn = nn.BatchNorm2d(out_channels)
        # 核心: 使用传入的固定激活函数，默认 GELU
        self.desc_act = act_instance if act_instance is not None else nn.GELU()
        self.upsample = upsample
        if upsample:
            self.up_layer = nn.Upsample(scale_factor=2, mode='bicubic', align_corners=False)

    def forward(self, x):
        out_put = self.desc_conv(x)
        if self.upsample:
            out_put = self.up_layer(out_put)
        out_put = self.desc_bn(out_put)
        out_put = self.desc_act(out_put)
        return out_put


# ==========================================
# 3. Decoders
# ==========================================
class GrowingDecoder(nn.Module):
    def __init__(self, view_offset_layer, fixed_blocks, candidate_block, out_channels=3, target_size=(512, 512)):
        super().__init__()
        self.view_offset_layer = view_offset_layer
        self.fixed_blocks = nn.ModuleList(fixed_blocks)
        self.candidate_block = candidate_block
        self.target_size = target_size
        curr_ch = CONFIG['k_channels']
        self.adapter = nn.Sequential(
            conv_layer(curr_ch, out_channels, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, net_input, u_idx, v_idx):
        out = self.view_offset_layer(net_input, u_idx, v_idx)
        for blk in self.fixed_blocks:
            out = blk(out)
        out = self.candidate_block(out)
        out = self.adapter(out)
        if out.shape[2:] != self.target_size:
            out = F.interpolate(out, size=self.target_size, mode='bicubic', align_corners=False)
        return out


class FinalDecoder(nn.Module):
    def __init__(self, view_offset_layer, blocks, last_channels, out_channels=3):
        super().__init__()
        self.view_offset_layer = view_offset_layer
        self.main_body = nn.ModuleList(blocks)
        self.output_head = nn.Sequential(
            conv_layer(last_channels, out_channels, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, net_input, u_idx, v_idx):
        out = self.view_offset_layer(net_input, u_idx, v_idx)
        for blk in self.main_body:
            out = blk(out)
        return self.output_head(out)


# ==========================================
# 4. Data Loading
# ==========================================
def load_lf_images(base_path, h=9, w=9):
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
                if img_np.shape[1] > 512:
                    img_np = img_np[:, :512, :512]
                images[(u, v)] = np_to_var(img_np).type(dtype)
                count += 1
    print(f"Loaded {count} images to VRAM dictionary.")
    return images


def plot_learning_curve(loss_hist, psnr_hist, save_path):
    steps = np.arange(len(loss_hist)) * 50
    fig, ax1 = plt.subplots(figsize=(9, 6))

    color_loss = 'tab:red'
    ax1.set_xlabel('Steps')
    ax1.set_ylabel('MSE Loss (Log Scale)', color=color_loss, fontsize=12)
    ax1.plot(steps, loss_hist, color=color_loss, alpha=0.3, label='Raw Loss')
    if len(loss_hist) > 5:
        smooth_loss = np.convolve(loss_hist, np.ones(5) / 5, mode='valid')
        ax1.plot(steps[2:-2], smooth_loss, color=color_loss, linewidth=2, label='Smooth Loss')
    ax1.tick_params(axis='y', labelcolor=color_loss)
    ax1.set_yscale('log')
    ax1.grid(True, which="both", ls="-", alpha=0.2)

    ax2 = ax1.twinx()
    color_psnr = 'tab:blue'
    ax2.set_ylabel('PSNR (dB)', color=color_psnr, fontsize=12)
    ax2.plot(steps, psnr_hist, color=color_psnr, linewidth=2, label='Avg PSNR')
    ax2.tick_params(axis='y', labelcolor=color_psnr)

    plt.title('Phase 2 Learning Curve (MIRE)', fontsize=14)
    fig.tight_layout()
    handler1, label1 = ax1.get_legend_handles_labels()
    handler2, label2 = ax2.get_legend_handles_labels()
    ax1.legend(handler1 + handler2, label1 + label2, loc='upper left')
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Successfully saved learning curve to {save_path}")


# ==========================================
# 5. Main: MIRE 逐层生长搜索 (多视角版)
# ==========================================
def main_final_optimized():
    scene_name = "boxes"
    data_path = "data/boxes"
    save_dir = f'outputs/MIRE_{scene_name}'
    os.makedirs(save_dir, exist_ok=True)

    k_channels = CONFIG['k_channels']
    # balanced_schedule = [(600, 2400), (800, 2200), (600, 2400), (1000, 2500), (800, 3200)]
    # balanced_schedule = [(1500, 3000), (1500, 3000), (2000, 2500), (2000, 2500), (2500, 2000)]
    search_schedule = [3000, 3000, 3500, 4000, 4500]
    upsample_configs = [True, True, True, True, False]
    dilations = [1, 2, 2, 2, 1]

    BATCH_SIZE = 81
    lf_data = load_lf_images(data_path)
    view_keys = list(lf_data.keys())
    TOTAL_VIEWS = len(view_keys)

    first_target = lf_data[view_keys[0]]
    target_h, target_w = first_target.shape[2], first_target.shape[3]
    print(f"Total Views: {TOTAL_VIEWS} | Size: {target_h}x{target_w}")

    noise_file = CONFIG['noise_file']
    if os.path.exists(noise_file):
        net_input = torch.load(noise_file).type(dtype)
    else:
        net_input = torch.zeros(1, k_channels, 32, 32).type(dtype)
        net_input.data.uniform_() * 0.1
    net_input_saved = net_input.data.clone()

    fixed_blocks = []
    best_act_names = []
    mse = nn.MSELoss()
    view_offset_layer = ViewInputOffset(
        channels=CONFIG['k_channels'], u_views=9, v_views=9,
        view_kernel_w=CONFIG['view_kernel_w'],
        view_kernel_h=CONFIG['view_kernel_h']
    ).type(dtype)

    # ================================================================
    # Phase 1: MIRE 逐层生长搜索
    #   仿照单图 mire_search.py 的逻辑:
    #   - 冻结已确定层
    #   - 遍历 ACT_DICT，每个激活函数训练 search_epochs 步
    #   - 按 PSNR 选 winner
    # ================================================================
    print(f"\n{'=' * 20} Phase 1: MIRE Layer-wise Growing Search {'=' * 20}")
    print(f"Per-layer budgets (search): {search_schedule} | "
          f"Num activations: {len(ACT_DICT)} | LR: {CONFIG['search_lr']}")

    num_acts = len(ACT_DICT)
    inherited_adapter_state = None  # 跨层继承的 adapter 权重 (第1层为 None，随机初始化)

    for layer_idx, (upsample, dilation, n_search) in enumerate(
            zip(upsample_configs, dilations, search_schedule)):
        search_total = n_search * num_acts
        print(f"\n{'─' * 60}")
        print(f"[Layer {layer_idx + 1}/5] upsample={upsample}, dilation={dilation}")
        print(f"  Search: {n_search} steps × {num_acts} acts = {search_total} steps")
        print(f"{'─' * 60}")

        performance_log = []
        layer_best_psnr = -1.0
        layer_best_act = None
        layer_best_block_state = None
        layer_best_adapter_state = None
        layer_best_view_offset_state = None
        layer_best_fixed_blocks_state = None     # 保存 winner 时旧层的训练状态

        # --- 解冻已确定的层，允许旧层参与训练  ---
        for blk in fixed_blocks:
            for param in blk.parameters():
                param.requires_grad = True
            blk.desc_bn.train()

        # --- 在每层搜索开始前，保存所有共享状态的快照 ---
        # 确保每个候选激活函数从相同起点开始，保证搜索公平
        view_offset_snapshot = copy.deepcopy(view_offset_layer.state_dict())
        fixed_blocks_snapshot = [copy.deepcopy(blk.state_dict()) for blk in fixed_blocks]
        adapter_snapshot = copy.deepcopy(inherited_adapter_state)  # 可能是 None

        # ===== Search: 遍历激活函数，每个训练 n_search 步 =====
        search_curves = {}  # {act_name: [(step, loss), ...]} 用于绘制收敛曲线
        LOG_INTERVAL = 10  # 每 10 步记录一次

        # --- 遍历激活函数字典 ---
        for act_name in ACT_DICT:
            # 每个候选者从相同的起点开始
            view_offset_layer.load_state_dict(copy.deepcopy(view_offset_snapshot))
            for blk, snap in zip(fixed_blocks, fixed_blocks_snapshot):
                blk.load_state_dict(copy.deepcopy(snap))

            # 创建候选层
            cand = HybridBlock(k_channels, k_channels, upsample=upsample,
                               dilation=dilation,
                               act_instance=get_activation_instance(act_name))

            # 构建生长解码器
            net = GrowingDecoder(view_offset_layer, fixed_blocks, cand, 3, (target_h, target_w)).type(dtype)

            # 继承上一层 winner 的 adapter 权重，避免每次从随机开始
            if inherited_adapter_state is not None:
                net.adapter.load_state_dict(copy.deepcopy(inherited_adapter_state))

            optimizer = torch.optim.Adam(net.parameters(), lr=CONFIG['search_lr'])
            net.train()

            curve = []  # 记录当前激活函数的 loss 曲线

            # 搜索训练
            for i in range(n_search):
                # 输入噪声正则化
                if CONFIG['reg_noise_std'] > 0:
                    cur_std = CONFIG['reg_noise_std'] if i % CONFIG['reg_noise_decay'] != 0 else 0
                    ni = net_input_saved + (torch.randn_like(net_input_saved) * cur_std)
                else:
                    ni = net_input_saved

                batch_indices = torch.randperm(TOTAL_VIEWS)[:BATCH_SIZE]


                for idx in batch_indices:
                    u, v = view_keys[idx]
                    target = lf_data[(u, v)]
                    optimizer.zero_grad()
                    with autocast('cuda', dtype=torch.bfloat16):
                        out = net(ni, u, v)
                        loss = mse(out, target) # / BATCH_SIZE
                    loss.backward()
                    optimizer.step()

                # 记录收敛曲线
                if i % LOG_INTERVAL == 0:
                    curve.append((i, loss.item()))

                if i % 200 == 0:
                    print(f"      [{act_name}] Search Step {i}/{n_search} - Loss: {loss.item():.6f}")

            search_curves[act_name] = curve

            # 评估
            with torch.no_grad():
                psnr_list = []
                for (u, v) in view_keys:
                    target = lf_data[(u, v)]
                    with autocast('cuda', dtype=torch.bfloat16):
                        out = net(net_input_saved, u, v)
                    p = psnr(out.float().cpu().numpy()[0], target.cpu().numpy()[0])
                    psnr_list.append(p)
                avg_psnr = np.mean(psnr_list)

            performance_log.append((act_name, loss.item(), avg_psnr))
            print(f"      [{act_name}] Search Done → PSNR: {avg_psnr:.2f} dB")

            # 选 winner
            if avg_psnr > layer_best_psnr:
                layer_best_psnr = avg_psnr
                layer_best_act = act_name
                layer_best_block_state = copy.deepcopy(net.candidate_block)
                layer_best_adapter_state = copy.deepcopy(net.adapter.state_dict())
                layer_best_view_offset_state = copy.deepcopy(view_offset_layer.state_dict())
                layer_best_fixed_blocks_state = [copy.deepcopy(blk.state_dict()) for blk in fixed_blocks]

        # 排行榜
        print(f"\n  [Layer {layer_idx + 1} Leaderboard (by PSNR)]")
        print(f"  {'Act Name':<12} | {'Loss':<12} | {'PSNR (dB)':<10}")
        print(f"  {'─' * 45}")
        performance_log.sort(key=lambda x: x[2], reverse=True)
        for rank, (name, loss_val, psnr_val) in enumerate(performance_log):
            marker = " <<< WINNER" if rank == 0 else ""
            print(f"  {name:<12} | {loss_val:.8f} | {psnr_val:.2f}{marker}")

        best_act_names.append(layer_best_act)

        # ===== 绘制搜索收敛曲线 =====
        fig, ax = plt.subplots(figsize=(10, 6))
        for act_name, curve in search_curves.items():
            steps_arr = [c[0] for c in curve]
            loss_arr = [c[1] for c in curve]
            linewidth = 2.5 if act_name == layer_best_act else 1.2
            alpha = 1.0 if act_name == layer_best_act else 0.6
            label = f"{act_name} ★" if act_name == layer_best_act else act_name
            ax.plot(steps_arr, loss_arr, linewidth=linewidth, alpha=alpha, label=label)
        ax.set_xlabel('Search Steps', fontsize=12)
        ax.set_ylabel('MSE Loss', fontsize=12)
        ax.set_yscale('log')
        ax.set_title(f'Layer {layer_idx + 1} Search Convergence ({n_search} steps/act)', fontsize=13)
        ax.legend(fontsize=9, loc='upper right')
        ax.grid(True, alpha=0.3)
        curve_path = os.path.join(save_dir, f"layer{layer_idx + 1}_search_curves.png")
        plt.tight_layout()
        plt.savefig(curve_path, dpi=200)
        plt.close()
        print(f"  > Search curves saved: {curve_path}")

        # ===== 移除原本冗余的 Refine 训练，只保留 Winner 状态恢复 =====
        print(f"\n  [Layer {layer_idx + 1}] Restoring state to Winner: {layer_best_act}...")

        # 1. 恢复之前保存的最佳层、View Offset 和固定的旧层
        fixed_blocks.append(layer_best_block_state)
        view_offset_layer.load_state_dict(layer_best_view_offset_state)
        for blk, snap in zip(fixed_blocks[:-1], layer_best_fixed_blocks_state):
            blk.load_state_dict(snap)

        # 2. 继承 Adapter 权重，给下一层 Search 用
        inherited_adapter_state = copy.deepcopy(layer_best_adapter_state)

        # 3. 【关键修复】使用 Winner 重建网络，确保外部的 `net` 变量指向正确的模型架构
        final_candidate = fixed_blocks.pop()
        net = GrowingDecoder(view_offset_layer, fixed_blocks, final_candidate, 3, (target_h, target_w)).type(dtype)
        fixed_blocks.append(final_candidate)
        net.adapter.load_state_dict(inherited_adapter_state)

        # 4. 打印结果
        print(f"  > Layer {layer_idx + 1} Winner: {layer_best_act} | Final PSNR: {layer_best_psnr:.2f} dB")

    print(f"\n>>> MIRE Search Complete: {best_act_names}")

    # ================================================================
    # Phase 2: Global Finetuning
    # ================================================================
    print(f"\n{'=' * 20} Phase 2: Global Finetuning (FP32) {'=' * 20}")

    for p in net.parameters():
        p.requires_grad = True

    net.train()

    total_params = sum(p.numel() for p in net.parameters())
    trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"  > Total params: {total_params:,}")
    print(f"  > Trainable params: {trainable_params:,}")
    print(f"  > Best Act per Layer: {best_act_names}")

    main_lr = 0.005
    optimizer = torch.optim.Adam(net.parameters(), lr=main_lr)

    t_start = time.time()
    TOTAL_STEPS = 500 * 5 * 10
    best_avg_psnr = 0.0
    best_state = None
    history_psnr = []
    history_loss = []
    lf_order = spiral_order(9)

    print(f"  > Training for {TOTAL_STEPS} steps...")

    for step in range(TOTAL_STEPS):
        if step in [500 * 5 * 3, 500 * 5 * 6, 500 * 5 * 9]:
            for pg in optimizer.param_groups:
                pg['lr'] *= 0.5
            print(f"    LR Decay at step {step}")

        epoch_loss = 0.0
        for [u_org, v_org] in lf_order:
            u, v = u_org - 1, v_org - 1
            target = lf_data[(u, v)]

            optimizer.zero_grad()
            with autocast('cuda', dtype=torch.bfloat16):
                out = net(net_input_saved, u, v)
                loss = mse(out, target)
            loss.backward()
            epoch_loss += loss.item()
            torch.nn.utils.clip_grad_value_(net.parameters(), clip_value=0.5)
            optimizer.step()

        avg_step_loss = epoch_loss / len(lf_order)
        if step % 50 == 0:
            elapsed = time.time() - t_start
            history_loss.append(avg_step_loss)
            with torch.no_grad():
                psnr_list = []
                for (u, v) in view_keys:
                    target = lf_data[(u, v)]
                    with autocast('cuda', dtype=torch.bfloat16):
                        out = net(net_input_saved, u, v)
                    p = psnr(out.float().cpu().numpy()[0], target.cpu().numpy()[0])
                    psnr_list.append(p)
                avg_psnr = np.mean(psnr_list)
                history_psnr.append(avg_psnr)
                print(f"  Step {step}/{TOTAL_STEPS} - Avg PSNR: {avg_psnr:.2f} dB | Time: {elapsed:.0f}s")
                if avg_psnr > best_avg_psnr:
                    best_avg_psnr = avg_psnr
                    best_state = copy.deepcopy(net.state_dict())

    scene_label = "boxes"
    plot_path = os.path.join(save_dir, f"{scene_label}_mire_learning_curve.pdf")
    plot_learning_curve(history_loss, history_psnr, plot_path)
    print(f"\nTraining Done. Best Avg PSNR: {best_avg_psnr:.2f} dB")
    print(f"Selected Activations: {best_act_names}")


if __name__ == "__main__":
    seed_everything(42)
    main_final_optimized()