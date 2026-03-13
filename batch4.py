import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import os
import matplotlib.pyplot as plt
from PIL import Image
from collections import Counter

from torch.nn import BatchNorm2d

# 引入辅助模块
from helpers import np_to_var, pil_to_np, psnr
from mire_config import CONFIG
from activations import ACT_DICT, get_activation_instance

dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor


# ==========================================
# 0. 基础组件
# ==========================================
def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def conv_layer(in_f, out_f, kernel_size, stride=1, pad='reflection'):
    padder = None
    to_pad = int((kernel_size - 1) / 2)
    if pad == 'reflection':
        padder = nn.ReflectionPad2d(to_pad)
        to_pad = 0
    convolver = nn.Conv2d(in_f, out_f, kernel_size, stride, padding=to_pad, bias=False)
    torch.nn.init.kaiming_normal_(convolver.weight, mode='fan_in', nonlinearity='relu')
    layers = filter(lambda x: x is not None, [padder, convolver])
    return nn.Sequential(*layers)


# ==========================================
# 1. Improved SA (保持不变)
# ==========================================
class ImprovedSelfAttentionAct(nn.Module):
    def __init__(self, channels, d_model=64, num_heads=4, fixed_tau=0.6, pos_emb_scale=0.5):
        super().__init__()
        self.channels = channels
        self.act_names = ACT_DICT
        self.acts = nn.ModuleList([get_activation_instance(name) for name in self.act_names])
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.chan_pos_emb = nn.Parameter(torch.randn(1, channels, d_model) * pos_emb_scale)
        self.embedding = nn.Linear(2 * len(self.act_names), d_model)
        self.norm_pre = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True)
        self.fc = nn.Linear(d_model, len(self.act_names))
        self.tau = fixed_tau
        self.is_fixed = False
        self.register_buffer('fixed_indices', torch.zeros(channels, dtype=torch.long))
        self._init_weights()

    def _init_weights(self):
        nn.init.orthogonal_(self.embedding.weight)
        nn.init.orthogonal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)

    def _get_logits(self, x):
        b, c, h, w = x.shape
        act_outputs = [act(x) for act in self.acts]
        stats_list = []
        for out in act_outputs:
            feat_avg = self.avg_pool(out).view(b, c)
            feat_std = torch.std(out, dim=(2, 3)).view(b, c)
            stats_list.append(torch.cat([feat_avg, feat_std], dim=1))
        channel_features = torch.stack(stats_list, dim=2).view(b, c, -1)
        tokens = self.embedding(channel_features) + self.chan_pos_emb
        tokens = self.norm_pre(tokens)
        attn_out, _ = self.self_attn(tokens, tokens, tokens)
        logits = self.fc(tokens + attn_out)
        return logits, act_outputs

    def forward(self, x):
        if self.is_fixed: return self._apply_hard(x)
        logits, act_outputs = self._get_logits(x)
        if self.training:
            weights = F.gumbel_softmax(logits, tau=self.tau, hard=False, dim=2)
        else:
            weights = F.softmax(logits, dim=2)
        output = 0
        for k, out_k in enumerate(act_outputs):
            w_k = weights[:, :, k].view(x.shape[0], x.shape[1], 1, 1)
            output = output + w_k * out_k
        return output

    def fix_selection(self, x_sample):
        self.eval()
        with torch.no_grad():
            logits, _ = self._get_logits(x_sample)
            if logits.shape[0] == 1:
                self.fixed_indices = torch.argmax(logits[0], dim=1)
            else:
                self.fixed_indices = torch.argmax(logits.mean(dim=0), dim=1)
        self.is_fixed = True
        return self.fixed_indices

    def _apply_hard(self, x):
        out = torch.zeros_like(x)
        for k, act_layer in enumerate(self.acts):
            idx = (self.fixed_indices == k).nonzero(as_tuple=True)[0]
            if len(idx) > 0:
                out.index_copy_(1, idx, act_layer(x.index_select(1, idx)))
        return out


# ==========================================
# 2. Modulator & Hybrid Block
# ==========================================
class ModulatorBank(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, u_views=2, v_views=2):
        super().__init__()
        assert out_channels == 2
        self.k = kernel_size
        self.pad = int((kernel_size - 1) / 2)
        self.kernels_u = nn.Parameter(torch.randn(u_views, 1, in_channels, kernel_size, kernel_size))
        self.kernels_v = nn.Parameter(torch.randn(v_views, 1, in_channels, kernel_size, kernel_size))
        nn.init.normal_(self.kernels_u, std=0.01)
        nn.init.normal_(self.kernels_v, std=0.01)
    '''
    def forward(self, x, u_idx, v_idx):
        # u_idx, v_idx: [B] 的 tensor
        B = x.shape[0]
        results = []
        x_pad = F.pad(x, (self.pad, self.pad, self.pad, self.pad), mode='reflect')
        for i in range(B):
            k_u = self.kernels_u[u_idx[i]]  # [1, C_in, k, k]
            k_v = self.kernels_v[v_idx[i]]  # [1, C_in, k, k]
            active_kernel = torch.cat([k_u, k_v], dim=0)  # [2, C_in, k, k]
            out_i = F.conv2d(x_pad[i:i + 1], active_kernel)
            results.append(out_i)
        return torch.cat(results, dim=0)
    '''
    # 修正并行逻辑
        def forward(self, x, u_idx, v_idx):
        B, C, H, W = x.shape
        x_pad = F.pad(x, (self.pad, self.pad, self.pad, self.pad), mode='reflect')

        # 1. 【并行收集】一次性获取整个 Batch 的卷积核
        # k_u 形状: [B, 1, C_in, k, k]
        k_u = self.kernels_u[u_idx]
        k_v = self.kernels_v[v_idx]

        # 2. 【形状变换】将核拼在一起，并转换为 group 卷积所需的形状
        # active_kernel 拼接后形状: [B, 2, C_in, k, k]
        # 转换为: [B * 2, C_in, k, k]
        active_kernel = torch.cat([k_u, k_v], dim=1).view(B * 2, C, self.k, self.k)

        # 3. 【Batch 融合】把输入图片的 Batch 维度融合进 Channel 维度
        # x_pad 从 [B, C_in, H+pad, W+pad] 变身为 [1, B * C_in, H+pad, W+pad]
        x_pad = x_pad.view(1, B * C, x_pad.shape[2], x_pad.shape[3])

        # 4. 【真正的 GPU 并行计算】利用 groups 参数，一次性算完所有不同的核！
        # groups=B 意味着把输入分为 B 组，每组刚好对应一个样本和它的专属核
        out = F.conv2d(x_pad, active_kernel, groups=B)

        # 5. 【还原形状】算完之后，把输出拆回正常的 Batch 形状 [B, 2, H, W]
        out = out.view(B, 2, H, W)

        return out

class HybridBlock(nn.Module):
    def __init__(self, in_channels, out_channels=64, upsample=True):
        super().__init__()
        self.mod_ch = 2
        self.desc_ch = out_channels - 2
        self.enable_modulator = False

        self.desc_conv = conv_layer(in_channels, self.desc_ch, 3)
        self.desc_bn = nn.BatchNorm2d(self.desc_ch, affine=True)
        # 依然使用 SA
        self.desc_act = ImprovedSelfAttentionAct(channels=self.desc_ch)

        self.mod_conv = ModulatorBank(in_channels, self.mod_ch, 3, u_views=2, v_views=2)
        # 新增
        self.mod_bn = BatchNorm2d(self.mod_ch, affine=True)
        self.mod_act = nn.GELU()

        self.upsample = upsample
        if upsample:
            self.up_layer = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, x, u_idx, v_idx):
        out_desc = self.desc_conv(x)
        out_desc = self.desc_bn(out_desc)
        if self.upsample: out_desc = self.up_layer(out_desc)
        out_desc = self.desc_act(out_desc)

        if self.enable_modulator:
            out_mod = self.mod_conv(x, u_idx, v_idx)
            out_mod = self.mod_bn(out_mod)
            if self.upsample: out_mod = self.up_layer(out_mod)
            out_mod = self.mod_act(out_mod)
        else:
            b, _, h, w = out_desc.shape
            out_mod = torch.zeros(b, self.mod_ch, h, w, device=x.device, dtype=x.dtype)

        return torch.cat([out_desc, out_mod], dim=1)

    def get_act_input(self, x, u_idx, v_idx):
        # 模拟训练时的前向传播路径
        out = self.desc_conv(x)
        out = self.desc_bn(out)
        if self.upsample:
            out = self.up_layer(out)
        return out


# ==========================================
# 3. Decoders
# ==========================================
class GrowingDecoder(nn.Module):
    def __init__(self, fixed_blocks, candidate_block, out_channels=3, target_size=(512, 512)):
        super().__init__()
        self.fixed_blocks = nn.ModuleList(fixed_blocks)
        self.candidate_block = candidate_block
        self.target_size = target_size
        curr_ch = CONFIG['k_channels']
        self.adapter = nn.Sequential(
            conv_layer(curr_ch, out_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x, u_idx, v_idx):
        out = x
        for blk in self.fixed_blocks:
            out = blk(out, u_idx, v_idx)
        out = self.candidate_block(out, u_idx, v_idx)
        out = self.adapter(out)
        if out.shape[2:] != self.target_size:
            out = F.interpolate(out, size=self.target_size, mode='bilinear', align_corners=False)
        return out

    def set_modulator_status(self, status):
        for blk in self.fixed_blocks:
            blk.enable_modulator = status
            # 如果关闭，强制将 BN 设置为 eval 模式，避免统计量漂移
            if not status:
                blk.mod_bn.eval()
            else:
                blk.mod_bn.train()
        self.candidate_block.enable_modulator = status
        if not status:
            self.candidate_block.mod_bn.eval()
        else:
            self.candidate_block.mod_bn.train()


class FinalDecoder(nn.Module):
    def __init__(self, blocks, last_channels, out_channels=3):
        super().__init__()
        self.main_body = nn.ModuleList(blocks)
        self.output_head = nn.Sequential(
            conv_layer(last_channels, out_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x, u_idx, v_idx):
        out = x
        for blk in self.main_body:
            out = blk(out, u_idx, v_idx)
        return self.output_head(out)

    def set_modulator_status(self, status):
        for blk in self.main_body:
            blk.enable_modulator = status
            # 如果关闭，强制将 BN 设置为 eval 模式，避免统计量漂移
            if not status:
                blk.mod_bn.eval()
            else:
                blk.mod_bn.train()



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


# ==========================================
# 5. Main Procedure (Batch 4 + SA + Full Unfreeze)
# ==========================================
def main_final_optimized():
    scene_name = "boxes"
    data_path = "data/boxes"
    save_dir = f'outputs/batch4_{scene_name}'
    os.makedirs(save_dir, exist_ok=True)

    k_channels = CONFIG['k_channels']
    balanced_schedule = [(800, 2200), (600, 2400), (800, 2200), (1200, 2300), (1000, 3000)]
    upsample_configs = [True, True, True, True, False]

    # === [设置 Batch Size] ===
    BATCH_SIZE = 4

    # 加载数据 (字典模式)
    lf_data = load_lf_images(data_path, h=2, w=2)
    view_keys = list(lf_data.keys())  # 所有的 (u, v)
    TOTAL_VIEWS = len(view_keys)

    batch_target = torch.cat([lf_data[k] for k in view_keys], dim=0)  # [4, 3, H, W]
    u_indices = torch.tensor([k[0] for k in view_keys], device=batch_target.device)  # [4]
    v_indices = torch.tensor([k[1] for k in view_keys], device=batch_target.device)  # [4]

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
    net_input_saved = net_input.data.clone().repeat(4, 1, 1, 1)  # [4, C, 32, 32]

    fixed_blocks = []
    last_head = None
    device = first_target.device
    mse = nn.MSELoss()

    # ---------------- Phase 1: Growing ----------------
    print(f"\n{'=' * 20} Phase 1: Layer-wise Growing (SA + Batch 4) {'=' * 20}")

    # 获取中心视图坐标
    center_idx = TOTAL_VIEWS // 2
    center_u, center_v = view_keys[center_idx]
    print(f"Phase 1 will search architecture ONLY on Center View: ({center_u}, {center_v})")

    # Phase 1 的数据固定为中心视图
    b_u_center = torch.tensor([center_u] * BATCH_SIZE, dtype=torch.long, device=device)
    b_v_center = torch.tensor([center_v] * BATCH_SIZE, dtype=torch.long, device=device)
    center_target = lf_data[(center_u, center_v)]

    for layer_idx, (upsample, (n_search, n_refine)) in enumerate(zip(upsample_configs, balanced_schedule)):
        print(f"\n[Layer {layer_idx + 1}] Search: {n_search} | Refine: {n_refine}")

        for blk in fixed_blocks:
            for p in blk.parameters(): p.requires_grad = True
            blk.desc_bn.train()

        cand = HybridBlock(k_channels, k_channels, upsample=upsample)
        net = GrowingDecoder(fixed_blocks, cand, 3, (target_h, target_w)).type(dtype)

        # 强制关闭 Modulator
        net.set_modulator_status(False)
        optimizer = torch.optim.Adam(net.parameters(), lr=CONFIG['search_lr'])
        net.train()

        # --- 1.1 Search 架构 ---
        for i in range(n_search):
            optimizer.zero_grad()
            # 【优化】：只取第 0 个样本，[1, C, H, W] 和 [1]
            out = net(net_input_saved[0:1], b_u_center[0:1], b_v_center[0:1])
            loss = mse(out, center_target)
            loss.backward()
            optimizer.step()

            if i % 200 == 0:
                print(f"    Search Step {i}/{n_search} - Loss: {loss.item():.6f}")

        # --- 1.2 Fix Selection (利用中心视角) ---
        with torch.no_grad():
            net.eval()
            # 【优化点】：这里加上 [0:1]
            x_f = net_input_saved[0:1]
            for blk in net.fixed_blocks: x_f = blk(x_f, b_u_center[0:1], b_v_center[0:1])
            # cand.desc_act.fix_selection(cand.desc_conv(x_f))
            # 使用新增的辅助函数获取经过 BN 和 Up 的特征
            act_input = cand.get_act_input(x_f,b_u_center[0:1], b_v_center[0:1])
            cand.desc_act.fix_selection(act_input)

            c = Counter([cand.desc_act.act_names[idx] for idx in cand.desc_act.fixed_indices.cpu().numpy()])
            print(f"  > Layer {layer_idx + 1} Selection: {dict(c)}")
            net.train()

        # --- 1.3 Refine 权重 ---
        for pg in optimizer.param_groups: pg['lr'] = CONFIG['search_lr'] * 0.8
        for i in range(n_refine):
            optimizer.zero_grad()
            out = net(net_input_saved[0:1], b_u_center[0:1], b_v_center[0:1])
            loss = mse(out, center_target)
            loss.backward()
            optimizer.step()

            if i % 500 == 0:
                print(f"    Refine Step {i}/{n_refine} - Avg Batch Loss: {loss.item():.6f}")

        fixed_blocks.append(net.candidate_block)
        last_head = copy.deepcopy(net.adapter)

        # 简单验证 (打印中心视角 PSNR)
        with torch.no_grad():
            # 【优化点】：这里加上 [0:1]
            out = net(net_input_saved[0:1], b_u_center[0:1], b_v_center[0:1])
            p = psnr(out.cpu().numpy()[0], center_target.cpu().numpy()[0])
            print(f"  > Layer {layer_idx + 1} Done. Center View PSNR: {p:.2f} dB")

    # ---------------- Phase 2: Global Finetuning (Full Unfreeze) ----------------
    print(f"\n{'=' * 20} Phase 2: Global Finetuning (Full Unfreeze & Batch 4) {'=' * 20}")

    final_net = FinalDecoder(fixed_blocks, k_channels, 3).type(dtype)
    if last_head:
        final_net.output_head[0][-1].weight.data = last_head[0][-1].weight.data.clone()
        if last_head[0][-1].bias is not None:
            final_net.output_head[0][-1].bias.data = last_head[0][-1].bias.data.clone()

    # 1. 打开 Modulator
    final_net.set_modulator_status(True)

    # ==========================================
    # 新增：Modulator 零初始化 (Zero-Init)
    # ==========================================
    for blk in final_net.main_body:
        # 将 Modulator 的 BN 层的 weight(gamma) 初始化为 0
        nn.init.constant_(blk.mod_bn.weight, 0)
        nn.init.constant_(blk.mod_bn.bias, 0)
    # ==========================================

    # 2. 【关键】解冻所有参数 (Full Unfreeze)
    # 不再调用 fix_descriptors，而是全部打开
    for p in final_net.parameters():
        p.requires_grad = True

    # 3. 开启训练模式 (允许 BN 更新统计量)
    final_net.train()

    # 4. 差异化学习率
    mod_params = []
    backbone_params = []
    for blk in final_net.main_body:
        mod_params += list(blk.mod_conv.parameters())
        mod_params += list(blk.mod_bn.parameters())  # 加上 BN 参数
        backbone_params += list(blk.desc_conv.parameters())
        backbone_params += list(blk.desc_bn.parameters())
        backbone_params += list(blk.desc_act.parameters())  # SA 的参数
    backbone_params += list(final_net.output_head.parameters())

    main_lr = 0.005
    optimizer_mod = torch.optim.Adam([
        {'params': mod_params, 'lr': main_lr},  # Modulator 强力更新
        {'params': backbone_params, 'lr': main_lr * 0.1}  # Backbone 微调
    ])

    TOTAL_STEPS = 80000
    print(f"  > Training for {TOTAL_STEPS} steps...")

    best_avg_psnr = 0.0
    best_state = None
    history_psnr = []

    for step in range(TOTAL_STEPS):

        if step in [25000, 45000, 65000]:
            for pg in optimizer_mod.param_groups: pg['lr'] *= 0.5
            print(f"    LR Decay at step {step}")

        optimizer_mod.zero_grad()
        out = final_net(net_input_saved, u_indices, v_indices)
        loss = mse(out, batch_target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(final_net.parameters(), max_norm=1.0)
        optimizer_mod.step()

        if step % 500 == 0:
            est_psnr = -10 * np.log10(loss.item())
            history_psnr.append(est_psnr)
            print(f"  Step {step}/{TOTAL_STEPS} - Loss: {loss.item():.6f} | Est. PSNR: {est_psnr:.2f} dB")
            if est_psnr > best_avg_psnr:
                best_avg_psnr = est_psnr
                best_state = copy.deepcopy(final_net.state_dict())

    print(f"Training Done. Best Estimated Batch PSNR: {best_avg_psnr:.2f} dB")

    # === Evaluation ===
    if best_state:
        final_net.load_state_dict(best_state)

    final_net.eval()
    avg_psnr = 0
    with torch.no_grad():
        for (u, v) in view_keys:
            target = lf_data[(u, v)]
            # 【修复点】：包装成 tensor，并放在正确的 device 上
            eval_u = torch.tensor([u], dtype=torch.long, device=device)
            eval_v = torch.tensor([v], dtype=torch.long, device=device)

            # 【修复点】：net_input_saved 是 [4, C, H, W]，评估时只取 [1, C, H, W]
            out = final_net(net_input_saved[0:1], eval_u, eval_v)
            p = psnr(out.cpu().numpy()[0], target.cpu().numpy()[0])
            avg_psnr += p

            plt.imsave(os.path.join(save_dir, f"recon_{u}_{v}.png"),
                       np.clip(out.cpu().numpy()[0].transpose(1, 2, 0), 0, 1))

    print(f"Final Validated Average PSNR: {avg_psnr / TOTAL_VIEWS:.2f} dB")

    plt.figure()
    plt.plot(history_psnr)
    plt.title("Phase 2 Learning Curve (Batch4)")
    plt.savefig(os.path.join(save_dir, "loss_curve.png"))


if __name__ == "__main__":
    seed_everything(42)

    main_final_optimized()
