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

# 引入你的辅助模块 (请确保这些文件在同一目录下)
from helpers import np_to_var, pil_to_np, psnr
from mire_config import CONFIG
from activations import ACT_DICT, get_activation_instance

dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor


# ==========================================
# 0. 基础设置
# ==========================================
def seed_everything(seed=42):
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
# 1. 动态激活函数搜索 (SA)
# ==========================================
class ImprovedSelfAttentionAct(nn.Module):
    def __init__(self, channels, d_model=64, num_heads=4, fixed_tau=0.6, pos_emb_scale=0.5):
        super().__init__()
        self.channels = channels
        self.act_names = ACT_DICT
        self.num_acts = len(self.act_names)
        self.acts = nn.ModuleList([get_activation_instance(name) for name in self.act_names])

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        input_feature_dim = 2 * self.num_acts

        self.chan_pos_emb = nn.Parameter(torch.randn(1, channels, d_model) * pos_emb_scale)
        self.embedding = nn.Linear(input_feature_dim, d_model)
        self.norm_pre = nn.LayerNorm(d_model)

        self.self_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True)
        self.fc = nn.Linear(d_model, self.num_acts)

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
# 2. Modulator (基于 Group Conv 的纯并行版 - 已修复 Reshape Bug)
# ==========================================
class ModulatorBank(nn.Module):
    def __init__(self, in_channels, u_views=9, v_views=9, kernel_size=3):
        super().__init__()
        self.pad = int((kernel_size - 1) / 2)
        # 初始化 9 个水平核和 9 个垂直核
        self.kernels_u = nn.Parameter(torch.randn(u_views, 1, in_channels, kernel_size, kernel_size) * 0.01)
        self.kernels_v = nn.Parameter(torch.randn(v_views, 1, in_channels, kernel_size, kernel_size) * 0.01)

    def forward(self, x, u_indices, v_indices):
        """
        x 形状: [B, C, H, W]
        u_indices, v_indices 形状: [B] (包含当前 Batch 内所有图的独立坐标)
        """
        B, C, H, W = x.shape

        # 1. 并行提取 Batch 中每个样本对应的专属卷积核
        k_u = self.kernels_u[u_indices]  # [B, 1, C, 3, 3]
        k_v = self.kernels_v[v_indices]  # [B, 1, C, 3, 3]

        # 2. 并行拼接，构成每个样本专属的 2 通道卷积核: [B, 2, C, 3, 3]
        active_kernels = torch.cat([k_u, k_v], dim=1)

        # --- 核心修复：把 .view(...) 全部替换为 .reshape(...) ---
        # 3. 展平输入图像，使其变成 1 个超级大图: [1, B*C, H, W]
        x_reshaped = x.reshape(1, B * C, H, W)
        x_pad = F.pad(x_reshaped, (self.pad, self.pad, self.pad, self.pad), mode='reflect')

        # 4. 展平这 B 组卷积核: [B*2, C, 3, 3]
        weights = active_kernels.reshape(B * 2, C, 3, 3)

        # 5. 执行分组卷积 (PyTorch 底层瞬间完成并行计算)
        out_reshaped = F.conv2d(x_pad, weights, groups=B)  # 输出形状: [1, B*2, H, W]

        # 6. 完美还原回 Batch 形状: [B, 2, H, W]
        out = out_reshaped.reshape(B, 2, H, W)
        return out


class HybridBlock(nn.Module):
    def __init__(self, in_channels, out_channels=64, upsample=True):
        super().__init__()
        self.mod_ch = 2  # 调制器 2 通道
        self.desc_ch = out_channels - 2  # 主干网络 62 通道
        self.enable_modulator = False

        self.desc_conv = conv_layer(in_channels, self.desc_ch, 3)
        self.desc_bn = nn.BatchNorm2d(self.desc_ch, affine=True)
        self.desc_act = ImprovedSelfAttentionAct(channels=self.desc_ch)

        self.mod_conv = ModulatorBank(in_channels, u_views=9, v_views=9, kernel_size=3)
        #新增
        self.mod_bn = BatchNorm2d(self.mod_ch,affine=True)
        self.mod_act = nn.GELU()

        self.upsample = upsample
        if upsample:
            self.up_layer = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, x, u_indices, v_indices):
        # --- 主干路 (62通道) ---
        out_desc = self.desc_conv(x)
        out_desc = self.desc_bn(out_desc)
        if self.upsample: out_desc = self.up_layer(out_desc)
        out_desc = self.desc_act(out_desc)
        # out_desc = F.gelu(out_desc)

        # --- 调制路 (2通道) ---
        if self.enable_modulator:
            # 此时 u_indices 和 v_indices 是大小为 B 的一维 Tensor
            out_mod = self.mod_conv(x, u_indices, v_indices)
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
# 3. Decoders (支持 Batched Coordinates)
# ==========================================
class GrowingDecoder(nn.Module):
    def __init__(self, fixed_blocks, candidate_block, out_channels=3, target_size=(512, 512)):
        super().__init__()
        self.fixed_blocks = nn.ModuleList(fixed_blocks)
        self.candidate_block = candidate_block
        self.target_size = target_size
        self.adapter = nn.Sequential(
            conv_layer(CONFIG['k_channels'], out_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x, u_indices, v_indices):
        out = x
        for blk in self.fixed_blocks: out = blk(out, u_indices, v_indices)
        out = self.candidate_block(out, u_indices, v_indices)
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

    def forward(self, x, u_indices, v_indices):
        out = x
        for blk in self.main_body: out = blk(out, u_indices, v_indices)
        return self.output_head(out)

    def set_modulator_status(self, status):
        for blk in self.main_body: blk.enable_modulator = status


# ==========================================
# 4. 数据加载
# ==========================================
def load_lf_images(base_path, h=9, w=9):
    images = {}
    print(f"Loading Light Field from {base_path}...")
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
                images[(u, v)] = np_to_var(img_np).type(dtype)
    return images


# ==========================================
# 5. 主训练流 (真正的并行版 Batch=10)
# ==========================================
def main():
    scene_name = "boxes"
    data_path = f"data/{scene_name}"
    save_dir = f'outputs/debug_exp1_{scene_name}'
    os.makedirs(save_dir, exist_ok=True)
    seed_everything(42)

    k_channels = CONFIG['k_channels']  # 64
    BATCH_SIZE = 10  # 真正的并行 Batch Size

    balanced_schedule = [(800, 2200), (600, 2400), (800, 2200), (1200, 2300), (1000, 3000)]
    upsample_configs = [True, True, True, True, False]

    lf_data = load_lf_images(data_path)
    view_keys = list(lf_data.keys())
    TOTAL_VIEWS = len(view_keys)

    first_target = lf_data[view_keys[0]]

    target_h, target_w = lf_data[view_keys[0]].shape[2:]

    noise_file = CONFIG['noise_file']
    # 固定的初始噪声 [1, C, H, W]
    net_input = torch.load(noise_file).type(dtype)
    net_input_saved = net_input.data.clone()

    fixed_blocks = []
    last_head = None
    device = first_target.device
    mse = nn.MSELoss()

    # ==========================================================
    # Phase 1: Layer-wise Growing
    # ==========================================================
    print(f"\n{'=' * 20} Phase 1: Layer-wise Growing  {'=' * 20}")
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
            '''
            # --- 构建并行的 Batch ---
            batch_indices = torch.randperm(TOTAL_VIEWS)[:BATCH_SIZE]
            u_list, v_list, target_list = [], [], []
            for idx in batch_indices:
                u, v = view_keys[idx]
                u_list.append(u)
                v_list.append(v)
                target_list.append(lf_data[(u, v)])  # [1, 3, H, W]

            u_tensor = torch.tensor(u_list, dtype=torch.long, device=net_input_saved.device)
            v_tensor = torch.tensor(v_list, dtype=torch.long, device=net_input_saved.device)
            target_tensor = torch.cat(target_list, dim=0)  # 拼接目标图像 [B, 3, H, W]

            # 将输入噪声沿着 Batch 维度展开: [B, C, H, W]
            ni_batch = net_input_saved.expand(BATCH_SIZE, -1, -1, -1)
        
            # 真正的一键并行前向传播
            out = net(ni_batch, u_tensor, v_tensor)

            # PyTorch 的 MSELoss 默认求均值，不需要再除以 BATCH_SIZE
            loss = mse(out, target_tensor)
            loss.backward()
            optimizer.step()
            '''
            # 强制输入只有中心视角
            out = net(net_input_saved, b_u_center, b_v_center)
            loss = mse(out, center_target)
            loss.backward()
            optimizer.step()

            if i % 200 == 0:
                print(f"    Search Step {i}/{n_search} - Loss: {loss.item():.6f}")

        # --- 1.2 Fix Selection (利用中心视角) ---
        with torch.no_grad():
            net.eval()
            x_f = net_input_saved
            for blk in net.fixed_blocks: x_f = blk(x_f, b_u_center, b_v_center)
            # cand.desc_act.fix_selection(cand.desc_conv(x_f))
            # 使用新增的辅助函数获取经过 BN 和 Up 的特征
            act_input = cand.get_act_input(x_f, b_u_center, b_v_center)
            cand.desc_act.fix_selection(act_input)

            c = Counter([cand.desc_act.act_names[idx] for idx in cand.desc_act.fixed_indices.cpu().numpy()])
            print(f"  > Layer {layer_idx + 1} Selection: {dict(c)}")
            net.train()


        # --- 1.3 Refine 权重 ---
        for pg in optimizer.param_groups: pg['lr'] = CONFIG['search_lr'] * 0.8
        for i in range(n_refine):
            optimizer.zero_grad()
            '''
            batch_indices = torch.randperm(TOTAL_VIEWS)[:BATCH_SIZE]
            u_list, v_list, target_list = [], [], []
            for idx in batch_indices:
                u, v = view_keys[idx]
                u_list.append(u)
                v_list.append(v)
                target_list.append(lf_data[(u, v)])

            u_tensor = torch.tensor(u_list, dtype=torch.long, device=net_input_saved.device)
            v_tensor = torch.tensor(v_list, dtype=torch.long, device=net_input_saved.device)
            target_tensor = torch.cat(target_list, dim=0)
            ni_batch = net_input_saved.expand(BATCH_SIZE, -1, -1, -1)

            out = net(ni_batch, u_tensor, v_tensor)
            loss = mse(out, target_tensor)
            loss.backward()
            optimizer.step()
            '''
            out = net(net_input_saved, b_u_center, b_v_center)
            loss = mse(out, center_target)
            loss.backward()
            optimizer.step()

            if i % 500 == 0:
                print(f"    Refine Step {i}/{n_refine} - Avg Batch Loss: {loss.item():.6f}")

        fixed_blocks.append(net.candidate_block)
        last_head = copy.deepcopy(net.adapter)

        # 简单验证 (打印中心视角 PSNR)
        with torch.no_grad():
            out = net(net_input_saved,  b_u_center, b_v_center)
            p = psnr(out.cpu().numpy()[0], center_target.cpu().numpy()[0])
            print(f"  > Layer {layer_idx + 1} Done. Center View PSNR: {p:.2f} dB")

    # ==========================================================
    # Phase 2: Global Finetuning
    # ==========================================================
    print(f"\n{'=' * 20} Phase 2: Global Finetuning (Parallel) {'=' * 20}")
    final_net = FinalDecoder(fixed_blocks, k_channels, 3).type(dtype)
    if last_head:
        final_net.output_head[0][-1].weight.data = last_head[0][-1].weight.data.clone()
        if last_head[0][-1].bias is not None:
            final_net.output_head[0][-1].bias.data = last_head[0][-1].bias.data.clone()

    # 【极其关键的一步】：全面唤醒 Modulator！
    final_net.set_modulator_status(True)

    # ==========================================
    # 新增：Modulator 零初始化 (Zero-Init)
    # ==========================================
    for blk in final_net.main_body:
        # 将 Modulator 的 BN 层的 weight(gamma) 初始化为 0
        nn.init.constant_(blk.mod_bn.weight, 0)
        nn.init.constant_(blk.mod_bn.bias, 0)
    # ==========================================

    for p in final_net.parameters(): p.requires_grad = True
    final_net.train()

    mod_params, backbone_params = [], []
    for blk in final_net.main_body:
        mod_params += list(blk.mod_conv.parameters())
        mod_params += list(blk.mod_bn.parameters())  # 必须加上 BN 参数
        backbone_params += list(blk.desc_conv.parameters())
        backbone_params += list(blk.desc_bn.parameters())
        backbone_params += list(blk.desc_act.parameters())
    backbone_params += list(final_net.output_head.parameters())

    main_lr = 0.005
    optimizer_mod = torch.optim.Adam([
        {'params': mod_params, 'lr': main_lr},
        {'params': backbone_params, 'lr': main_lr * 0.1}
    ])

    TOTAL_STEPS = 80000
    best_avg_psnr = 0.0
    best_state = None

    for step in range(TOTAL_STEPS):
        if step in [25000, 45000, 65000]:
            for pg in optimizer_mod.param_groups: pg['lr'] *= 0.5
            print(f"    LR Decay at step {step}")

        optimizer_mod.zero_grad()

        # --- 并行构造 Batch ---
        batch_indices = torch.randperm(TOTAL_VIEWS)[:BATCH_SIZE]
        u_list, v_list, target_list = [], [], []
        for idx in batch_indices:
            u, v = view_keys[idx]
            u_list.append(u)
            v_list.append(v)
            target_list.append(lf_data[(u, v)])

        u_tensor = torch.tensor(u_list, dtype=torch.long, device=net_input_saved.device)
        v_tensor = torch.tensor(v_list, dtype=torch.long, device=net_input_saved.device)
        target_tensor = torch.cat(target_list, dim=0)
        ni_batch = net_input_saved.expand(BATCH_SIZE, -1, -1, -1)

        # 前向传播
        out = final_net(ni_batch, u_tensor, v_tensor)
        loss = mse(out, target_tensor)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(final_net.parameters(), max_norm=1.0)
        optimizer_mod.step()

        if step % 500 == 0:
            est_psnr = 10 * np.log10(1.0 / max(loss.item(), 1e-10))

            # ==========================================
            # 新增：计算所有视图的真实平均 PSNR
            # ==========================================
            final_net.eval()  # 1. 切换到评估模式 (锁定 BN 统计量)
            current_total_psnr = 0.0

            with torch.no_grad():  # 2. 禁用梯度计算，节省显存并加速
                for (u, v) in view_keys:
                    u_t = torch.tensor([u], dtype=torch.long, device=device)
                    v_t = torch.tensor([v], dtype=torch.long, device=device)

                    # 前向传播 (单图推理)
                    out_val = final_net(net_input_saved, u_t, v_t)

                    # 使用导入的 psnr 函数计算
                    p = psnr(out_val.cpu().numpy()[0], lf_data[(u, v)].cpu().numpy()[0])
                    current_total_psnr += p

            actual_avg_psnr = current_total_psnr / TOTAL_VIEWS
            final_net.train()  # 3. 恢复训练模式
            # ==========================================

            # 打印估算 PSNR 和 真实 PSNR
            print(
                f"  Step {step}/{TOTAL_STEPS} | Batch Est. PSNR: {est_psnr:.2f} dB | True Avg PSNR: {actual_avg_psnr:.2f} dB")

            # 使用真实的平均 PSNR 来更新最佳状态
            if actual_avg_psnr > best_avg_psnr:
                best_avg_psnr = actual_avg_psnr
                best_state = copy.deepcopy(final_net.state_dict())
                print(f"    --> New Best Model Saved! (True Avg PSNR: {best_avg_psnr:.2f} dB)")

    # ==========================================================
    # 评估与保存
    # ==========================================================
    final_net.load_state_dict(best_state)
    final_net.eval()
    avg_psnr = 0
    with torch.no_grad():
        for (u, v) in view_keys:
            u_t = torch.tensor([u], dtype=torch.long, device=net_input_saved.device)
            v_t = torch.tensor([v], dtype=torch.long, device=net_input_saved.device)

            out = final_net(net_input_saved, u_t, v_t)
            p = psnr(out.cpu().numpy()[0], lf_data[(u, v)].cpu().numpy()[0])
            avg_psnr += p
            plt.imsave(os.path.join(save_dir, f"recon_{u}_{v}.png"),
                       np.clip(out.cpu().numpy()[0].transpose(1, 2, 0), 0, 1))

    print(f"\n Training Done! Final Average PSNR: {avg_psnr / TOTAL_VIEWS:.2f} dB")


if __name__ == "__main__":
    seed_everything(42)
    main()

