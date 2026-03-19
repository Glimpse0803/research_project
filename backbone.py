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
from helpers import np_to_var, pil_to_np, psnr
from mire_config import CONFIG
from activations import ACT_DICT, get_activation_instance
from conv_layer import asy_dir_conv_block
from frequency import frequency_analysis
from conv_layer import DirectionalConv2d

###############################################################################
################### 参考性能（可训练参数量 -- PSNR）, 81张图，boxes ###############
# 20904 -- 36.08dB
# 33009 -- 37.87dB
# 47814 -- 38.50dB
# 65319 -- 39.60dB
# 108429 -- 40.57dB
# 162339 -- 41.38dB
# 227049 -- 41.79dB
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
# 0. 基础组件 + 预处理函数
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
    convolver = nn.Conv2d(in_f, out_f, kernel_size, stride, padding='same', bias=False,
                          dilation=dilation)  # Jinglei: 1. 这里padding直接用“same"即可   5.这里添加了dilation，分别为[1,2,2,2,1]
    torch.nn.init.kaiming_normal_(convolver.weight, mode='fan_in', nonlinearity='relu')
    # layers = filter(lambda x: x is not None, [padder, convolver])
    layers = filter(lambda x: x is not None, [convolver])
    return nn.Sequential(*layers)

def pretreatment(lf_data, branch_channels):
    ratio_list = []
    channel_list = []

    ratio_list = frequency_analysis(lf_data)

    for i in range(3):
        channel_list.append(round(branch_channels * ratio_list[i]))

    channel_list.append(branch_channels - sum(channel_list))

    return channel_list


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
    def __init__(self, in_channels, out_channels, kernel_size=3, u_views=9, v_views=9):
        super().__init__()
        assert out_channels == 2
        self.k = kernel_size
        self.pad = int((kernel_size - 1) / 2)
        self.kernels_u = nn.Parameter(torch.randn(u_views, 1, in_channels, kernel_size, kernel_size))
        self.kernels_v = nn.Parameter(torch.randn(v_views, 1, in_channels, kernel_size, kernel_size))
        torch.nn.init.kaiming_normal_(self.kernels_u, mode='fan_in',
                                      nonlinearity='relu')  # Jinglei: 2. 这里我稍微改了一下初始化方法，不过对性能影响不大
        torch.nn.init.kaiming_normal_(self.kernels_v, mode='fan_in', nonlinearity='relu')
        # nn.init.normal_(self.kernels_u, std=0.01)
        # nn.init.normal_(self.kernels_v, std=0.01)

    def forward(self, x, u_idx, v_idx):
        k_u = self.kernels_u[u_idx]
        k_v = self.kernels_v[v_idx]
        active_kernel = torch.cat([k_u, k_v], dim=0)
        x_pad = F.pad(x, (self.pad, self.pad, self.pad, self.pad), mode='reflect')
        out = F.conv2d(x_pad, active_kernel)
        return out


class HybridBlock(nn.Module):
    def __init__(self, channel_list, in_channels, out_channels=64, upsample=True, dilation=1):
    # def __init__(self, in_channels, out_channels=64, upsample=True, dilation=1):
        super().__init__()
        self.mod_ch = 2
        self.desc_ch = out_channels - 2
        self.enable_modulator = False

        # self.desc_conv = conv_layer(in_channels, self.desc_ch, 3, dilation)
        self.desc_conv = asy_dir_conv_block(in_channels,self.desc_ch,channel_list,'45','135', dilation=dilation)
        self.desc_bn = nn.BatchNorm2d(out_channels)  # nn.BatchNorm2d(self.desc_ch, affine=True)
        # self.batch_norm = nn.BatchNorm2d(out_channels)
        # 依然使用 SA
        self.desc_act = nn.GELU()  # ImprovedSelfAttentionAct(channels=self.desc_ch)

        self.mod_conv = ModulatorBank(in_channels, self.mod_ch, 3)
        self.mod_act = nn.GELU()

        self.upsample = upsample
        if upsample:
            self.up_layer = nn.Upsample(scale_factor=2, mode='bicubic',
                                        align_corners=False)  # 4. Jinglei: upsampling方法用bicubic，更加准确

    def forward(self, x, u_idx, v_idx):
        # up batch act
        out_desc = self.desc_conv(x)
        # out_desc = self.desc_bn(out_desc)
        # if self.upsample: out_desc = self.up_layer(out_desc)
        # out_desc = self.desc_act(out_desc)

        if self.enable_modulator:
            out_mod = self.mod_conv(x, u_idx, v_idx)
            # out_mod = self.mod_act(out_mod)
        else:
            b, _, h, w = out_desc.shape
            out_mod = torch.zeros(b, self.mod_ch, h, w, device=x.device)

        out_put = torch.cat([out_desc, out_mod], dim=1)
        if self.upsample: out_put = self.up_layer(out_put)
        out_put = self.desc_bn(out_put)  # 3. Jinglei: 这里，我简单使用了 concat - bn - act的形式，后面我们根据需要调整
        out_put = self.mod_act(out_put)

        return out_put

        # return torch.cat([out_desc, out_mod], dim=1)


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
            conv_layer(curr_ch, out_channels, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x, u_idx, v_idx):
        out = x
        for blk in self.fixed_blocks:
            out = blk(out, u_idx, v_idx)
        out = self.candidate_block(out, u_idx, v_idx)
        out = self.adapter(out)
        if out.shape[2:] != self.target_size:
            out = F.interpolate(out, size=self.target_size, mode='bicubic',
                                align_corners=False)  # 4. Jinglei: upsampling方法用bicubic，更加准确
        return out

    def set_modulator_status(self, status):
        for blk in self.fixed_blocks: blk.enable_modulator = status
        self.candidate_block.enable_modulator = status


class FinalDecoder(nn.Module):
    def __init__(self, blocks, last_channels, out_channels=3):
        super().__init__()
        self.main_body = nn.ModuleList(blocks)
        self.output_head = nn.Sequential(
            conv_layer(last_channels, out_channels, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x, u_idx, v_idx):
        out = x
        for blk in self.main_body:
            out = blk(out, u_idx, v_idx)
        return self.output_head(out)

    def set_modulator_status(self, status):
        for blk in self.main_body: blk.enable_modulator = status


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
# 5. Main Procedure (Batch 20 + SA + Full Unfreeze)
# ==========================================
def main_final_optimized():
    scene_name = "boxes"
    data_path = "data/boxes"
    save_dir = f'outputs/backbone_{scene_name}'
    os.makedirs(save_dir, exist_ok=True)

    # 确保模型保存目录存在
    model_save_dir = f'outputs/model_{scene_name}'
    os.makedirs(model_save_dir, exist_ok=True)
    best_model_path = os.path.join(model_save_dir, 'best_model.pth')

    k_channels = CONFIG['k_channels']
    branch_channels = CONFIG['branch_channels']
    # balanced_schedule = [(600, 2400), (800, 2200), (600, 2400), (1000, 2500), (800, 3200)]
    balanced_schedule = [(10, 20), (10, 20), (10, 20), (10, 20), (10, 20)]  # Jinglei: 这里需要调整回去
    upsample_configs = [True, True, True, True, False]
    dilations = [1, 2, 2, 2, 1]  # Jinglei: 这里是我设置的各层dilation

    # === [设置 Batch Size] ===
    BATCH_SIZE = 81

    # 加载数据 (字典模式)
    lf_data = load_lf_images(data_path)
    view_keys = list(lf_data.keys())  # 所有的 (u, v)
    TOTAL_VIEWS = len(view_keys)

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

    # 预处理：计算卷积层各支路比例

    fixed_blocks = []
    channel_list = []
    last_head = None
    mse = nn.MSELoss()

    # 预处理：计算卷积层各支路比例
    channel_list = pretreatment(lf_data, branch_channels)

    # ---------------- Phase 1: Growing ----------------
    print(f"\n{'=' * 20} Phase 1: Layer-wise Growing (SA + Batch 20) {'=' * 20}")

    for layer_idx, (upsample, (n_search, n_refine), dilation) in enumerate(
            zip(upsample_configs, balanced_schedule, dilations)):
        print(f"\n[Layer {layer_idx + 1}] Search: {n_search} | Refine: {n_refine}")

        for blk in fixed_blocks:
            for p in blk.parameters(): p.requires_grad = True
            blk.desc_bn.train()
            # blk.batch_norm.train()

        # cand = HybridBlock(k_channels, k_channels, upsample=upsample, dilation=dilation)
        cand = HybridBlock(channel_list, k_channels, k_channels, upsample=upsample, dilation=dilation)
        net = GrowingDecoder(fixed_blocks, cand, 3, (target_h, target_w)).type(dtype)

        net.set_modulator_status(False)
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

            # 随机采样 20 个索引
            batch_indices = torch.randperm(TOTAL_VIEWS)[:BATCH_SIZE]

            for idx in batch_indices:
                u, v = view_keys[idx]  # 从列表中取 Key
                target = lf_data[(u, v)]  # 从字典中取 Tensor

                optimizer.zero_grad()
                with autocast('cuda',dtype=torch.bfloat16):  # ← 包裹前向和 loss
                    out = net(ni, u, v)
                    loss = mse(out, target) / BATCH_SIZE
                loss.backward()

            optimizer.step()
            if i % 100 == 0:
                print(f"    Search Step {i}/{n_search} - Avg Batch Loss: {loss:.6f}")

        # === Fix Selection ===
        # with torch.no_grad():
        #     x_f = net_input_saved
        #     for blk in net.fixed_blocks: x_f = blk(x_f, 4, 4)
        #     cand.desc_act.fix_selection(cand.desc_conv(x_f))
        #     # 打印选择结果
        #     indices = cand.desc_act.fixed_indices.cpu().numpy()
        #     c = Counter([cand.desc_act.act_names[idx] for idx in indices])
        #     print(f"  > Layer {layer_idx + 1} Selection: {dict(c)}")

        # === Phase 1.2 Refine ===

        for pg in optimizer.param_groups: pg['lr'] = CONFIG['search_lr'] * 0.8
        for i in range(n_refine):
            accum_loss = 0.0

            batch_indices = torch.randperm(TOTAL_VIEWS)[:BATCH_SIZE]
            for idx in batch_indices:
                u, v = view_keys[idx]
                target = lf_data[(u, v)]

                optimizer.zero_grad()
                with autocast('cuda',dtype=torch.bfloat16):
                    out = net(net_input_saved, u, v)
                    loss = mse(out, target)
                loss.backward()
                optimizer.step()
            if i % 500 == 0:
                print(f"    Refine Step {i}/{n_refine} - Avg Batch Loss: {accum_loss:.6f}")

        fixed_blocks.append(net.candidate_block)
        last_head = copy.deepcopy(net.adapter)

        # 简单验证
        with torch.no_grad():
            mid_key = view_keys[TOTAL_VIEWS // 2]
            out = net(net_input_saved, mid_key[0], mid_key[1])
            p = psnr(out.cpu().numpy()[0], lf_data[mid_key].cpu().numpy()[0])
            print(f"  > Layer {layer_idx + 1} Done. Mid-View PSNR: {p:.2f} dB")

    # ---------------- Phase 2: Global Finetuning (Full Unfreeze) ----------------
    print(f"\n{'=' * 20} Phase 2: Global Finetuning (Full Unfreeze & Batch 20) {'=' * 20}")

    final_net = FinalDecoder(fixed_blocks, k_channels, 3).type(dtype)
    if last_head:
        final_net.output_head[0][-1].weight.data = last_head[0][-1].weight.data.clone()
        if last_head[0][-1].bias is not None:
            final_net.output_head[0][-1].bias.data = last_head[0][-1].bias.data.clone()

    # 1. 打开 Modulator
    final_net.set_modulator_status(True)

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
        backbone_params += list(blk.desc_conv.parameters())
        backbone_params += list(blk.desc_bn.parameters())
        backbone_params += list(blk.desc_act.parameters())  # SA 的参数 (如PReLU)
    backbone_params += list(final_net.output_head.parameters())

    main_lr = 0.005  # Jinglei: 我稍微试了大一点的学习率，可以调小为0.003
    optimizer_mod = torch.optim.Adam([
        {'params': mod_params, 'lr': main_lr},  # Modulator 强力更新
        {'params': backbone_params, 'lr': main_lr}],  # Backbone 微调
        betas=(0.9, 0.99),  # Jinglei: 这里我细化了一下，对结果影响不大
        eps=1e-14,
    )

    TOTAL_STEPS = 500 * 5 * 10

    # 统计参数
    total, effective = count_model_parameters(final_net)
    print(f"  > Training for {TOTAL_STEPS} steps...")

    best_avg_psnr = 0.0
    best_state = None
    history_psnr = []

    lf_order = spiral_order(9)  # Jinglei: 采用螺旋扫描方式

    for step in range(TOTAL_STEPS):
        if step in [500 * 5 * 3, 500 * 5 * 6,
                    500 * 5 * 9]:  # [25000, 45000, 65000]:                   # Jinglei: 我把TOTAL_STEPS写成a*b*c的形式，可以调整c控制阶段
            for pg in optimizer_mod.param_groups: pg['lr'] *= 0.5
            print(f"    LR Decay at step {step}")

        # accum_loss = 0.0

        # 随机 Batch 20
        # batch_indices = torch.randperm(TOTAL_VIEWS)[:BATCH_SIZE]
        # print("U V is", batch_indices) # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # batch_indices = torch.randperm(81)[:5]
        for [u_org, v_org] in lf_order:  # Jinglei: 不要随机化，直接用spiral order扫描
            u, v = u_org - 1, v_org - 1  # 光场图片下标是从1开始的，所以要-1

            target = lf_data[(u, v)]

            optimizer_mod.zero_grad()
            with autocast('cuda', dtype=torch.bfloat16):    # ← BF16，不需要 scaler
                out = final_net(net_input_saved, u, v)
                loss = mse(out, target)
            loss.backward()
            torch.nn.utils.clip_grad_value_(final_net.parameters(), clip_value=0.5)
            '''
            out = final_net(net_input_saved, u, v)
            loss = mse(out, target)  # / BATCH_SIZE
            torch.nn.utils.clip_grad_value_(final_net.parameters(), clip_value=0.5)  # Jinglei: 我把梯度剪裁改成按value进行剪裁了
            loss.backward()
            '''
            optimizer_mod.step()

            # accum_loss += loss.item()

        # 梯度裁剪
        # torch.nn.utils.clip_grad_norm_(final_net.parameters(), max_norm=1.0)
        # optimizer_mod.step()

        if step % 2500 == 0:
            # est_psnr = -10 * np.log10(accum_loss)
            # history_psnr.append(est_psnr)
            # print(f"  Step {step}/{TOTAL_STEPS} - Avg Loss: {accum_loss:.6f} | Est. PSNR: {est_psnr:.2f} dB")
            with torch.no_grad():
                psnr_list = []
                for (u, v) in view_keys:
                    target = lf_data[(u, v)]
                    with autocast('cuda',dtype=torch.bfloat16):  # ← 加上这个
                        out = final_net(net_input_saved, u, v)
                    p = psnr(out.float().cpu().numpy()[0], target.cpu().numpy()[0])
                    # out = final_net(net_input_saved, u, v)
                    # p = psnr(out.cpu().numpy()[0], target.cpu().numpy()[0])
                    psnr_list.append(p)
                avg_psnr = np.mean(psnr_list)  # Jinglei: PSNR 平均可以用List来算
                print(f"  Step {step}/{TOTAL_STEPS} - Est. PSNR: {avg_psnr:.2f} dB")
            if avg_psnr > best_avg_psnr:
                best_avg_psnr = avg_psnr
                # best_state = copy.deepcopy(final_net.state_dict())
                # 覆盖保存最佳模型到同一个文件
                torch.save({
                        'psnr': best_avg_psnr,
                        'model_state_dict': final_net.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }, best_model_path)

    print(f"Training Done. Best Estimated Batch PSNR: {best_avg_psnr:.2f} dB")

    # # === Evaluation ===
    # if best_state:
    #     final_net.load_state_dict(best_state)

    # # final_net.train()
    # final_net.eval()
    # avg_psnr = 0
    # with torch.no_grad():
    #     for (u, v) in view_keys:
    #         target = lf_data[(u, v)]
    #         with autocast('cuda',dtype=torch.bfloat16):  # ← 评估也用 autocast 保持一致
    #             out = final_net(net_input_saved, u, v)
    #         out = out.float()  # ← 统一转 FP32
    #         p = psnr(out.cpu().numpy()[0], target.cpu().numpy()[0])
    #         avg_psnr += p  # ← 用 avg_psnr 而非 psnr_list

    #         plt.imsave(os.path.join(save_dir, f"recon_{u}_{v}.png"),
    #                    np.clip(out.cpu().numpy()[0].transpose(1, 2, 0), 0, 1))

    # print(f"Final Validated Average PSNR: {avg_psnr / TOTAL_VIEWS:.2f} dB")

    # plt.figure()
    # plt.plot(history_psnr)
    # plt.title("Phase 2 Learning Curve (SA + Unfreeze + Batch20)")
    # plt.savefig(os.path.join(save_dir, "loss_curve.png"))


if __name__ == "__main__":
    seed_everything(42)
    main_final_optimized()