"""
RAD (ICML2026) inference script -- 单卡（24GB）显存改造版.

原实现的瓶颈
------------
MVTec 多类协议下，记忆库由 15 个类的全部训练图构成：
    N = 3643 张图 x L = 784 patches x C = 768 维 x 4 layers x 4 B
    = 32.69 GiB
原代码在 `eval_patch_knn_multilayer_cluster` 里一次性
`patch_banks_dev = [pb.to(device) for pb in patch_banks]`，
把 32.69 GiB 直接搬上显卡 —— 单张 4090(24GB) 必然 OOM。

改造方案
--------
1. patch bank 常驻 CPU（`GPUPatchBankCache` 管理），只有"当前测试图真正
   用到的邻居行"[k, L, C] 才按需拷到 GPU，并留在显存里做缓存。
   第一层检索（CLS top-k）仍然在 GPU 上全库比，因为 cls bank 只有 42.7 MiB。
2. 位姿对齐的 patch-KNN 去掉 784 次 Python 循环，改成 "(2r+1)^2 个偏移 +
   批量 gather + 掩码 max" 的向量化实现（r=1 时 9 个偏移）。
3. 所有归约都是 max，分块计算与整体计算在数学上等价，因此**精度不受影响**。

新增命令行参数
--------------
--gpu_bank_cache_mb   显存里给记忆库缓存留的预算(MiB)。0=自动(取空闲显存的
                      55%, 且不超过整库大小)；-1=整库常驻显存(等价原实现)。
--normalize_bank      入缓存时做 L2 归一化(默认开, 与原实现等价)。
--knn_chunk_imgs      位姿分支每次参与比对的邻居图数(控制瞬时显存)。
--global_chunk_imgs   非位姿分支每次参与比对的邻居图数。
--batch_size/--num_workers  原先硬编码为 8/4, 现可配置。
"""

import os
import random
import time
import numpy as np
import warnings
import logging
import argparse

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torch.nn import functional as F

from sklearn.metrics import roc_auc_score, average_precision_score

from dataset import get_data_transforms, MVTecDataset, resolve_item_list
from dinov3.hub.backbones import load_dinov3_model

from utils import compute_pro, f1_score_max, get_gaussian_kernel

from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


def get_logger(name, save_path=None, level='INFO'):
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level))

    log_format = logging.Formatter('%(message)s')
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(log_format)
    logger.addHandler(streamHandler)

    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        fileHandler = logging.FileHandler(os.path.join(save_path, 'log.txt'))
        fileHandler.setFormatter(log_format)
        logger.addHandler(fileHandler)

    return logger


def setup_seed(seed: int = 1):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def tensor_to_np(img_tensor):
    if img_tensor.dim() == 4:
        img_tensor = img_tensor[0]
    img = img_tensor.detach().cpu().unsqueeze(0)   # [1,3,H,W]
    img = img * IMAGENET_STD + IMAGENET_MEAN       # de-normalize
    img = img.clamp(0, 1)[0]
    img = img.permute(1, 2, 0).numpy()
    return img


# ============================================================================
# 1) 显存受限的 patch bank 缓存
# ============================================================================
class GPUPatchBankCache:
    """CPU 常驻的多层 patch bank + 显存行缓存。

    CPU 侧每层是 [N_l, L, C] 的 float32 张量；一行(row) = 一张训练图的
    patch 集合 [L, C]。最多把 ``cap_l`` 行放在显存的一块预分配 buffer 里，
    请求到未驻留的行时按需拷贝（并按原实现的语义做 L2 归一化）。
    某层缓存满时直接整层丢弃重来 —— 检索式 AD 中一个数据集类真正用到的
    行集合远小于整库，命中率极高，丢弃很少发生。

    数值等价性说明：
      * 存入缓存的行只在"入缓存时"归一化一次，与"每次用前归一化"逐元素等价；
      * 请求时按行号 gather 回 [k, L, C]，与原实现的 `bank_l[neigh_indices]`
        完全一致（同一顺序、同一 dtype）。
    """

    def __init__(self, cpu_banks, device, budget_bytes,
                 normalize=True, verbose=True):
        self.banks = list(cpu_banks)
        self.device = torch.device(device)
        self.num_layers = len(self.banks)
        self.N = [int(b.shape[0]) for b in self.banks]
        self.L = int(self.banks[0].shape[1])
        self.C = int(self.banks[0].shape[2])
        self.normalize = bool(normalize)
        self.row_bytes = self.L * self.C * 4

        for li, b in enumerate(self.banks):
            assert b.dim() == 3 and b.shape[1] == self.L and b.shape[2] == self.C, \
                "all bank layers must share the same [L, C]"

        self.total_bytes = sum(n * self.row_bytes for n in self.N)

        per_layer = max(1, int(budget_bytes) // self.num_layers // self.row_bytes)
        self.cap = [min(self.N[li], per_layer) for li in range(self.num_layers)]

        self.store = None
        self.slot = None
        self.used = None
        self.hits = 0
        self.misses = 0
        self._allocate()

        if verbose:
            print(
                "[bank-cache] total bank = {:.2f} GiB | GPU budget = {:.2f} GiB | "
                "cap/layer = {} | time={}".format(
                    self.total_bytes / 2 ** 30,
                    self.resident_bytes() / 2 ** 30,
                    self.cap,
                    time.strftime("%H:%M:%S"),
                )
            )

    # ------------------------------------------------------------------
    def _allocate(self):
        self.store = [
            torch.empty((c, self.L, self.C), dtype=torch.float32, device=self.device)
            for c in self.cap
        ]
        self.slot = [
            torch.full((n,), -1, dtype=torch.int64, device=self.device)
            for n in self.N
        ]
        self.used = [0] * self.num_layers

    def resident_bytes(self):
        return sum(self.cap) * self.row_bytes

    def clear_layer(self, li):
        self.slot[li].fill_(-1)
        self.used[li] = 0

    def reset(self):
        self.hits = 0
        self.misses = 0
        self._allocate()

    def report(self):
        tot = self.hits + self.misses
        hit = (self.hits / tot * 100.0) if tot else 0.0
        return (f"bank-cache rows: hit={self.hits} miss={self.misses} "
                f"(hit rate {hit:.2f}%)")

    # ------------------------------------------------------------------
    @torch.no_grad()
    def get(self, li, rows):
        """rows: 1-D LongTensor(CPU). 返回显存上的 [k, L, C]（已归一化）。"""
        rows = rows.detach().to("cpu", torch.long).contiguous()
        k = int(rows.numel())
        if k == 0:
            return self.store[li][:0]

        # 缓存太小装不下（例如 VisA k_image=900 但预算很小）-> 直通，不缓存
        if k > self.cap[li]:
            feat = self.banks[li].index_select(0, rows).to(self.device)
            if self.normalize:
                feat = F.normalize(feat, dim=-1)
            self.misses += k
            return feat

        slot = self.slot[li]
        rows_dev = rows.to(self.device)
        s = slot[rows_dev]
        miss = s < 0
        n_miss = int(miss.sum())
        if n_miss > 0:
            if self.used[li] + n_miss > self.cap[li]:
                self.clear_layer(li)
                s = slot[rows_dev]
                miss = s < 0
                n_miss = int(miss.sum())

            miss_rows = rows[miss.cpu()]
            base = self.used[li]
            dst = self.store[li][base:base + n_miss]
            dst.copy_(self.banks[li].index_select(0, miss_rows).to(self.device))
            if self.normalize:
                # 与 F.normalize(x, dim=-1, eps=1e-12) 完全相同的算子序列
                dst.div_(dst.norm(dim=-1, keepdim=True).clamp_min_(1e-12))
            new_slots = torch.arange(
                base, base + n_miss, dtype=torch.int64, device=self.device
            )
            slot.scatter_(0, miss_rows.to(self.device), new_slots)
            self.used[li] = base + n_miss
            self.misses += n_miss
            s = slot[rows_dev]

        self.hits += k - n_miss
        return self.store[li].index_select(0, s)


# ============================================================================
# 2) 向量化的 patch-KNN
# ============================================================================
def _grid_hw(encoder, img, L):
    """patch 网络的 H/W。优先用 encoder.patch_size 推，避免 int(L**0.5) 猜错。"""
    ps = getattr(encoder, "patch_size", None)
    if ps is not None:
        h = int(img.shape[-2]) // int(ps)
        w = int(img.shape[-1]) // int(ps)
        if h * w == L:
            return h, w
    g = int(round(L ** 0.5))
    if g * g == L:
        return g, g
    raise RuntimeError(f"cannot infer patch grid from L={L}")


def build_pos_context(H, W, r, device):
    """预计算位姿分支需要的一切静态信息。

    offsets[k] = (dr, dc) 为第 k 个空间偏移；
    tables[k]  = (idx, valid)：
        idx[L]   查询位置 j -> bank 扁平下标 pos(j)+Δ（越界处 clamp 占位）
        valid[L] 该 (j, Δ) 组合是否落在网格内
    """
    rows = torch.arange(H, device=device)
    cols = torch.arange(W, device=device)
    offsets, tables = [], []
    for dr in range(-r, r + 1):
        rr = (rows + dr).clamp_(0, H - 1)
        row_ok = ((rows + dr) >= 0) & ((rows + dr) <= H - 1)
        for dc in range(-r, r + 1):
            cc = (cols + dc).clamp_(0, W - 1)
            col_ok = ((cols + dc) >= 0) & ((cols + dc) <= W - 1)
            idx = (rr[:, None] * W + cc[None, :]).reshape(-1)          # [L]
            valid = (row_ok[:, None] & col_ok[None, :]).reshape(-1)     # [L]
            offsets.append((dr, dc))
            tables.append((idx.contiguous(), valid.contiguous()))
    return {"H": H, "W": W, "r": r, "L": H * W,
            "offsets": offsets, "tables": tables}


@torch.no_grad()
def patch_scores_positional(q_norm, bank_norm, ctx, chunk_imgs=32, normalize=False):
    """位姿受限 patch-KNN（向量化，无 Python 位置循环）。

    q_norm:    [L, C]      pre-normalised when ``normalize=False``
    bank_norm: [k, L, C]   pre-normalised when ``normalize=False`` (L == H*W)
    normalize: True 时内部做 L2 归一化（等价原实现）；False 时要求调用方
               已归一化（默认路径由 GPUPatchBankCache 完成）
    ctx:       build_pos_context(...) 的返回值
    返回:      [L]   patch 异常分 = 1 - max_{k, 邻域} cos

    做法：把「位置」当成 bmm 的 batch 维，一次矩阵乘算出所有
    (查询位置 p, 邻居图 m, 偏移 Δ) 的内积，再做 (2r+1)^2 次极小的
    重排 + max。整图整层只读写记忆库一次，而不是每个偏移各读一次。
    """
    if normalize:
        q_norm = F.normalize(q_norm, dim=-1)
        bank_norm = F.normalize(bank_norm, dim=-1)

    H, W, r = ctx["H"], ctx["W"], ctx["r"]
    tables, offsets = ctx["tables"], ctx["offsets"]
    L = ctx["L"]

    k, Lb, C = bank_norm.shape
    if Lb != L:
        raise RuntimeError(
            "positional bank requires bank and test patch grids to match, "
            f"got bank L={Lb} vs test L={L} (H={H}, W={W})"
        )

    # 查询侧窗口：Qc[p, c, kk] = q[p - Δkk]
    qp = F.pad(q_norm.reshape(H, W, C).permute(2, 0, 1), (r, r, r, r))  # [C, H+2r, W+2r]
    Kp = len(offsets)
    Qc = torch.empty(L, C, Kp, dtype=q_norm.dtype, device=q_norm.device)
    for kk, (dr, dc) in enumerate(offsets):
        win = qp[:, r - dr:r - dr + H, r - dc:r - dc + W].reshape(C, L)  # [C, L]
        Qc[:, :, kk] = win.t()

    best = torch.full((L,), float("-inf"), dtype=bank_norm.dtype, device=bank_norm.device)
    for s0 in range(0, k, chunk_imgs):
        blk = bank_norm[s0:s0 + chunk_imgs]                 # [m, L, C]
        m = blk.shape[0]
        Bt = blk.permute(1, 0, 2).contiguous()              # [L, m, C]
        out = torch.bmm(Bt, Qc)                             # [L, m, Kp]
        out = out.permute(2, 0, 1).contiguous()             # [Kp, L, m]
        for kk, (idx, valid) in enumerate(tables):
            cur = out[kk].index_select(0, idx)              # [L, m]
            cur = cur.masked_fill(~valid.unsqueeze(1), float("-inf"))
            torch.maximum(best, cur.max(dim=1).values, out=best)

    # 查询位置自身永远有效，-inf 只可能出现在不可能的空邻域上（原实现记为 1.0）
    best = torch.nan_to_num(best, neginf=0.0)
    return 1.0 - best


@torch.no_grad()
def patch_scores_global(q_norm, bank_norm, chunk_imgs=64, normalize=False):
    """全局 patch-KNN（不带位姿约束）。输入语义同上。"""
    if normalize:
        q_norm = F.normalize(q_norm, dim=-1)
        bank_norm = F.normalize(bank_norm, dim=-1)

    k, L, C = bank_norm.shape
    flat = bank_norm.reshape(k * L, C)
    best = None
    for s0 in range(0, k, chunk_imgs):
        s1 = min(s0 + chunk_imgs, k)
        blk = flat[s0 * L:s1 * L]                          # [m*L, C] (view)
        sim = q_norm @ blk.t()                             # [L, m*L]
        m = sim.max(dim=-1).values                         # [L]
        best = m if best is None else torch.maximum(best, m)
    return 1.0 - best


# ============================================================================
# 3) 主评测循环
# ============================================================================
@torch.no_grad()
def eval_patch_knn_multilayer_cluster(
    encoder,
    layer_indices,           # list[int], same order as in bank
    cls_banks,               # list[Tensor], each [N, C]
    patch_banks,             # list[Tensor], each [N, L, C]  (CPU-resident)
    test_dataloader,
    device,
    k_image=5,
    resize_mask=256,
    max_ratio=0.0,
    vis_dir=None,
    max_vis=0,
    retrieve_transform=None,
    train_img_paths=None,
    use_positional_bank=False,
    pos_radius=1,
    layer_weights=None,      # list[float] or None
    bank_cache=None,         # GPUPatchBankCache；None 时按整库建一个（小库场景）
    knn_chunk_imgs=32,
    global_chunk_imgs=64,
    verbose=True,
    log_every=20,
):
    """Multi-layer patch-KNN.

    cls_banks:   list[Tensor], each [N, C]          -> 常驻显存（很小）
    patch_banks: list[Tensor], each [N, L, C]       -> 常驻 CPU，按需上卡
    layer_weights: list[float], length = num_layers, if None then uniform weights
    """

    encoder.eval()
    device = torch.device(device)

    num_layers = len(layer_indices)
    assert len(cls_banks) == num_layers
    assert len(patch_banks) == num_layers

    if bank_cache is None:
        total = sum(b.numel() * b.element_size() for b in patch_banks)
        bank_cache = GPUPatchBankCache(
            patch_banks, device, budget_bytes=total, normalize=True, verbose=verbose
        )

    # image-level nearest neighbors use the highest layer CLS (只占 42.7 MiB)
    cls_banks_dev = [cb.to(device) for cb in cls_banks]
    cls_bank_global = cls_banks_dev[-1]            # [N, C]
    cls_bank_global_norm = F.normalize(cls_bank_global, dim=-1)

    # layer weights (default: uniform)
    if layer_weights is None or len(layer_weights) != num_layers:
        layer_weights = [1.0 / num_layers] * num_layers
    else:
        s = sum(layer_weights)
        if s <= 0:
            layer_weights = [1.0 / num_layers] * num_layers
        else:
            layer_weights = [w / s for w in layer_weights]

    gaussian_kernel = get_gaussian_kernel(kernel_size=5, sigma=1.0).to(device)

    gt_list_px = []
    pr_list_px = []
    gt_list_sp = []
    pr_list_sp = []

    remain_vis = max_vis
    if vis_dir is not None:
        os.makedirs(vis_dir, exist_ok=True)

    pos_ctx = None
    n_seen = 0
    t0 = time.time()

    for img, gt, label, img_path in test_dataloader:
        img = img.to(device)
        gt = gt.to(device)

        # 1) extract multi-layer features
        inter = encoder.get_intermediate_layers(
            img,
            n=layer_indices,
            return_class_token=True,
            reshape=False,
            norm=True,
        )
        patch_list = [p for (p, _) in inter]   # each [B, L, C]
        cls_list = [c for (_, c) in inter]     # each [B, C]

        # image-level similarity with highest layer CLS
        cls_x_global_norm = F.normalize(cls_list[-1], dim=-1)                 # [B, C]
        sim_img = torch.matmul(cls_x_global_norm, cls_bank_global_norm.t())   # [B, N]
        _, topk_idx = torch.topk(sim_img, k_image, dim=-1)                    # [B, k]

        B, L, C = patch_list[0].shape
        h, w = _grid_hw(encoder, img, L)

        if use_positional_bank and pos_ctx is None:
            pos_ctx = build_pos_context(h, w, pos_radius, device)

        topk_idx_cpu = topk_idx.detach().to("cpu")

        # 2) multi-layer patch-KNN
        patch_scores_batch = []
        for b in range(B):
            rows = topk_idx_cpu[b]           # [k] (CPU)
            scores_per_layer = []

            for li in range(num_layers):
                q_feat = F.normalize(patch_list[li][b], dim=-1)     # [L, C]
                # 只把这张测试图需要的 k 行邻居搬到显存（未命中部分才真正传输）
                neigh_feat = bank_cache.get(li, rows)               # [k, L, C]

                if use_positional_bank:
                    patch_score_l = patch_scores_positional(
                        q_feat, neigh_feat, pos_ctx, chunk_imgs=knn_chunk_imgs
                    )
                else:
                    patch_score_l = patch_scores_global(
                        q_feat, neigh_feat, chunk_imgs=global_chunk_imgs
                    )

                scores_per_layer.append(patch_score_l)

            # multi-layer linear fusion
            fused = torch.zeros_like(scores_per_layer[0])
            for li in range(num_layers):
                fused = fused + layer_weights[li] * scores_per_layer[li]

            patch_scores_batch.append(fused)

        patch_scores_batch = torch.stack(patch_scores_batch, dim=0)    # [B, L]

        # 3) reshape to patch map and upsample to pixel-level map
        patch_maps = patch_scores_batch.view(B, 1, h, w)  # [B,1,h,w]

        anomaly_map = F.interpolate(
            patch_maps,
            size=resize_mask,
            mode='bilinear',
            align_corners=False
        )  # [B,1,H',W']

        # 4) align GT and apply Gaussian smoothing
        gt = F.interpolate(gt, size=resize_mask, mode='nearest')
        anomaly_map = gaussian_kernel(anomaly_map)

        gt = gt.bool()
        if gt.shape[1] > 1:
            gt = torch.max(gt, dim=1, keepdim=True)[0]

        # 5) visualization
        if vis_dir is not None and remain_vis > 0:
            for b in range(B):
                if remain_vis <= 0:
                    break

                inp_np = tensor_to_np(img[b])
                amap_np = anomaly_map[b, 0].detach().cpu().numpy()
                gt_np = gt[b, 0].detach().cpu().numpy()

                # visualize nearest neighbor from highest layer CLS
                nn_idx = topk_idx[b, 0].item()
                nn_img_np = None
                if train_img_paths is not None:
                    nn_path = train_img_paths[nn_idx]
                    nn_pil = Image.open(nn_path).convert('RGB')
                    if retrieve_transform is not None:
                        nn_tensor = retrieve_transform(nn_pil)
                        nn_img_np = tensor_to_np(nn_tensor)
                    else:
                        nn_img_np = np.array(nn_pil)

                fig, axes = plt.subplots(1, 4 if nn_img_np is None else 5, figsize=(15, 3))

                axes[0].imshow(inp_np)
                axes[0].set_title("Input")
                axes[0].axis('off')

                col = 1
                if nn_img_np is not None:
                    axes[1].imshow(nn_img_np)
                    axes[1].set_title(f"Nearest (idx={nn_idx})")
                    axes[1].axis('off')
                    col = 2

                im = axes[col].imshow(amap_np, cmap='jet')
                axes[col].set_title("Anomaly Map")
                axes[col].axis('off')
                fig.colorbar(im, ax=axes[col], fraction=0.046, pad=0.04)

                axes[col + 1].imshow(gt_np, cmap='gray')
                axes[col + 1].set_title("GT mask")
                axes[col + 1].axis('off')

                plt.tight_layout()
                img_name = os.path.basename(img_path[b])
                save_path = os.path.join(vis_dir, f"vis_{img_name}")
                plt.savefig(save_path, dpi=150)
                plt.close()

                remain_vis -= 1

        # 6) collect metrics
        gt_list_px.append(gt)              # [B,1,H,W]
        pr_list_px.append(anomaly_map)     # [B,1,H,W]
        gt_list_sp.append(label)           # [B]

        # image-level score: max or top-ratio mean
        if max_ratio == 0:
            sp_score = torch.max(anomaly_map.flatten(1), dim=1)[0]       # [B]
        else:
            amap_flat = anomaly_map.flatten(1)
            k = int(amap_flat.shape[1] * max_ratio)
            sp_score = torch.sort(amap_flat, dim=1, descending=True)[0][:, :k].mean(dim=1)
        pr_list_sp.append(sp_score)

        n_seen += B
        if verbose and log_every and (n_seen // B) % log_every == 0:
            txt = bank_cache.report()
            print(f"    [{n_seen:5d} imgs | {time.time() - t0:6.1f}s] {txt}", flush=True)

    # 7) compute AUROC / AUPRO etc.
    gt_list_px = torch.cat(gt_list_px, dim=0)[:, 0].cpu().numpy()
    pr_list_px = torch.cat(pr_list_px, dim=0)[:, 0].cpu().numpy()
    gt_list_sp = torch.cat(gt_list_sp).flatten().cpu().numpy()
    pr_list_sp = torch.cat(pr_list_sp).flatten().cpu().numpy()

    aupro_px = compute_pro(gt_list_px, pr_list_px)

    gt_list_px, pr_list_px = gt_list_px.ravel(), pr_list_px.ravel()

    auroc_px = roc_auc_score(gt_list_px, pr_list_px)
    auroc_sp = roc_auc_score(gt_list_sp, pr_list_sp)
    ap_px = average_precision_score(gt_list_px, pr_list_px)
    ap_sp = average_precision_score(gt_list_sp, pr_list_sp)

    f1_sp = f1_score_max(gt_list_sp, pr_list_sp)
    f1_px = f1_score_max(gt_list_px, pr_list_px)

    return [auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px]


def load_bank(bank_path, use_mmap=True, print_fn=print):
    """加载记忆库，返回 (ckpt, 模式说明)。

    VisA multi-class 协议的 bank 有 77 GiB，用默认的 torch.load 一次性读进内存，
    在 128 GB 机器上会和构建脚本一样被顶爆。torch.load(..., mmap=True) 走内存
    映射按需分页：常驻内存近乎为零，页缓存随取随放，正好配 GPUPatchBankCache 的
    「只把当前测试图需要的行搬上卡」。两者读的是同一份字节，数值完全一致。
    """
    if use_mmap:
        try:
            return torch.load(bank_path, map_location="cpu", mmap=True), "mmap lazy"
        except TypeError:
            print_fn("[bank] 当前 torch 不支持 mmap=True（需 >= 2.1），回退常规加载")
        except Exception as exc:                       # 文件系统不支持映射等
            print_fn(f"[bank] mmap 加载失败({type(exc).__name__}: {exc})，回退常规加载")
    return torch.load(bank_path, map_location="cpu"), "eager RAM"


def validate_bank_consistency(patch_banks, n_train_imgs, item_list, data_path,
                              bank_path, crop_size, encoder_patch_size, k_image):
    """在开跑评测前，拦住三类「要么崩、要么静默算错」的配置不一致。

    1) 训练图数 != bank 行数  -> item_list/data_path 与构建 bank 时不是同一套
       （最容易被忽略：类别写少了会照常跑完，只是数字全错）
    2) patch 网格 L 不一致     -> crop_size / encoder patch_size 与构建时不同
    3) k_image 超出 [1, N]     -> torch.topk 会抛索引越界，报错信息看不出原因

    返回 (N, L)；任一项不通过则 SystemExit 并打印可操作的提示。
    """
    n_bank = int(patch_banks[0].shape[0])
    bank_L = int(patch_banks[0].shape[1])

    if n_train_imgs != n_bank:
        raise SystemExit(
            f"\n[bank 自检失败] 训练图数量与记忆库不一致：\n"
            f"  按 --item_list {item_list}\n"
            f"  在 {data_path} 数到 {n_train_imgs} 张训练图，\n"
            f"  但 {bank_path} 里有 {n_bank} 行。\n"
            f"  说明本次的 --item_list / --data_path 与构建 bank 时用的不是同一套。\n"
            f"  请用构建 bank 时的那份 --item_list（顺序也要一致），或重建 bank。\n"
        )

    if encoder_patch_size:
        expect_L = (crop_size // int(encoder_patch_size)) ** 2
        if expect_L != bank_L:
            raise SystemExit(
                f"\n[bank 自检失败] patch 网格数不匹配：\n"
                f"  crop_size={crop_size}, encoder patch_size={encoder_patch_size}"
                f" -> L={expect_L}，但 bank 里 L={bank_L}。\n"
                f"  说明构建 bank 时的 --image_size / --crop_size 与本次不同，\n"
                f"  请用 --crop_size 对齐（或重建 bank）。\n"
            )

    if not (0 < k_image <= n_bank):
        raise SystemExit(
            f"\n[参数自检失败] --k_image={k_image} 超出范围（bank 共 {n_bank} 行）。\n"
            f"  k_image 是「取多少个最近邻训练图」，必须 1 <= k_image <= {n_bank}。\n"
        )

    return n_bank, bank_L


def main():
    parser = argparse.ArgumentParser(
        description='Multi-layer Patch-KNN on MVTEC with DINOv3 (single-GPU friendly)'
    )
    parser.add_argument('--data_path', type=str, default='../mvtec_anomaly_detection')
    parser.add_argument(
        '--bank_path',
        type=str,
        default='./bank/mvtec_dinov3_vitb16_multilayer36911_224_bank.pth'
    )
    parser.add_argument('--encoder_name', type=str, default='dinov3_vitb16')
    parser.add_argument(
        '--encoder_weight',
        type=str,
        default='../weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth'
    )
    parser.add_argument('--save_dir', type=str, default='./saved_results_mvtec')
    parser.add_argument('--save_name', type=str, default='mvtec_multilayer_36911_224')
    parser.add_argument(
        '--k_image',
        type=int,
        default=150,
        help='top-K nearest neighbor images for local patch bank'
    )
    parser.add_argument('--resize_mask', type=int, default=256)
    parser.add_argument(
        '--max_ratio',
        type=float,
        default=0.01,
        help='top ratio for image-level score (0 means use max)'
    )
    parser.add_argument(
        '--vis_max',
        type=int,
        default=8,
        help='max number of test visualizations per class'
    )

    parser.add_argument(
        '--use_positional_bank',
        action='store_true',
        help='use position-aware local patch bank (with neighborhood)'
    )
    parser.add_argument(
        '--pos_radius',
        type=int,
        default=1,
        help='neighborhood radius (in patch grid) for position-aware bank (0 = only same position)'
    )

    parser.add_argument(
        '--layer_weights',
        nargs='+',
        type=float,
        default=None,
        help='weights for each layer (same length as bank layers), default uniform'
    )

    # ---- 单卡显存相关（新增） -------------------------------------------
    parser.add_argument(
        '--gpu_bank_cache_mb',
        type=int,
        default=0,
        help='显存中给 patch 记忆库缓存留的预算(MiB)。'
             '0 = 自动(空闲显存的 55%%, 上限为整库大小)；-1 = 整库常驻显存(等价原实现)'
    )
    parser.add_argument(
        '--normalize_bank',
        type=int,
        default=1,
        help='1 = 行入缓存时做 L2 归一化(与原实现等价), 0 = 不归一化'
    )
    parser.add_argument(
        '--load_mmap',
        type=int,
        default=1,
        choices=[0, 1],
        help='1 = 用 mmap 懒加载 bank（推荐：VisA 级 bank(77 GiB) 常驻内存≈0，'
             '页缓存可回收，数值与一次性加载完全一致）；0 = 一次性读进内存'
    )
    parser.add_argument(
        '--knn_chunk_imgs', type=int, default=32,
        help='位姿分支每次参与比对的邻居图数(控制瞬时显存)'
    )
    parser.add_argument(
        '--global_chunk_imgs', type=int, default=64,
        help='非位姿分支每次参与比对的邻居图数'
    )
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=4)

    # item_list made configurable
    # 注意：默认值是 MVTec 的 15 类。用 VisA / Real-IAD 时必须覆写，否则会在
    # ImageFolder(<data_path>/carpet/train) 处抛 FileNotFoundError。
    # 现在有兜底：若默认列表在 --data_path 下不可用，会自动落到磁盘探测结果，
    # 并打印警告（见 dataset.resolve_item_list）。传 'auto' 可强制探测。
    parser.add_argument(
        '--item_list',
        nargs='+',
        type=str,
        default=[
            'carpet', 'grid', 'leather', 'tile', 'wood',
            'bottle', 'cable', 'capsule', 'hazelnut', 'metal_nut',
            'pill', 'screw', 'toothbrush', 'transistor', 'zipper'
        ],
        help="类别列表；MVTec 用默认值(15 类)，VisA 用 "
             "'candle capsules cashew chewinggum fryum macaroni1 macaroni2 "
             "pcb1 pcb2 pcb3 pcb4 pipe_fryum'，也可以用 'auto' 让脚本从 "
             "--data_path 自动探测。默认列表在 data_path 下不可用时会自动探测。"
    )

    # 与构建 bank 时必须一致；默认值即原硬编码值，不改动既有调用
    parser.add_argument(
        '--image_size', type=int, default=512,
        help='与构建 bank 时的 --image_size 保持一致'
    )
    parser.add_argument(
        '--crop_size', type=int, default=448,
        help='与构建 bank 时的 --crop_size 保持一致（决定 patch 网格 L）'
    )

    args = parser.parse_args()

    setup_seed(1)

    os.makedirs(os.path.join(args.save_dir, args.save_name), exist_ok=True)
    logger = get_logger(
        args.save_name,
        os.path.join(args.save_dir, args.save_name)
    )
    print_fn = logger.info

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print_fn(f"device: {device}")
    if device.startswith('cuda'):
        print_fn("GPU: {} | total {:.2f} GiB".format(
            torch.cuda.get_device_name(0),
            torch.cuda.get_device_properties(0).total_memory / 2 ** 30))
    print_fn(
        f"use_positional_bank = {args.use_positional_bank}, "
        f"pos_radius = {args.pos_radius}"
    )
    # ---- 类别列表：与磁盘上的 dataset 对齐（不再直接信任默认值）----------
    item_list = resolve_item_list(
        args.data_path, args.item_list, layout='folder', print_fn=print_fn
    )
    print_fn(f"item_list = {item_list}")

    # test-time transforms (can differ from bank-building transforms if desired)
    image_size = args.image_size
    crop_size = args.crop_size
    data_transform, gt_transform = get_data_transforms(image_size, crop_size)

    train_data_list = []
    test_data_list = []
    train_img_paths = []

    for i, item in enumerate(item_list):
        train_path = os.path.join(args.data_path, item, 'train')
        test_path = os.path.join(args.data_path, item)

        train_data = ImageFolder(root=train_path, transform=data_transform)
        train_data.classes = item
        train_data.class_to_idx = {item: i}
        train_data.samples = [(sample[0], i) for sample in train_data.samples]

        test_data = MVTecDataset(
            root=test_path,
            transform=data_transform,
            gt_transform=gt_transform,
            phase="test"
        )

        train_data_list.append(train_data)
        test_data_list.append(test_data)

        for path, _ in train_data.samples:
            train_img_paths.append(path)

    print_fn(f"train image number (for bank): {len(train_img_paths)}")

    # encoder and bank
    encoder = load_dinov3_model(
        args.encoder_name,
        pretrained_weight_path=args.encoder_weight
    )
    encoder = encoder.to(device)
    for p in encoder.parameters():
        p.requires_grad = False
    encoder.eval()

    t_load = time.time()
    bank_ckpt, load_mode = load_bank(args.bank_path, use_mmap=bool(args.load_mmap),
                                     print_fn=print_fn)
    layer_indices = bank_ckpt["layers"]          # list[int], 0-based block indices
    cls_banks = bank_ckpt["cls_banks"]           # list[Tensor], each [N, C]
    patch_banks = bank_ckpt["patch_banks"]       # list[Tensor], each [N, L, C]
    print_fn(f"bank loaded in {time.time() - t_load:.1f}s [{load_mode}] "
             f"({torch.cuda.memory_allocated() / 2**30:.2f} GiB already on GPU)")

    num_layers = len(layer_indices)
    print_fn(f"Loaded multi-layer bank with {num_layers} layers: {layer_indices}")
    for li in range(num_layers):
        print_fn(
            f"  Layer {layer_indices[li]}: cls_bank {tuple(cls_banks[li].shape)}, "
            f"patch_bank {tuple(patch_banks[li].shape)}"
        )

    # ---- bank 与本次数据/编码器的三项一致性自检 ---------------------------
    # 这三点不一致时，下面要么报错要么**静默跑出错误数字**，所以前置拦掉。
    n_bank, bank_L = validate_bank_consistency(
        patch_banks=patch_banks,
        n_train_imgs=len(train_img_paths),
        item_list=item_list,
        data_path=args.data_path,
        bank_path=args.bank_path,
        crop_size=crop_size,
        encoder_patch_size=getattr(encoder, 'patch_size', None),
        k_image=args.k_image,
    )
    print_fn(f"bank 自检通过: N={n_bank}, L={bank_L}, "
             f"k_image={args.k_image}, crop_size={crop_size}")

    # ---- patch bank 缓存预算 ---------------------------------------------
    total_bank_bytes = sum(b.numel() * b.element_size() for b in patch_banks)
    if args.gpu_bank_cache_mb < 0:
        budget = total_bank_bytes                     # 整库上卡（等价原实现）
        mode = "full-GPU (legacy behaviour)"
    elif args.gpu_bank_cache_mb > 0:
        budget = args.gpu_bank_cache_mb * 1024 ** 2
        mode = f"manual {args.gpu_bank_cache_mb} MiB"
    else:
        free, total = torch.cuda.mem_get_info()
        budget = min(int(0.55 * free), total_bank_bytes)
        mode = f"auto (0.55 x {free / 2**30:.2f} GiB free)"
    budget = max(budget, min(total_bank_bytes, 4 * args.k_image *
                             patch_banks[0].shape[1] * patch_banks[0].shape[2] * 4))
    print_fn(f"patch-bank on CPU = {total_bank_bytes / 2**30:.2f} GiB | "
             f"GPU cache budget = {budget / 2**30:.2f} GiB [{mode}]")

    bank_cache = GPUPatchBankCache(
        patch_banks, device, budget_bytes=budget, normalize=bool(args.normalize_bank)
    )
    print_fn(f"  -> cap per layer = {bank_cache.cap} rows "
             f"(row = {bank_cache.row_bytes / 2**20:.2f} MiB)")

    # layer weights
    if args.layer_weights is None:
        layer_weights = None
        print_fn("layer_weights: uniform")
    else:
        assert len(args.layer_weights) == num_layers, \
            f"len(layer_weights) must == {num_layers}"
        layer_weights = args.layer_weights
        print_fn(f"layer_weights: {layer_weights}")

    # per-class evaluation
    auroc_sp_list, ap_sp_list, f1_sp_list = [], [], []
    auroc_px_list, ap_px_list, f1_px_list, aupro_px_list = [], [], [], []

    t_all = time.time()
    for item, test_data in zip(item_list, test_data_list):
        test_dataloader = DataLoader(
            test_data,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers
        )

        vis_dir = os.path.join(
            args.save_dir, args.save_name, "test_vis_multilayer_cluster", item
        )

        t_cls = time.time()
        print_fn(f"--- {item}: {len(test_data)} test images ---")
        results = eval_patch_knn_multilayer_cluster(
            encoder=encoder,
            layer_indices=layer_indices,
            cls_banks=cls_banks,
            patch_banks=patch_banks,
            test_dataloader=test_dataloader,
            device=device,
            k_image=args.k_image,
            resize_mask=args.resize_mask,
            max_ratio=args.max_ratio,
            vis_dir=vis_dir,
            max_vis=args.vis_max,
            retrieve_transform=data_transform,
            train_img_paths=train_img_paths,
            use_positional_bank=args.use_positional_bank,
            pos_radius=args.pos_radius,
            layer_weights=layer_weights,
            bank_cache=bank_cache,
            knn_chunk_imgs=args.knn_chunk_imgs,
            global_chunk_imgs=args.global_chunk_imgs,
        )

        auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px = results
        auroc_sp_list.append(auroc_sp)
        ap_sp_list.append(ap_sp)
        f1_sp_list.append(f1_sp)
        auroc_px_list.append(auroc_px)
        ap_px_list.append(ap_px)
        f1_px_list.append(f1_px)
        aupro_px_list.append(aupro_px)

        print_fn(
            f'{item} (MultiLayer): '
            f'I-AUROC:{auroc_sp:.4f}, I-AP:{ap_sp:.4f}, I-F1:{f1_sp:.4f}, '
            f'P-AUROC:{auroc_px:.4f}, P-AP:{ap_px:.4f}, P-F1:{f1_px:.4f}, P-AUPRO:{aupro_px:.4f}'
            f'  [{time.time() - t_cls:.1f}s]'
        )
        print_fn("  " + bank_cache.report())

    print_fn(
        'Mean (MultiLayer): '
        'I-AUROC:{:.4f}, I-AP:{:.4f}, I-F1:{:.4f}, '
        'P-AUROC:{:.4f}, P-AP:{:.4f}, P-F1:{:.4f}, P-AUPRO:{:.4f}'.format(
            np.mean(auroc_sp_list), np.mean(ap_sp_list), np.mean(f1_sp_list),
            np.mean(auroc_px_list), np.mean(ap_px_list),
            np.mean(f1_px_list), np.mean(aupro_px_list))
    )
    print_fn(f"total time: {time.time() - t_all:.1f}s | "
             f"peak GPU mem: {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")


if __name__ == '__main__':
    main()
