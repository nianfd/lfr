import os
import sys
import random
import numpy as np
import warnings
import argparse

import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from dataset import get_data_transforms, RealIADDataset
from dinov3.hub.backbones import load_dinov3_model

from bank_builder_disk import (
    DiskMultiLayerBank,
    estimate_bank_bytes,
    free_bytes,
    human_bytes,
    remove_tmp_dir,
    same_filesystem,
)

warnings.filterwarnings("ignore")


def setup_seed(seed: int = 1):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class IndexedDataset(Dataset):
    """
    Wrap a base Dataset so that __getitem__ returns (img, label, idx).
    Useful for index-based operations (e.g., excluding self in retrieval).
    """
    def __init__(self, base_dataset):
        self.base = base_dataset

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, label = self.base[idx]
        return img, label, idx


@torch.no_grad()
def build_memory_bank_multilayer(
    encoder,
    dataset,
    device,
    batch_size: int = 32,
    layer_idx_list=(3, 6, 9, 11),
    num_workers: int = 4,
    tmp_dir=None,
):
    """
    Build a multi-layer feature bank using DINOv3 intermediate features
    on Real-IAD train set (normal images).

    encoder.get_intermediate_layers(..., n=layer_idx_list, return_class_token=True)
    returns a list whose elements are (patch_tokens, cls_token):
        patch_tokens: [B, L, C]
        cls_token:    [B, C]

    改造点（原实现把全部特征累积在内存里再 torch.cat，VisA/Real-IAD 规模必然 OOM）：
    特征按顺序直接写进磁盘映射文件 tmp_dir，峰值常驻内存只有「一个 batch」，
    最后由 torch.save 从映射零拷贝流式写出 .pth。
    行顺序 / dtype / 布局都与原实现一致 -> 落盘产物逐字节相同。

    Returns:
      layers:      list[int], sorted layer indices
      cls_banks:   list[Tensor[N, C]], aligned with layers
      patch_banks: list[Tensor[N, L, C]], aligned with layers
                   （磁盘映射张量；torch.save 完成后调用 remove_tmp_dir 清理）
    """
    encoder.eval()
    layer_idx_list = sorted(list(layer_idx_list))
    num_layers = len(layer_idx_list)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=str(device).startswith("cuda"),
    )

    total_images = len(dataset)
    sink = DiskMultiLayerBank(tmp_dir, num_layers, total_images, verbose=True)

    order_ok = True
    for step, (img, label, idx) in enumerate(loader):
        img = img.to(device, non_blocking=True)

        inter = encoder.get_intermediate_layers(
            img,
            n=layer_idx_list,
            return_class_token=True,
            reshape=False,
            norm=True,
        )

        # inter follows the same order as layer_idx_list
        for li, (patch_tok, cls_tok) in enumerate(inter):
            # [B, L, C] / [B, C] 直接落盘（GPU 特征 -> 映射文件，一次 D2H 拷贝）
            sink.append(li, patch_tok, cls_tok)

        # 写入顺序 = DataLoader 顺序；顺带确认没有 shuffle（否则 bank 行序会错位）
        start = int(sink.written) - int(img.shape[0])
        if order_ok and int(idx.flatten()[0]) != start:
            order_ok = False

        if step % 50 == 0:
            print(f"[Build disk] step={step}/{len(loader)} "
                  f"rows={sink.written}/{total_images}")

    if not order_ok:
        print("[Build disk] 警告: 批次 idx 不是 0..N-1 顺序，写入顺序仍与 DataLoader "
              "顺序一致（与原实现相同），但如果这不是你想要的，请检查 shuffle 设置。")

    cls_banks, patch_banks = sink.tensors()
    for li in range(num_layers):
        print(f"[Build disk] layer {layer_idx_list[li]} 完成: "
              f"cls={tuple(cls_banks[li].shape)} patch={tuple(patch_banks[li].shape)}")

    return layer_idx_list, cls_banks, patch_banks


def main():
    parser = argparse.ArgumentParser(
        description="Build MULTI-LAYER memory bank for Real-IAD using DINOv3 encoder"
    )
    parser.add_argument('--data_path', type=str, default='../Real-IAD')

    parser.add_argument('--item_list', nargs='+', default=[
        'audiojack', 'bottle_cap', 'button_battery', 'end_cap', 'eraser', 'fire_hood',
        'mint', 'mounts', 'pcb', 'phone_battery', 'plastic_nut', 'plastic_plug',
        'porcelain_doll', 'regulator', 'rolled_strip_base', 'sim_card_set', 'switch', 'tape',
        'terminalblock', 'toothbrush', 'toy', 'toy_brick', 'transistor1', 'usb',
        'usb_adaptor', 'u_block', 'vcpill', 'wooden_beads', 'woodstick', 'zipper'
    ])

    parser.add_argument('--encoder_name', type=str, default='dinov3_vitb16')
    parser.add_argument('--encoder_weight', type=str,
                        default='../weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth')
    parser.add_argument('--image_size', type=int, default=512)
    parser.add_argument('--crop_size', type=int, default=448)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)

    parser.add_argument('--layer_idx_list', nargs='+', type=int,
                        default=[3, 6, 9, 11],
                        help='0-based block indices for multi-layer bank, e.g. 3 6 9 11')

    parser.add_argument('--bank_path', type=str,
                        default='./bank/realiad_dinov3_vitb16_multilayer36911_448_bank.pth',
                        help='output .pth containing multi-layer banks')

    parser.add_argument('--tmp_dir', type=str, default=None,
                        help='特征临时文件目录（默认 <bank_path 所在目录>/.bank_tmp_<名字>）；'
                             'bank 很大时建议指到空间充裕的盘')
    parser.add_argument('--keep_tmp', action='store_true',
                        help='构建完成后保留临时特征文件（排错用）')

    args = parser.parse_args()

    setup_seed(1)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print("device:", device)
    print("layers (0-based):", args.layer_idx_list)
    print("item_list:", args.item_list)

    # Data augmentation (consistent with training)
    data_transform, gt_transform = get_data_transforms(args.image_size, args.crop_size)

    # Build multi-category training set (Real-IAD, phase='train')
    train_data_list = []
    for i, item in enumerate(args.item_list):
        train_data = RealIADDataset(
            root=args.data_path,
            category=item,
            transform=data_transform,
            gt_transform=gt_transform,
            phase='train'
        )
        # Give each category a unique label id i
        train_data.classes = item
        train_data.class_to_idx = {item: i}
        train_data_list.append(train_data)

    train_data = ConcatDataset(train_data_list)
    indexed_train_data = IndexedDataset(train_data)
    print("Total train images (Real-IAD):", len(train_data))

    # Load and freeze DINOv3 encoder
    encoder = load_dinov3_model(
        args.encoder_name,
        pretrained_weight_path=args.encoder_weight
    )
    encoder.to(device)
    for p in encoder.parameters():
        p.requires_grad = False
    encoder.eval()

    # ---- 临时目录 / 磁盘空间预检 -----------------------------------------
    out_dir = os.path.dirname(os.path.abspath(args.bank_path))
    tmp_dir = args.tmp_dir or os.path.join(
        out_dir, ".bank_tmp_" + os.path.basename(args.bank_path))

    dim = getattr(encoder, "embed_dim", None)
    patch_size = int(getattr(encoder, "patch_size", 16) or 16)
    if dim is not None:
        grid = args.crop_size // patch_size
        bank_bytes = estimate_bank_bytes(
            len(train_data), len(args.layer_idx_list), grid * grid, int(dim), 4)
        same = same_filesystem(tmp_dir, args.bank_path)
        free_tmp = free_bytes(tmp_dir)
        free_out = free_bytes(args.bank_path)
        print(f"[bank] 预估: {len(train_data)} 图 x {len(args.layer_idx_list)} 层 x "
              f"{grid}x{grid} patch x {dim}d, dtype=float32")
        print(f"[bank] bank .pth ≈ {human_bytes(bank_bytes)}（原实现峰值内存需 "
              f"{human_bytes(2 * bank_bytes)}）")
        print(f"[bank] 临时特征 ≈ {human_bytes(bank_bytes)} -> {tmp_dir} "
              f"(free {human_bytes(free_tmp)})")
        print(f"[bank] 输出 .pth   ≈ {human_bytes(bank_bytes)} -> {out_dir} "
              f"(free {human_bytes(free_out)})")
        if same:
            need_ok = free_tmp >= 2 * bank_bytes
            need_txt = f"{human_bytes(2 * bank_bytes)}（同一文件系统，临时+输出）"
        else:
            need_ok = free_tmp >= bank_bytes and free_out >= bank_bytes
            need_txt = f"{human_bytes(bank_bytes)} 临时 + {human_bytes(bank_bytes)} 输出"
        if not need_ok:
            sys.exit(
                f"[bank] 磁盘空间不足: 需要 {need_txt}。\n"
                f"       临时目录剩余 {human_bytes(free_tmp)}，输出目录剩余 "
                f"{human_bytes(free_out)}。\n"
                f"       可用 --tmp_dir /path/to/big/disk 把临时文件指到别的盘，"
                f"或用 --item_list 减少类别数。")
    else:
        print("[bank] 提示: encoder 没有 embed_dim 属性，跳过磁盘空间预检")

    # Build multi-layer memory bank (磁盘落地，峰值内存 ≈ 单个 batch)
    layers, cls_banks, patch_banks = build_memory_bank_multilayer(
        encoder=encoder,
        dataset=indexed_train_data,
        device=device,
        batch_size=args.batch_size,
        layer_idx_list=args.layer_idx_list,
        num_workers=args.num_workers,
        tmp_dir=tmp_dir,
    )

    # Print shapes
    for li, layer in enumerate(layers):
        print(f"layer={layer} cls_bank={tuple(cls_banks[li].shape)} "
              f"patch_bank={tuple(patch_banks[li].shape)}")

    os.makedirs(out_dir, exist_ok=True)
    torch.save(
        {
            "layers": layers,          # list[int]
            "cls_banks": cls_banks,    # list[Tensor[N, C]]
            "patch_banks": patch_banks # list[Tensor[N, L, C]]
        },
        args.bank_path
    )
    print(f"Saved multi-layer bank to {args.bank_path} "
          f"({human_bytes(os.path.getsize(args.bank_path))})")

    # 释放映射并清理临时文件（此时 .pth 已写完，删掉不影响结果）
    del cls_banks, patch_banks
    remove_tmp_dir(tmp_dir, keep=args.keep_tmp)


if __name__ == '__main__':
    main()
