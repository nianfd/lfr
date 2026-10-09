# -*- coding: utf-8 -*-
"""磁盘落地的多层特征库构建工具（RAM 安全）。

为什么需要它
------------
官方 ``build_bank_multilayer.py`` 的写法是::

    patch_feats[li].append(patch_tok.cpu())       # 全量累积在 list 里
    ...
    patch_bank_l = torch.cat(patch_feats[li], 0)  # 再整体 cat 一份

VisA（multi-class 协议，12 类，8659 张训练图，448x448 -> 28x28 patch，C=768）::

    单层 = 8659 * 784 * 768 * 4B = 19.42 GiB
    4 层 = 77.69 GiB
    累积 list + torch.cat 峰值 = 2 * 77.69 = 155.4 GiB   # 128 GB 机器必被 OOM kill

本模块把每个 batch 的特征直接写进**磁盘映射文件**（临时），峰值常驻内存只有
「一个 batch」；最后由 ``torch.save`` 从映射里零拷贝流式写出 .pth。
落盘产物与原实现 **逐字节相同**（行顺序 / dtype / 布局都不变），所以下游
评测脚本不需要任何改动。

代价：构建期间需要约 2x bank 大小的临时磁盘（特征临时文件 + 最终 .pth），
构建结束后临时文件自动删除；用 ``--tmp_dir`` 可以把临时文件指到别的盘。

实现要点
--------
* 用 ``torch.from_file(path, shared=True, ...)`` 拿到一个直接映射到文件的
  张量，``buf[n:n+b].copy_(feature)`` 就是一次 D2H 写入（GPU 特征直接落盘，
  比原来 ``.cpu()`` + ``cat`` 还少一次拷贝）。
* 分配靠 ``ftruncate``，不预分配内存；文件是稀疏的，写到哪算哪。
* ``torch.save`` 写出时读的是映射视图（page cache），因此不产生额外常驻内存。
"""

import gc
import os
import shutil

import torch

__all__ = [
    "DiskMultiLayerBank",
    "human_bytes",
    "free_bytes",
    "same_filesystem",
    "estimate_bank_bytes",
    "remove_tmp_dir",
    "release_all",
]

# 模块级注册表：持有所有活跃 sink，便于统一释放映射（删临时文件前必须释放）
_ACTIVE_SINKS = []


def human_bytes(nbytes):
    n = float(nbytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if n < 1024.0 or unit == "TiB":
            return "%.2f %s" % (n, unit)
        n /= 1024.0


def _existing_prefix(path):
    probe = os.path.abspath(path)
    while probe and not os.path.exists(probe):
        parent = os.path.dirname(probe)
        if parent == probe:
            break
        probe = parent
    return probe or "."


def free_bytes(path):
    """path 所在文件系统的剩余字节数（path 不存在时向上找最近的父目录）。"""
    return shutil.disk_usage(_existing_prefix(path)).free


def same_filesystem(path_a, path_b):
    """两个路径是否在同一个文件系统上（决定临时空间是否要翻倍）。"""
    try:
        return os.stat(_existing_prefix(path_a)).st_dev == \
            os.stat(_existing_prefix(path_b)).st_dev
    except OSError:
        return False


def estimate_bank_bytes(num_images, num_layers, patch_grid, dim, itemsize=4):
    """预估 bank .pth 的字节数（patch 部分 + cls 部分）。"""
    per_img = int(patch_grid) * int(dim) * int(itemsize) + int(dim) * int(itemsize)
    return int(num_images) * int(num_layers) * per_img


def release_all():
    """丢掉模块持有的一切映射引用（删临时文件之前必须调用）。"""
    del _ACTIVE_SINKS[:]
    gc.collect()


def remove_tmp_dir(tmp_dir, keep=False, verbose=True):
    """释放映射并删除临时目录。"""
    release_all()
    if not tmp_dir:
        return
    if keep:
        if verbose:
            print("[bank] --keep_tmp: 临时文件保留在 %s" % tmp_dir)
        return
    shutil.rmtree(tmp_dir, ignore_errors=True)
    if verbose and os.path.isdir(tmp_dir):
        print("[bank] 提示: 临时目录未能删除（可能仍有映射引用），可手动删除: %s"
              % tmp_dir)


class DiskMultiLayerBank:
    """把 ``[B, L, C]`` / ``[B, C]`` 特征顺序追加到磁盘映射文件。

    用法::

        sink = DiskMultiLayerBank(tmp_dir, num_layers=4, total_images=8659)
        for each batch:
            for li, (patch_tok, cls_tok) in enumerate(inter):
                sink.append(li, patch_tok, cls_tok)
        cls_banks, patch_banks = sink.tensors()
        torch.save({...}, bank_path)
        remove_tmp_dir(tmp_dir)
    """

    def __init__(self, tmp_dir, num_layers, total_images, verbose=True):
        self.tmp_dir = tmp_dir
        self.num_layers = int(num_layers)
        self.total = int(total_images)
        self.verbose = verbose
        os.makedirs(self.tmp_dir, exist_ok=True)

        self._cls = [None] * self.num_layers      # list[Tensor[N, C]]
        self._patch = [None] * self.num_layers    # list[Tensor[N, L, C]]
        self._paths = []
        self._n = [0] * self.num_layers
        self.dtype = None
        _ACTIVE_SINKS.append(self)

    # ------------------------------------------------------------------
    @property
    def written(self):
        """已写入的行数（各层同步推进，取第一层即可）。"""
        return self._n[0]

    def bytes_allocated(self):
        return sum(os.path.getsize(p) for p in self._paths if os.path.exists(p))

    def _alloc_layer(self, li, patch_shape, cls_shape, dtype):
        itemsize = torch.empty(0, dtype=dtype).element_size()
        p_numel = self.total * int(patch_shape[0]) * int(patch_shape[1])
        c_numel = self.total * int(cls_shape[0])
        p_path = os.path.join(self.tmp_dir, "layer%02d_patch.dat" % li)
        c_path = os.path.join(self.tmp_dir, "layer%02d_cls.dat" % li)

        for path, numel in ((p_path, p_numel), (c_path, c_numel)):
            with open(path, "wb") as fh:
                fh.truncate(numel * itemsize)     # 稀疏文件，不占常驻内存
            self._paths.append(path)

        self._patch[li] = torch.from_file(
            p_path, shared=True, size=p_numel, dtype=dtype
        ).view(self.total, *patch_shape)
        self._cls[li] = torch.from_file(
            c_path, shared=True, size=c_numel, dtype=dtype
        ).view(self.total, *cls_shape)

        if self.verbose:
            print("[bank] layer %d 落盘分配: patch %s (%s) | cls %s (%s)" % (
                li, tuple(self._patch[li].shape),
                human_bytes(p_numel * itemsize),
                tuple(self._cls[li].shape),
                human_bytes(c_numel * itemsize)))

    # ------------------------------------------------------------------
    @torch.no_grad()
    def append(self, li, patch_tok, cls_tok):
        """写入一个 batch（patch_tok 可以是 CPU 或 CUDA 张量）。"""
        if patch_tok.shape[0] != cls_tok.shape[0]:
            raise RuntimeError("patch/cls batch 大小不一致: %s vs %s"
                               % (tuple(patch_tok.shape), tuple(cls_tok.shape)))

        if self._patch[li] is None:
            if self.dtype is None:
                self.dtype = patch_tok.dtype
            elif patch_tok.dtype != self.dtype:
                raise RuntimeError("各层 dtype 不一致: %s vs %s"
                                   % (patch_tok.dtype, self.dtype))
            self._alloc_layer(li, tuple(patch_tok.shape[1:]),
                              tuple(cls_tok.shape[1:]), patch_tok.dtype)
        elif patch_tok.dtype != self.dtype:
            raise RuntimeError("同一层 dtype 变化: %s vs %s"
                               % (patch_tok.dtype, self.dtype))

        n = self._n[li]
        b = int(patch_tok.shape[0])
        if n + b > self.total:
            raise RuntimeError(
                "写入行数超出预估: 已写 %d + 本批 %d > 总图数 %d（如果换了数据集，"
                "请确认 len(dataset) 与实际迭代行数一致）" % (n, b, self.total))

        # D2H（或 CPU->文件）一次拷贝，直接落盘
        self._patch[li][n:n + b].copy_(patch_tok)
        self._cls[li][n:n + b].copy_(cls_tok)
        self._n[li] = n + b

    # ------------------------------------------------------------------
    def tensors(self):
        """返回 ``(cls_banks, patch_banks)``（均为磁盘映射张量，torch.save 零拷贝）。"""
        if self._patch[0] is None:
            raise RuntimeError("bank 是空的：没有写入任何 batch")
        for li in range(self.num_layers):
            if self._n[li] != self.total:
                raise RuntimeError(
                    "layer %d 只写了 %d 行 / 预期 %d 行"
                    % (li, self._n[li], self.total))
        return list(self._cls), list(self._patch)

    def close(self, remove=True):
        self._cls = [None] * self.num_layers
        self._patch = [None] * self.num_layers
        if self in _ACTIVE_SINKS:
            _ACTIVE_SINKS.remove(self)
        gc.collect()
        if remove:
            shutil.rmtree(self.tmp_dir, ignore_errors=True)
