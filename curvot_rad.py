"""
CurvOT-RAD evaluation script.

Drop-in replacement for ``rad_mvtec_visa_3dadam.py`` that keeps the same bank
and the same multi-layer fusion, but makes the two retrieval/scoring mechanisms
switchable so the ablation table can be produced with a single code path.

  --use_aptr            1 : anomaly-resilient partial-transport re-ranking (innovation 1)
  --use_lonm            1 : local osculating normal manifold scoring      (innovation 2)
  --lonm_no_quadratic   1 : tangent-only (local PCA) scoring  -> SubspaceAD-style control
  --use_transport_window1 : replace RAD's fixed rho window by transport-guided centres

Ablation table produced by this script
--------------------------------------
  RAD                      aptr=0 lonm=0
  RAD + full OT            aptr=1 ot_mass=1.0 lonm=0     (m=1 control)
  RAD + partial OT         aptr=1 lonm=0                 (innovation 1)
  RAD + local PCA          aptr=0 lonm=1 quadratic=0     (first-order control)
  RAD + quadratic          aptr=0 lonm=1 quadratic=1     (innovation 2)
  CurvOT-RAD               aptr=1 lonm=1 transport=1     (full model)
"""

import argparse
import ast
import json
import os
import random
import time
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import functional as TF

from sklearn.metrics import roc_auc_score, average_precision_score

from dataset import get_data_transforms, MVTecDataset, RealIADDataset, resolve_item_list
from dinov3.hub.backbones import load_dinov3_model
from utils import compute_pro, f1_score_max, get_gaussian_kernel

from rad_mvtec_visa_3dadam import (
    GPUPatchBankCache, load_bank, validate_bank_consistency,
    build_pos_context, patch_scores_positional, _grid_hw,
)
import curvot_utils as cu
import baseline_scores as bsl

warnings.filterwarnings('ignore')


# ============================================================================
def setup_seed(seed=1):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ctx_from_pos_context(pc):
    """Adapter: reuse RAD's original window tables as a candidate-gather context."""
    return {
        'H': pc['H'], 'W': pc['W'], 'r': pc['r'], 'L': pc['L'],
        'Kp': len(pc['offsets']),
        'idx': torch.stack([t[0] for t in pc['tables']], dim=1),
        'valid': torch.stack([t[1] for t in pc['tables']], dim=1),
        'offsets': pc['offsets'],
    }


# ---------------------------------------------------------------------------
# DAD (=--agg dad): bank-side null tables for the dimensionality-aware
# deviation score.  kdist(o) and LID(o) depend only on the NORMAL bank of the
# class, never on a test image, so they are computed once per class and cached
# in RAM (and on disk between runs).
# ---------------------------------------------------------------------------
_DAD_CACHE = {}


def _dad_stats(patch_banks, num_layers, dad_rows, ctx, dad_k, dad_metric,
               dad_refn, dad_dir, item, device, verbose=True, normalize=True):
    h, w, r = int(ctx['H']), int(ctx['W']), int(ctx['r'])
    # The block signature is (first row, last row, size); first+size alone would
    # collide between two equally-sized classes at different offsets, which is
    # exactly the confusion that produced the [dad-miss]=602112 run.
    key = (str(item), int(h), int(w), int(r), int(dad_k), str(dad_metric),
           int(dad_refn), bool(normalize),
           int(dad_rows[0]) if dad_rows is not None else -1,
           int(dad_rows[-1]) if dad_rows is not None else -1,
           int(dad_rows.numel()) if dad_rows is not None else -1)
    if key in _DAD_CACHE:
        return _DAD_CACHE[key]
    fname = None
    if dad_dir:
        os.makedirs(dad_dir, exist_ok=True)
        off = int(dad_rows[0]) if dad_rows is not None else -1
        fname = os.path.join(
            dad_dir, f'dad_{item}_o{off}_h{h}w{w}_r{r}_k{dad_k}_{dad_metric}_'
                     f'n{dad_refn}{"_raw" if not normalize else ""}.pt')
    if fname and os.path.exists(fname):
        blob = torch.load(fname, map_location='cpu')
        out = {int(p): dict(rk=blob['rk'][p].to(device),
                            lid=blob['lid'][p].to(device),
                            rk_med=blob['rk_med'][p].to(device),
                            lid_mean=float(blob['lid_mean'][p]))
               for p in range(num_layers)}
        _DAD_CACHE[key] = out
        if verbose:
            print(f'[dad] {item}: loaded bank-side kdist/LID cache {fname}',
                  flush=True)
        return out

    out = {}
    rk_list, lid_list, med_list, lmean_list = [], [], [], []
    t_all = time.time()
    for p in range(num_layers):
        rk, lid, nm = cu.bank_self_knn_stats(
            patch_banks[p], ctx, k=dad_k, device=device, refn=dad_refn,
            metric=dad_metric, row_sel=dad_rows, normalize=normalize)
        v = rk > 0
        frac = float(v.to(torch.float32).mean())
        if frac < 0.95:
            raise SystemExit(
                f'\n[dad] {item} layer#{p}: only {frac * 100:.2f}% of bank patches '
                f'have a valid kdist.  This is the signature of a broken scale, '
                f'not of a degenerate dataset: unnormalised bank rows make '
                f'2-2*cos negative, clamp to 0 and invalidate every row, after '
                f'which rk_med falls back to a constant 1.0 and every DAO ratio '
                f'becomes 1.  Check --normalize_bank / the bank dtype; do NOT '
                f'let this run continue.\n')
        rk_med = rk.masked_fill(~v, float('nan')).nanmedian(dim=0).values
        rk_med = torch.nan_to_num(rk_med, nan=1.0).clamp_min(1e-6)
        lid_mean = float(lid[v].mean()) if bool(v.any()) else 1.0
        out[p] = dict(rk=rk.to(device), lid=lid.to(device),
                      rk_med=rk_med.to(device), lid_mean=lid_mean)
        rk_list.append(rk)
        lid_list.append(lid)
        med_list.append(rk_med)
        lmean_list.append(lid_mean)
        if verbose:
            print(f'[dad] {item} layer#{p}: N={rk.shape[0]} '
                  f'valid={frac * 100:.1f}% '
                  f'kdist_med={float(rk_med.median()):.4f} '
                  f'LID_med={float(lid[v].median()):.3f} '
                  f'nmiss={int(nm.sum())} [{time.time() - t_all:.1f}s]',
                  flush=True)
    _DAD_CACHE[key] = out
    if fname:
        torch.save(dict(rk=torch.stack(rk_list), lid=torch.stack(lid_list),
                        rk_med=torch.stack(med_list), lid_mean=lmean_list),
                   fname)
        if verbose:
            print(f'[dad] {item}: wrote {fname} [{time.time() - t_all:.1f}s]',
                  flush=True)
    return out


def _count_train_images(root):
    """How many training images ImageFolder/data_path/<item>/train would list."""
    n = 0
    for _r, _d, files in os.walk(root):
        n += sum(1 for f in files if f.lower().endswith(
            ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp')))
    return n


# ---------------------------------------------------------------------------
# Recovering the bank's per-class row blocks
# ---------------------------------------------------------------------------
# The checkpoint stores ONLY {layers, cls_banks, patch_banks} -- no item order.
# But the single global bank is laid out as one contiguous row block per class,
# and DAD's bank-side null (kdist / LID of every normal patch) has to be built
# per class: a patch of class A at position p has its true neighbours among
# class A's images at position p.  Pooling classes would inflate kdist(o) and
# silently deflate the DAO ratio.
#
# The tempting shortcut -- "use the order resolve_item_list returns at eval
# time" -- is WRONG: a subset run (--item_list grid) resolves to ['grid'], so
# grid's offset computes as 0 while its real block starts at 280.  That bug
# produced [dad-miss] = 602112, i.e. every lookup falling back.
#
# So the order is *recovered and verified*, never assumed.  Block SIZES come
# from disk (cheap: count files).  The ORDER is chosen by CLS block purity --
# the fraction of rows whose nearest CLS neighbour in the bank lands in the
# same candidate block.  Same-class images have near-identical global
# embeddings, so the true partition scores ~1.0 while any wrong one does not.
# Measured on the MVTec bank: alphabetical 0.7716 vs true order 1.0000.  The
# check is therefore decisive, and it is re-run on every launch, so a bank
# rebuilt with a different order at the same path cannot pass unnoticed.
_MVTEC_CANON = ['carpet', 'grid', 'leather', 'tile', 'wood',
                'bottle', 'cable', 'capsule', 'hazelnut', 'metal_nut',
                'pill', 'screw', 'toothbrush', 'transistor', 'zipper']
_BANK_ORDER_CACHE = {}


def _train_count(data_path, item, layout='folder'):
    # # --- RI-PATCH H1
    # Real-IAD keeps its split in realiad_jsons/realiad_jsons/<item>.json --
    # there is no <data_path>/<item>/train on disk.  The old folder-shaped
    # path therefore returned 0 for every class, `_bank_class_order` summed
    # those zeros into n_tot = 0, scored nan purity and raised SystemExit
    # before any Real-IAD run could start.  The json is the only place the
    # count exists.
    if layout == 'realiad':
        js = os.path.join(data_path, 'realiad_jsons', 'realiad_jsons',
                          item + '.json')
        try:
            with open(js) as fh:
                return len(json.load(fh).get('train') or [])
        except Exception:
            return 0
    # unchanged for folder layouts: this is the same single join the old two
    # lines produced, so MVTec / VisA / 3D-ADAM row ranges cannot move.
    root = os.path.join(data_path, item, 'train', 'good')
    return _count_train_images(root)


def _cls_block_purity(sizes, cls_banks, device='cuda:0', n_sub=1500, seed=0):
    """Thin wrapper: the scorer lives in curvot_utils so it can be unit-tested."""
    return cu.cls_block_purity(sizes, cls_banks, device=device, n_sub=n_sub,
                               seed=seed)


def _bank_order_from_log(pattern='logs/bankbuild.log'):
    """Opportunistically read candidate orders out of the build log.

    Every `item_list:` line is one bank that was built, so this returns a LIST --
    the log holds MVTec first and VisA second, and keeping only the first would
    silently drop the VisA candidate (the MVTec one is rejected for a VisA class
    set, leaving the coincidence that VisA's disk order equals its build order as
    the only support).
    """
    out = []
    try:
        with open(pattern) as fh:
            for line in fh:
                if line.strip().startswith('item_list:'):
                    try:
                        od = ast.literal_eval(line.split('item_list:', 1)[1].strip())
                        if isinstance(od, (list, tuple)):
                            out.append([str(x) for x in od])
                    except Exception:
                        pass
    except Exception:
        pass
    return out


def _bank_class_order(data_path, layout, bank_path, item_list, cls_banks,
                      device, explicit=None, verbose=True):
    """Resolve (and verify) the full per-class row blocks of the global bank.

    Returns (order, counts, starts, rows_of) over ALL classes on disk, not just
    the requested --item_list subset, so that a subset run still gets the true
    global row ranges.
    """
    ckey = (os.path.abspath(bank_path) if bank_path else '', str(layout),
            os.path.abspath(data_path))
    if ckey in _BANK_ORDER_CACHE:
        return _BANK_ORDER_CACHE[ckey]

    disk_items = resolve_item_list(data_path, [], layout=layout,
                                   print_fn=lambda *a, **k: None)
    counts = {it: _train_count(data_path, it, layout) for it in disk_items}
    total = int(sum(counts.values()))

    cands = []

    def _add(name, order):
        if not order:
            return
        order = [str(x) for x in order]
        if len(order) == len(disk_items) and set(order) == set(disk_items):
            cands.append((name, order))

    _add('--bank_item_order', explicit)
    sidecar = (bank_path + '.item_order.json') if bank_path else None
    if sidecar and os.path.exists(sidecar):
        try:
            with open(sidecar) as fh:
                _add('sidecar', json.load(fh).get('item_order'))
        except Exception as exc:
            print(f'[bank-order] 侧车解析失败 {sidecar}: {exc}')
    _add('disk-probe (alphabetical)', disk_items)
    _add('mvtec canonical', _MVTEC_CANON)
    for _i, _od in enumerate(_bank_order_from_log()):
        _add(f'bankbuild log #{_i}', _od)

    seen, uniq = set(), []
    for name, order in cands:
        key = tuple(order)
        if key not in seen:
            seen.add(key)
            uniq.append((name, order))

    if verbose:
        print(f'[bank-order] 需要 {total} 行类块布局，bank 实际 {cls_banks[0].shape[0]} 行；'
              f'{len(uniq)} 个候选顺序待验')
    scored = []
    for name, order in uniq:
        purity = _cls_block_purity([counts[c] for c in order], cls_banks, device)
        scored.append((purity, name, order))
        if verbose:
            print(f'[bank-order]   {name:28s} CLS 块纯度 = {purity:.4f}')
    scored.sort(key=lambda t: -t[0])
    purity, best_name, best_order = scored[0]

    if not (purity >= 0.999):
        # # --- RI-PATCH H2
        # A DECLARED order is a label, not an inference.  When the caller passes
        # --bank_item_order and that order is also the best-scoring candidate,
        # a purity below 1.0 tells us about the data, not about the label.
        # Everything the partition feeds -- few-shot row ranges, DAD blocks --
        # is unused by a full-data run, whose retrieval is global, so a warning
        # is the right level here.  Without this, one diagnostic could kill the
        # whole 30-class table.
        _declared_ok = bool(explicit) and list(explicit) == list(best_order)
        if not _declared_ok:
            raise SystemExit(
                f'\n[bank-order] 无法确定 bank 的类块顺序：最高纯度仅 {purity:.4f}'
                f'（{best_name}），应当 ≈1.0。\n'
                f'  说明候选顺序都不是真实布局；请用 --bank_item_order 显式给出\n'
                f'  构建该 bank 时传给 --item_list 的顺序。\n'
                f'  已尝试: ' + ', '.join(f'{n}({p:.4f})' for p, n, _ in scored) + '\n'
            )
        print(f'[bank-order] WARNING 纯度 {purity:.4f} < 0.999，但该顺序由 '
              f'--bank_item_order 显式声明且为最高分候选 -> 继续 '
              f'(full-data 检索是全局的，类块划分只影响 few-shot/DAD)')
    if verbose:
        print(f'[bank-order] 采用 {best_name} (纯度 {purity:.4f})')

    starts = np.concatenate(
        [[0], np.cumsum([counts[c] for c in best_order])]).astype(np.int64)
    rows_of = {c: (int(starts[i]), int(starts[i + 1]))
               for i, c in enumerate(best_order)}

    if sidecar and not os.path.exists(sidecar):
        try:
            with open(sidecar, 'w') as fh:
                json.dump(dict(item_order=best_order, counts=counts,
                               purity=float(purity), source=best_name), fh,
                          indent=1)
            if verbose:
                print(f'[bank-order] 判决写入侧车 {sidecar}')
        except Exception as exc:
            print(f'[bank-order] 侧车写入失败 {sidecar}: {exc}')

    missing = [it for it in item_list if it not in rows_of]
    if missing:
        raise SystemExit(f'[bank-order] --item_list 中的类别不在 bank 里: {missing}')
    out = (best_order, counts, starts, rows_of)
    _BANK_ORDER_CACHE[ckey] = out
    return out


def apply_pose(x, angle_deg=0.0, shift_px=0, is_mask=False):
    """Rotate / translate a 448x448 tensor (image already normalised)."""
    if angle_deg == 0.0 and shift_px == 0:
        return x
    h, w = x.shape[-2], x.shape[-1]
    tx = float(shift_px) / w
    ty = float(shift_px) / h
    return TF.affine(
        x, angle=float(angle_deg), translate=(tx, ty), scale=1.0, shear=0.0,
        interpolation=TF.InterpolationMode.NEAREST if is_mask
        else TF.InterpolationMode.BILINEAR,
        fill=0.0,
    )


# ============================================================================
@torch.no_grad()
def _parse_band_w(s):
    """Parse --band_w  a,h,v,d  (four comma-separated floats)."""
    vals = [float(v) for v in str(s).replace(' ', '').split(',') if v != '']
    if len(vals) != 4:
        raise SystemExit('--band_w needs four comma-separated numbers '
                         'a,h,v,d (e.g. 0,1,1,1 for a pure high-pass)')
    return tuple(vals)


def eval_curvot(
    encoder, layer_indices, cls_banks, patch_banks, test_dataloader, device,
    k_image=150, resize_mask=256, max_ratio=0.01,
    use_positional_bank=True, pos_radius=1,
    layer_weights=None, bank_cache=None,
    knn_chunk_imgs=32, global_chunk_imgs=64,
    use_aptr=False, ot_k0=128, ot_kr=24, ot_mass=0.8, ot_reg=0.05,
    ot_lam_pos=0.05, ot_iter=120, coarse_grid=7,
    use_lonm=False, lonm_M=24, lonm_dmax=4, lonm_energy=0.9, lonm_ridge=1e-4,
    lonm_trust=0.9, lonm_quadratic=True, lonm_ridge_rel=0.0,
    lonm_res_rank=0, lonm_curv_gain=1.0, lonm_norm=False, lonm_scale_q=0.9,
    lonm_radial=False, lonm_norm_mode='ratio', lonm_scale_beta=0.0,
    lonm_maha=False, lonm_maha_ridge=0.1, lonm_maha_rank=0, anc_beta=0.0,
    lonm_fast=False,
    lonm_unbounded=False, lonm_clipstat=False,
    baseline='none', baseline_d=64, baseline_power=2.0,
    baseline_padim_shrink='published', baseline_padim_ridge_rel=0.0,
    baseline_padim_ridge_abs=0.01, baseline_padim_sqrt=1,
    baseline_pc_ratio=0.01, baseline_pc_k=9, baseline_pc_rw=1.0,
    baseline_pc_cap=2000, baseline_pc_proj=64, baseline_pc_subcap=120000,
    baseline_anoco_lambda=1.0, baseline_anoco_anchor=True,
    use_transport_window=False, max_shift=3,
    use_unmatched=False, unmatched_w=0.05, layer_fuse='mean',
    use_centroid=0,
    img_cal='none', img_cal_q=0.5,
    img_cal_ref='none', img_cal_alpha=1.0,
    img_cal_scope='batch', img_cal_stats=0,
    img_cal_red='median', img_cal_red_q=0.9,
    img_cal_scale_q=0.95,
    agg='min', ia_K=0,
    dad_k=8, dad_lid='est', dad_comb='dao', dad_metric='l2', dad_refn=0,
    dad_dir='', dad_rows=None, item_name='', dad_post=0.0, dad_floor=0.0,
    cfar_comb='logp', cfar_post=0.0,
    band_mode='off', band_w=(1.0, 1.0, 1.0, 1.0), band_beta=0.0,
    pose_angle=0.0, pose_shift=0, limit_imgs=0,
    verbose=True, log_every=40,
    allowed_idx=None,
):
    encoder.eval()
    device = torch.device(device)
    num_layers = len(layer_indices)

    if bank_cache is None:
        total = sum(b.numel() * b.element_size() for b in patch_banks)
        bank_cache = GPUPatchBankCache(patch_banks, device, budget_bytes=total,
                                       normalize=True, verbose=verbose)

    cls_bank_all = F.normalize(cls_banks[-1].to(device), dim=-1)
    if allowed_idx is None:
        cls_ref = cls_bank_all
    else:                                  # few-shot: only k rows per class
        cls_ref = cls_bank_all[allowed_idx]
    N = cls_ref.shape[0]

    if layer_weights is None or len(layer_weights) != num_layers:
        layer_weights = [1.0 / num_layers] * num_layers
    gaussian_kernel = get_gaussian_kernel(kernel_size=5, sigma=1.0).to(device)

    # ---- matched-feature baseline reference (offline part) ----------------
    # Everything here is inert unless --baseline is set.  The reference is
    # fitted once per category from the SAME bank block the other rows'
    # retrieval draws on, so the normal memory is matched; the per-image score
    # then goes through the identical map / pooling / metric pipeline.
    baseline_ref = None
    if baseline != 'none':
        if dad_rows is None or int(dad_rows.numel()) == 0:
            raise SystemExit(
                '[baseline] the offline reference needs this category\'s bank '
                'block; _bank_class_order did not resolve it for '
                f'{item_name!r}.  Check the [bank-order] lines above.')
        if baseline in ('subspacead', 'padim'):
            # One category block at a time: the VisA blocks are large enough
            # that holding all four layers at once would double the cache.
            baseline_ref = []
            for li in range(num_layers):
                blk = bank_cache.get(li, dad_rows).detach()
                if baseline == 'subspacead':
                    baseline_ref += bsl.fit_subspacead([blk], None, None,
                                                       d=baseline_d)
                else:
                    baseline_ref.append(bsl.fit_padim(
                        blk, d=baseline_d, shrink=baseline_padim_shrink,
                        ridge_rel=baseline_padim_ridge_rel,
                        ridge_abs=baseline_padim_ridge_abs))
                    print(f'[baseline] padim L{li}: shrink='
                          f'{baseline_padim_shrink} '
                          f'delta={baseline_ref[-1]["shrink_delta"]:.4f} '
                          f'cond(mean/med)={baseline_ref[-1]["cond_mean"]:.3e}'
                          f'/{baseline_ref[-1]["cond_med"]:.3e}', flush=True)
                del blk
        elif baseline == 'patchcore':
            baseline_ref = []
            for li in range(num_layers):
                blk = bank_cache.get(li, dad_rows)
                feat = blk.reshape(-1, blk.shape[-1])
                mm = int(round(baseline_pc_ratio * feat.shape[0]))
                if baseline_pc_cap > 0:
                    mm = min(mm, baseline_pc_cap)
                mm = max(1, mm)
                baseline_ref.append(
                    bsl.patchcore_coreset(feat, mm, seed=0,
                                          proj_dim=baseline_pc_proj,
                                          sub_cap=baseline_pc_subcap,
                                          verbose=verbose))
                print(f'[baseline] patchcore L{li}: coreset {mm} of '
                      f'{int(feat.shape[0])} patches '
                      f'({100.0 * mm / max(1, feat.shape[0]):.3f}%), '
                      f'proj_dim={baseline_pc_proj}, sub_cap={baseline_pc_subcap}',
                      flush=True)
                del blk, feat
        elif baseline == 'anoco':
            baseline_ref = 'pool'
        else:
            raise SystemExit(f'[baseline] unknown --baseline {baseline!r}')
        if verbose:
            print(f'[baseline] {baseline}: reference fitted on '
                  f'{int(dad_rows.numel())} normal bank rows of {item_name!r}',
                  flush=True)

    gt_list_px, pr_list_px, gt_list_sp, pr_list_sp = [], [], [], []
    norm_scores = []
    pos_ctx_cache = {}
    dad_tabs = None
    dad_stat = {}
    # radius-clipping diagnostics for THIS class (only filled when
    # --lonm_clipstat 1); see lonm_score's docstring.
    clip_stat = {} if lonm_clipstat else None
    # global bank index of the class block's first row: the bank-side tables are
    # class-local, the retrieval rows are global, and the two coincide only when
    # the evaluated class is the entire bank.
    dad_base = int(dad_rows.min()) if dad_rows is not None and dad_rows.numel() else 0
    n_seen = 0
    t0 = time.time()

    # ---- per-image null calibration bookkeeping -----------------------------
    # The pooled pixel-AP is computed PER CLASS (main() calls this function once
    # per item, and gt_px/pr_px are built inside), so the reference level that
    # `img_cal_ref` subtracts must also be a CLASS-level statistic.  With the
    # default `img_cal_scope='batch'` it is instead the median of one batch of
    # `batch_size` images, which is an arbitrary subset of the class: subtracting
    # a different constant in every batch injects a between-batch shift into a
    # ranking that is evaluated across the whole class.  `img_cal_scope='class'`
    # buffers the fused maps for the class and resolves the reference once, so
    # that a class whose images already share a normal level really is returned
    # untouched (a uniform shift is a no-op for a pooled ranking) -- the property
    # the mechanism is supposed to have.
    defer_class_fuse = layer_fuse in ('geo_cls', 'zfuse_cls', 'gls')
    defer_class_cal = (
        (not defer_class_fuse)
        and img_cal in ('matched', 'matched_z', 'matched_sg', 'matched_sgo')
        and img_cal_scope == 'class'
        and (img_cal_ref != 'none' or img_cal in ('matched_sg', 'matched_sgo')))
    cls_cal = dict(fused=[], offs=[], scs=[], gt=[], label=[], ghgw=[],
                   lay=[], u=[])
    cal_stat_buf = []
    cal_digest, cal_digest_n = [], []

    def _calib_params_batch(fused_b, u_b_list):
        """(offset [B,1], scale [B,1], subtract_off, gate_scale) for a block.

        Shared by the in-loop path and the class-end fusion path so the two can
        never drift apart -- the offset/scale split is exactly the kind of change
        that silently breaks one caller while the other keeps passing.
        """
        so = img_cal in ('matched', 'matched_z', 'matched_sgo')
        gs = img_cal in ('matched_sg', 'matched_sgo')
        offs, scs = [], []
        for b in range(fused_b.shape[0]):
            row = fused_b[b:b + 1]
            u = u_b_list[b]
            if img_cal == 'matched_z':
                o, s = cu.matched_null_stats(row, u, img_cal_q)
            else:
                o = cu.matched_null_offset(row, u, img_cal_q,
                                           img_cal_red, img_cal_red_q)
                s = (cu.matched_null_scale(row, u, img_cal_q, img_cal_scale_q)
                     if img_cal in ('matched_s', 'matched_sg', 'matched_sgo')
                     else torch.ones_like(o))
            offs.append(o)
            scs.append(s)
        return torch.cat(offs, dim=0), torch.cat(scs, dim=0), so, gs

    def _apply_calib(fused_b, offs_t, scs_t, subtract_off, gate_scale):
        if gate_scale:
            scs_t = torch.maximum(scs_t, scs_t.median())
        if subtract_off:
            if img_cal_ref == 'median':
                offs_t = (offs_t - offs_t.median()).clamp_min(0.0)
            elif img_cal_ref == 'q':
                offs_t = (offs_t - torch.quantile(
                    offs_t, float(img_cal_q))).clamp_min(0.0)
            if float(img_cal_alpha) != 1.0:
                offs_t = float(img_cal_alpha) * offs_t
        else:
            offs_t = torch.zeros_like(offs_t)
        return (fused_b - offs_t) / scs_t

    def _emit(fused_b, gt_b, label_b, gh, gw):
        """map/score emission shared by the in-loop and deferred paths."""
        Bb = fused_b.shape[0]
        if img_cal_stats:
            # label-free digest of the CALIBRATED patch scores.  Comparing it is
            # how the scope tests assert bit-equality without depending on the
            # metric (a truncated test prefix can be single-class, which makes
            # AUROC undefined) and without needing a memory-hungry giant batch.
            fd = fused_b.detach().double()
            cal_digest.append(float(fd.sum()))
            cal_digest.append(float((fd * fd).sum()))
            cal_digest_n.append(int(fused_b.numel()))
        patch_maps = fused_b.view(Bb, 1, gh, gw)
        if band_mode != 'off':
            # BAND-PASS PROBE.  `_emit` is the single point both the in-loop and
            # the deferred fusion paths funnel through, so putting the filter
            # here means every row is filtered identically no matter which
            # --layer_fuse branch produced the map.
            #
            # It is applied to the PATCH GRID and BEFORE the evaluator upsamples
            # to `resize_mask`, which is deliberately the opposite of the naive
            # "post-process the 256 x 256 heat map": filtering after the
            # bilinear upsample would touch only interpolation artefacts, while
            # filtering the 28 x 28 grid acts on the frequencies the network
            # actually produced.  Neutral settings return `patch_maps` itself,
            # so old rows stay bit-identical (test_dad.py asserts this with `is`).
            patch_maps = cu.grid_band_pass(patch_maps, gh, gw, mode=band_mode,
                                           band_w=band_w, beta=band_beta)
        anomaly_map = F.interpolate(patch_maps, size=resize_mask, mode='bilinear',
                                    align_corners=False)
        gt_r = F.interpolate(gt_b, size=resize_mask, mode='nearest')
        anomaly_map = gaussian_kernel(anomaly_map)
        gt_r = gt_r.bool()
        if gt_r.shape[1] > 1:
            gt_r = torch.max(gt_r, dim=1, keepdim=True)[0]

        gt_list_px.append(gt_r)
        pr_list_px.append(anomaly_map)
        gt_list_sp.append(label_b)
        if max_ratio == 0:
            sp_score = torch.max(anomaly_map.flatten(1), dim=1)[0]
        else:
            amap_flat = anomaly_map.flatten(1)
            kk = max(1, int(amap_flat.shape[1] * max_ratio))
            sp_score = torch.sort(amap_flat, dim=1, descending=True)[0][:, :kk].mean(dim=1)
        pr_list_sp.append(sp_score)

        lab = label_b.flatten()
        for i in range(Bb):
            if lab[i].item() == 0:
                norm_scores.append(float(sp_score[i]))

    for img, gt, label, img_path in test_dataloader:
        img = img.to(device)
        gt = gt.to(device)
        if pose_angle != 0.0 or pose_shift != 0:
            img = apply_pose(img, pose_angle, pose_shift, is_mask=False)
            gt = apply_pose(gt, pose_angle, pose_shift, is_mask=True)

        inter = encoder.get_intermediate_layers(
            img, n=layer_indices, return_class_token=True, reshape=False, norm=True)
        patch_list = [p for (p, _) in inter]
        cls_list = [c for (_, c) in inter]

        B, L, C = patch_list[0].shape
        h, w = _grid_hw(encoder, img, L)

        cls_x_norm = F.normalize(cls_list[-1], dim=-1)
        sim_img = torch.matmul(cls_x_norm, cls_ref.t())                   # [B,N']
        K0 = min(int(ot_k0) if use_aptr else int(k_image), N)
        _, topk_idx = torch.topk(sim_img, K0, dim=-1)                     # [B,K0]
        if allowed_idx is None:
            short_idx_cpu = topk_idx.detach().cpu()
        else:
            short_idx_cpu = allowed_idx[topk_idx].detach().cpu()

        last_li = num_layers - 1
        patch_scores_batch = []
        u_list = []

        for b in range(B):
            rows0 = short_idx_cpu[b]                                      # [K0]
            tp_ctx = None
            u_map = None
            if use_aptr:
                ref_coarse = cu.grid_pool_tokens(
                    bank_cache.get(last_li, rows0), h, w, coarse_grid)     # [K0,G^2,C]
                q_coarse = cu.grid_pool_tokens(
                    F.normalize(patch_list[last_li][b], dim=-1).unsqueeze(0),
                    h, w, coarse_grid)[0]
                cost, plan, match = cu.partial_ot_rerank(
                    q_coarse, ref_coarse, reg=ot_reg, mass=ot_mass,
                    lam_pos=ot_lam_pos, n_iter=ot_iter,
                    return_plan=True, return_match=True)
                Kr = min(int(ot_kr), K0)
                sel = torch.topk(-cost, Kr).indices
                rows = rows0[sel.detach().cpu()]
                if use_transport_window:
                    G = coarse_grid
                    plan_agg = plan[sel].sum(dim=0)                        # [G^2,G^2]
                    tp_ctx = cu.transport_offset_field(plan_agg, G, h, w, max_shift)
                if use_unmatched or img_cal in ('matched', 'matched_z',
                                                'matched_pit', 'matched_s',
                                                'matched_sg', 'matched_sgo'):
                    u_map = cu.unmatched_mass_field(plan[sel], coarse_grid, h, w)
            else:
                rows = rows0

            # ---- candidate window: RAD-fixed vs transport-guided ----
            if use_lonm or use_transport_window or use_centroid \
                    or baseline == 'anoco':
                if use_transport_window and tp_ctx is not None:
                    cand_ctx = cu.build_cand_context(h, w, pos_radius, device,
                                                     center_rc=tp_ctx)
                else:
                    key = (h, w, pos_radius)
                    if key not in pos_ctx_cache:
                        pos_ctx_cache[key] = build_pos_context(h, w, pos_radius, device)
                    cand_ctx = ctx_from_pos_context(pos_ctx_cache[key])
            else:
                key = (h, w, pos_radius)
                if key not in pos_ctx_cache:
                    pos_ctx_cache[key] = build_pos_context(h, w, pos_radius, device)
                # `patch_scores_positional` wants RAD's raw table list; the
                # order-statistic and DAD aggregations walk the (image, offset)
                # axes separately and therefore need the idx/valid layout.
                # --dad_post / --cfar_post need it too: they rewrite whatever
                # score the configured path produced, so they can be combined
                # with agg='min', and a raw table list would fail inside them.
                cand_ctx = (ctx_from_pos_context(pos_ctx_cache[key])
                            if (agg in ('ia', 'dad', 'cfar')
                                or dad_post != 0.0 or cfar_post != 0.0)
                            else pos_ctx_cache[key])

            if (agg in ('dad', 'cfar') or dad_post != 0.0 or cfar_post != 0.0) \
                    and dad_tabs is None:
                if dad_rows is None:
                    raise SystemExit('--agg dad / --agg cfar / --dad_post need the per-class '
                                     'bank row range')
                dad_tabs = _dad_stats(patch_banks, num_layers, dad_rows, cand_ctx,
                                      dad_k, dad_metric, dad_refn, dad_dir,
                                      item_name, device, verbose=verbose,
                                      normalize=bool(getattr(bank_cache,
                                                             'normalize', True)))

            scores_per_layer = []
            for li in range(num_layers):
                q_feat = F.normalize(patch_list[li][b], dim=-1)           # [L,C]
                # The retrieved-neighbour block is only fetched when the
                # configured readout actually consumes it.  This is a no-op for
                # the default path (baseline == 'none' always needs it) and it
                # keeps SubspaceAD / PaDiM / PatchCore from paying a bank fetch
                # per image and layer that their references do not use.
                if baseline in ('none', 'anoco'):
                    neigh = bank_cache.get(li, rows)                      # [Kr,L,C]
                else:
                    neigh = None
                if baseline != 'none':
                    # ---- matched-feature baselines --------------------------
                    # Same descriptors, same bank, same pool, same map recipe
                    # and evaluator; only the reference/readout differs.
                    if baseline == 'subspacead':
                        mu_b, V_b = baseline_ref[li]
                        sc = bsl.subspacead_score(q_feat, mu_b, V_b,
                                                  power=baseline_power)
                    elif baseline == 'padim':
                        sc = bsl.padim_score(q_feat, baseline_ref[li],
                                             sqrt=bool(baseline_padim_sqrt))
                    elif baseline == 'patchcore':
                        sc = bsl.patchcore_score(q_feat, baseline_ref[li],
                                                 k=baseline_pc_k,
                                                 rw=baseline_pc_rw,
                                                 chunk=knn_chunk_imgs)
                    elif baseline == 'anoco':
                        sc = bsl.anoco_score(q_feat, neigh, cand_ctx,
                                             lambda_rel=baseline_anoco_lambda,
                                             chunk=knn_chunk_imgs,
                                             anchor=bool(baseline_anoco_anchor))
                    else:
                        raise SystemExit(f'[baseline] {baseline!r}')
                elif use_lonm:
                    dmax = lonm_dmax if lonm_quadratic else max(lonm_dmax, 1)
                    sc = cu.lonm_score(
                        q_feat, neigh, cand_ctx, M=lonm_M, d_max=dmax,
                        energy=lonm_energy, ridge=lonm_ridge, trust_q=lonm_trust,
                        quadratic=lonm_quadratic, chunk=knn_chunk_imgs,
                        ridge_rel=lonm_ridge_rel, res_rank=lonm_res_rank,
                        curv_gain=lonm_curv_gain, radial=lonm_radial,
                        norm_scale=lonm_norm, scale_q=lonm_scale_q,
                        norm_mode=lonm_norm_mode, scale_beta=lonm_scale_beta,
                        maha=bool(lonm_maha), maha_ridge=lonm_maha_ridge,
                        maha_rank=lonm_maha_rank, anc_beta=anc_beta,
                        fast_window=lonm_fast,
                        unbounded=lonm_unbounded, clip_stat=clip_stat)
                else:
                    if use_centroid:
                        sc = cu.centroid_score(
                            q_feat, neigh, cand_ctx, M=lonm_M,
                            mode=('cos' if use_centroid == 1 else 'raw'),
                            chunk=knn_chunk_imgs)
                    elif agg == 'ia':
                        sc = cu.window_ia_score(q_feat, neigh, cand_ctx,
                                                chunk=knn_chunk_imgs,
                                                K_ia=ia_K)
                    elif agg == 'dad':
                        sc = cu.window_dad_score(
                            q_feat, neigh, rows, cand_ctx,
                            dad_tabs[li]['rk'], dad_tabs[li]['lid'],
                            k=dad_k, lid_mode=dad_lid, comb=dad_comb,
                            metric=dad_metric, chunk=knn_chunk_imgs,
                            rk_fallback=dad_tabs[li]['rk_med'],
                            lid_fallback=dad_tabs[li]['lid_mean'],
                            stats=dad_stat, row_base=dad_base,
                            rk_med=dad_tabs[li]['rk_med'], dad_floor=dad_floor)
                    elif agg == 'cfar':
                        # position-wise empirical p-value against the class's
                        # OWN normal kdist distribution at the same position
                        # (CFAR / distribution-free rank calibration).  Needs no
                        # retrieval identity at all, so unlike --agg dad it is
                        # immune to cross-class retrieval rows.
                        sc = cu.window_cfar_score(
                            q_feat, neigh, cand_ctx, dad_tabs[li]['rk'],
                            k=dad_k, comb=cfar_comb, chunk=knn_chunk_imgs,
                            stats=dad_stat)
                    elif use_transport_window:
                        sc = cu.window_1nn(q_feat, neigh, cand_ctx,
                                           chunk=knn_chunk_imgs)
                    else:
                        # NB: take the RAW table list from the cache rather than
                        # `cand_ctx`.  --dad_post / --cfar_post force `cand_ctx`
                        # to the idx/valid layout so the post factor can be
                        # applied on top of agg='min', but the RAD path itself
                        # still expects its own tables -- passing `cand_ctx`
                        # here silently mixes the two layouts.
                        sc = patch_scores_positional(
                            q_feat, neigh,
                            pos_ctx_cache[(h, w, pos_radius)],
                            chunk_imgs=knn_chunk_imgs)
                scores_per_layer.append(sc)

            if cfar_post != 0.0:
                # ① + ② : keep the layer's own non-conformity (the reference
                # point contribution) and multiply in the position-wise
                # calibrated factor.  Unlike --agg cfar, which REPLACES the
                # score, this preserves whatever absolute structure the
                # configured path already produced.
                for li in range(num_layers):
                    q_feat = F.normalize(patch_list[li][b], dim=-1)
                    neigh = bank_cache.get(li, rows)
                    _f = cu.window_cfar_score(
                        q_feat, neigh, cand_ctx, dad_tabs[li]['rk'],
                        k=dad_k, comb='logp', chunk=knn_chunk_imgs,
                        stats=dad_stat)
                    scores_per_layer[li] = scores_per_layer[li] * (
                        _f.clamp_min(1e-6) ** float(cfar_post))

            if dad_post != 0.0:
                # ① and ② sit on different axes -- ① moves the reference point
                # (`k_NN` -> the query's projection onto its local tangent
                # chart), ② changes the statistic that the reference point is
                # fed into (`kdist(q)` -> the dimensionality-aware ratio to the
                # bank patches' OWN kdist).  `--agg dad` replaces RAD's score
                # outright and therefore cannot be stacked on ①; `--dad_post`
                # can, because it multiplies whatever score the configured path
                # produced by DAO**w.  A pure factor is the right composition
                # here: an image that is anomalous everywhere still has a large
                # DAO, because the denominator is always a NORMAL bank patch.
                for li in range(num_layers):
                    q_feat = F.normalize(patch_list[li][b], dim=-1)
                    neigh = bank_cache.get(li, rows)
                    scores_per_layer[li] = scores_per_layer[li] * (
                        cu.window_dad_score(
                            q_feat, neigh, rows, cand_ctx,
                            dad_tabs[li]['rk'], dad_tabs[li]['lid'],
                            k=dad_k, lid_mode=dad_lid, comb=dad_comb,
                            metric=dad_metric, chunk=knn_chunk_imgs,
                            rk_fallback=dad_tabs[li]['rk_med'],
                            lid_fallback=dad_tabs[li]['lid_mean'],
                            stats=dad_stat, row_base=dad_base,
                            rk_med=dad_tabs[li]['rk_med'],
                            dad_floor=dad_floor) ** float(dad_post))

            if defer_class_fuse:
                # Geo/z-fusion need a statistic taken over the CLASS (see the
                # block after the loop), so the per-layer maps are buffered and
                # the fusion is resolved once the whole class has been seen.
                cls_cal['lay'].append(
                    torch.stack(scores_per_layer, dim=0).detach().cpu())
                cls_cal['u'].append(None if u_map is None
                                    else u_map.detach().cpu())
                continue

            if layer_fuse == 'min':
                fused = torch.stack(scores_per_layer, dim=0).min(dim=0).values
            elif layer_fuse == 'geo':
                lg = torch.stack([torch.log(s.clamp_min(1e-6))
                                  for s in scores_per_layer], dim=0)      # [P, L]
                # NOTE: must not be called ``w`` -- that name holds the patch-grid
                # width in the enclosing scope and shadowing it breaks the
                # candidate-context cache on the next image of the batch.
                lw = torch.as_tensor(layer_weights, device=lg.device,
                                     dtype=lg.dtype).view(-1, 1)
                fused = (lg * lw).sum(dim=0).exp()
            elif layer_fuse in ('zfuse', 'rankfuse'):
                # SCALE-INVARIANT fusion.
                #
                # A weighted geometric mean is NOT scale invariant: log(s) is
                # shifted by log of the layer's typical magnitude, and that shift
                # is a different number on MVTec and on VisA (different textures,
                # different normal-patch variance), so the SAME formula wins on
                # one dataset and loses on the other.  A method whose sign flips
                # with the dataset cannot be reported as a method.
                #
                # Both variants below put every layer on a common scale using the
                # image's own scores only, which removes exactly the free
                # parameter the geometric mean was implicitly fitting:
                #   zfuse    : per-layer standardisation  (s - mu_l) / sd_l
                #   rankfuse : per-layer rank in [0, 1]   (no magnitude at all)
                # They differ in what is discarded -- zfuse keeps magnitude,
                # rankfuse keeps only order -- and the paired run on MVTec and
                # VisA decides which one, if either, is the dataset-independent
                # replacement for `geo`.
                norm_layers = []
                for sc in scores_per_layer:
                    if layer_fuse == 'zfuse':
                        mu = sc.mean(dim=-1, keepdim=True)
                        sd = sc.std(dim=-1, keepdim=True, unbiased=False)
                        norm_layers.append((sc - mu) / sd.clamp_min(1e-6))
                    else:
                        r = sc.argsort(dim=-1).argsort(dim=-1).to(sc.dtype)
                        r = r / max(1, sc.shape[-1] - 1)
                        norm_layers.append(r)
                fused = torch.zeros_like(norm_layers[0])
                lwt = torch.as_tensor(layer_weights, device=fused.device,
                                      dtype=fused.dtype)
                for li in range(num_layers):
                    fused = fused + lwt[li] * norm_layers[li]
            else:
                fused = torch.zeros_like(scores_per_layer[0])
                for li in range(num_layers):
                    fused = fused + layer_weights[li] * scores_per_layer[li]
            # NOTE: `use_unmatched` must gate this explicitly.  u_map is also
            # materialised whenever img_cal == 'matched' (the plan-anchored
            # calibration needs it), so testing only `u_map is not None` silently
            # switched the unmatched-mass term ON at the default weight for every
            # `matched` row.  That confounded two different mechanisms inside one
            # ablation row, and it is why `--img_cal matched --img_cal_alpha 0.0`
            # used to score above plain no-calibration instead of matching it.
            if use_unmatched and u_map is not None and unmatched_w:
                fused = fused + unmatched_w * u_map.to(fused.dtype)
            patch_scores_batch.append(fused)
            u_list.append(u_map)

        if defer_class_fuse:
            # Nothing is fused yet.  Record what the class-end pass needs and
            # move on -- gt/label/ghgw are buffered in the SAME lists the
            # deferred-calibration path uses, so `_emit` is called once, at the
            # very end, from a single place.
            cls_cal['gt'].append(gt.detach().cpu())
            cls_cal['label'].append(label.detach().cpu())
            cls_cal['ghgw'].append((h, w))
            n_seen += B
            if limit_imgs and n_seen >= limit_imgs:
                break
            continue

        patch_scores_batch = torch.stack(patch_scores_batch, dim=0)        # [B,L]
        # ---- per-image null calibration (before the class-wide pooled AP) ----
        # `img_cal_ref` / `img_cal_alpha` make the plan-anchored offset robust.
        #
        # Subtracting a per-image offset o_i cannot change the ordering INSIDE an
        # image, so it can only ever fix (or break) the *between-image* scale
        # mismatch of the pooled ranking.  Empirically the two sides are very
        # unequal: on MVTec-15 the correction is worth +2.9% on grid and +1.5% on
        # leather but costs -45.7% on metal_nut, -33.7% on transistor and -29.1%
        # on pill, because there o_i itself carries the defect signal (a deformed
        # nut raises the whole image's normal level, so removing o_i also removes
        # the evidence).  A *uniform* shift of all o_i is a no-op for the pooled
        # ranking, so only the part of o_i that EXCEEDS the class-typical level
        # can possibly help:
        #     o_i <- max(0, o_i - median_j(o_j))          (img_cal_ref = median)
        #     o_i <- max(0, o_i - Q_q(o))                 (img_cal_ref = q)
        # which is self-gating: when every image sits on a common normal level the
        # correction collapses to ~0 and the class is returned untouched, instead
        # of being damaged.  `img_cal_alpha` additionally scales the offset so a
        # partial correction can be screened.  Defaults (none / 1.0) reproduce the
        # plain `matched` behaviour bit-for-bit.
        if img_cal == 'matched_pit':
            # null-CDF transform: needs no reference level at all (it is a rank
            # transform), so it never defers and never consults img_cal_ref.
            pit_rows = [cu.matched_pit(patch_scores_batch[b:b + 1], u_list[b],
                                       img_cal_q)
                        for b in range(patch_scores_batch.shape[0])]
            patch_scores_batch = torch.cat(pit_rows, dim=0)
        elif img_cal in ('matched', 'matched_z', 'matched_s', 'matched_sg',
                         'matched_sgo'):
            # The three `matched_s*` modes are the SCALE family.  They exist
            # because the diagnosis and the original fix disagreed: every
            # measurement in this project says the loss is a *scale* mismatch
            # (VisA pcb2/pcb3 have pixel-AUROC 0.99 with pixel-AP 0.26), yet the
            # fix that was implemented subtracted a per-image *level*.  A level
            # carries defect evidence on MVTec metal_nut/transistor, which is why
            # that form is negative there.
            #   matched_s    : s / tau_i                       (width only)
            #   matched_sg   : s / max(tau_i, median_j tau_j)  (gated width)
            #   matched_sgo  : (s - gated offset) / gated width (both, gated)
            # The `max(tau_i, median_j tau_j)` gate is the scale analogue of the
            # offset self-gate: it removes excess width but can never divide by a
            # below-typical width, so a clean class is never AMPLIFIED -- which is
            # the failure mode a pure 1/tau would introduce.
            subtract_off = img_cal in ('matched', 'matched_z', 'matched_sgo')
            gate_scale = img_cal in ('matched_sg', 'matched_sgo')
            offs_t, scs_t, _, _ = _calib_params_batch(patch_scores_batch, u_list)
            if defer_class_cal:
                # resolve the reference once for the whole class, after the loop
                cls_cal['fused'].append(patch_scores_batch.detach().cpu())
                cls_cal['offs'].append(offs_t.detach().cpu())
                cls_cal['scs'].append(scs_t.detach().cpu())
                cls_cal['gate_scale'] = bool(cls_cal.get('gate_scale', False)
                                             or gate_scale)
                cls_cal['subtract_off'] = bool(
                    cls_cal.get('subtract_off', False) or subtract_off)
                cls_cal['gt'].append(gt.detach().cpu())
                cls_cal['label'].append(label.detach().cpu())
                cls_cal['ghgw'].append((h, w))
            else:
                patch_scores_batch = _apply_calib(patch_scores_batch, offs_t,
                                                  scs_t, subtract_off,
                                                  gate_scale)
            if img_cal_stats:
                cal_stat_buf.append(offs_t.detach().cpu().flatten())
        elif img_cal != 'none':
            patch_scores_batch = cu.image_level_calibrate(
                patch_scores_batch, img_cal, img_cal_q)

        if not defer_class_cal:
            _emit(patch_scores_batch, gt, label, h, w)

        n_seen += B
        if verbose and log_every and (n_seen // max(B, 1)) % log_every == 0:
            print(f'    [{n_seen:5d} imgs | {time.time() - t0:6.1f}s] '
                  f'{bank_cache.report()}', flush=True)
        if limit_imgs and n_seen >= limit_imgs:
            break

    # ---- deferred CLASS-level fusion (geo_cls / zfuse_cls / gls) ------------
    # Why the layer statistic has to come from the CLASS, and why the obvious
    # repairs are wrong:
    #
    #   `geo` is a weighted geometric mean of the four layer score maps.  It is
    #   a good fusion on MVTec (+0.0055) and a bad one on VisA (-0.0103), and the
    #   tempting explanation -- per-layer units -- is WRONG: rescaling a layer by
    #   a constant a_l multiplies the geometric mean by the constant
    #   prod a_l^w_l, which does not change any ranking.  So `geo` is already
    #   invariant to per-layer scale.
    #
    #   What actually differs is its CHARACTER: AM-GM makes the geometric mean a
    #   soft minimum across layers, so where the layers disagree -- one says
    #   normal, another says anomalous -- `geo` pulls the fused score DOWN.  That
    #   helps when the layer disagreement is noise (MVTec textures) and hurts
    #   when it is signal (VisA, where the useful layer varies by class).  That
    #   is a property of the operator, not of the units, so it cannot be repaired
    #   by normalising anything -- which is exactly what `geo_cls` demonstrates:
    #   it is algebraically `geo / const`, so its printed metrics must come out
    #   IDENTICAL to `geo`'s.  It is kept as a self-verification probe.
    #
    #   The other tempting repair, standardising each layer PER IMAGE (`zfuse`),
    #   is worse: it deletes the between-image scale, and that scale is the
    #   signal.  Measured: u_c1zf_m 0.5352 against u_c1_m 0.7722, image-level
    #   AUROC 0.9951 -> 0.9104.
    #
    #   So what is left is to stop guessing the combination and ESTIMATE it, from
    #   the class's own unlabelled scores, on the correct population:
    #     gls        w ~ Sigma^-1 1 over the class  (Gauss-Markov / BLUE)
    #     zfuse_cls  the diagonal-only special case  w_l ~ 1 / sd_l^class
    #   Both need no labels and have no free hyper-parameter to fit per dataset,
    #   which is the whole point: the layer weighting is measured, not chosen.
    # -------------------------------------------------------------------------
    if defer_class_fuse and cls_cal['lay']:
        # stack, NOT cat: each buffered entry is [num_layers, L] and the class
        # axis has to stay separable from the layer axis -- cat would fuse them
        # into [N*num_layers, L] and every per-layer statistic below would then
        # be taken over the wrong axis.
        S = torch.stack(cls_cal['lay'], dim=0).to(device)        # [N, num_layers, L]
        lwt = torch.as_tensor(layer_weights, device=device, dtype=S.dtype)
        lwv = lwt.view(1, -1, 1)
        if layer_fuse == 'geo_cls':
            lev = S.mean(dim=(0, 2)).clamp_min(1e-6)
            fused_all = ((S / lev.view(1, -1, 1)).clamp_min(1e-6).log() * lwv
                         ).sum(dim=1).exp()
        elif layer_fuse == 'gls':
            # GAUSS-MARKOV (BLUE) COMBINATION OF THE LAYER SCORES.
            #
            # Each layer produces an estimate of the same latent quantity -- how
            # anomalous this patch is -- with a different noise level.  Combining
            # independent unbiased estimators with inverse-variance weights is the
            # minimum-variance unbiased combination, and with correlated
            # estimators the generalised version weights by the inverse
            # COVARIANCE.  With only `num_layers` (4) estimators the covariance is
            # a 4x4 matrix, and it is estimable from the class's own UNLABELLED
            # patch scores, so this costs nothing, needs no labels, and has NO
            # free hyper-parameter to tune per dataset:
            #
            #     w  ∝  Sigma^{-1} 1 ,      fused = w^T s
            #
            # This replaces a fixed uniform weighting (the baseline) and the
            # geometric mean, whose "soft min" character is what makes its sign
            # depend on the dataset.  Negative weights are allowed and expected:
            # they are how the BLUE removes what two layers merely agree on.
            nl = S.shape[1]
            X = S.permute(0, 2, 1).reshape(-1, nl).double()      # [N*L, num_layers]
            Xc = X - X.mean(dim=0, keepdim=True)
            Sig = (Xc.t() @ Xc) / max(1, Xc.shape[0] - 1)
            ridge = 1e-6 * Sig.diagonal().mean()
            Sig = Sig + ridge * torch.eye(nl, device=Sig.device, dtype=Sig.dtype)
            w = torch.linalg.solve(Sig, torch.ones(nl, 1, device=Sig.device,
                                                   dtype=Sig.dtype)).flatten()
            w = w / w.abs().sum().clamp_min(1e-12)
            print(f'[gls-w] {[round(float(x), 4) for x in w]}', flush=True)
            del X, Xc, Sig
            fused_all = (S * w.view(1, -1, 1).to(S.dtype)).sum(dim=1)
        else:
            mu = S.mean(dim=(0, 2))
            sd = S.std(dim=(0, 2), unbiased=False).clamp_min(1e-6)
            fused_all = (((S - mu.view(1, -1, 1)) / sd.view(1, -1, 1))
                         * lwv).sum(dim=1)
        del S
        u_all = list(cls_cal['u'])
        if use_unmatched and unmatched_w:
            U = torch.cat([
                (u.to(device) if u is not None
                 else torch.zeros(fused_all.shape[1], device=device,
                                  dtype=fused_all.dtype)) for u in u_all], dim=0)
            fused_all = fused_all + unmatched_w * U
        if img_cal in ('matched', 'matched_z', 'matched_s', 'matched_sg',
                       'matched_sgo'):
            offs_t, scs_t, sub_off, gate_sc = _calib_params_batch(fused_all, u_all)
            fused_all = _apply_calib(fused_all, offs_t, scs_t, sub_off, gate_sc)
            if img_cal_stats:
                cal_stat_buf.append(offs_t.detach().cpu().flatten())
        ptr = 0
        for gtb, labb, (gh, gw) in zip(cls_cal['gt'], cls_cal['label'],
                                       cls_cal['ghgw']):
            nb = gtb.shape[0]
            _emit(fused_all[ptr:ptr + nb], gtb.to(device), labb.to(device),
                  gh, gw)
            ptr += nb

    # ---- deferred CLASS-level resolution of the calibration reference -------
    if defer_class_cal and cls_cal['fused']:
        fused_all = torch.cat(cls_cal['fused'], dim=0).to(device)
        offs_all = torch.cat(cls_cal['offs'], dim=0).to(device)
        scs_all = torch.cat(cls_cal['scs'], dim=0).to(device)
        if img_cal_ref == 'median':
            ref = offs_all.median()
        elif img_cal_ref == 'q':
            ref = torch.quantile(offs_all, float(img_cal_q))
        else:
            ref = offs_all.new_zeros(())
        if img_cal_stats:
            cal_stat_buf.append(offs_all.detach().cpu().flatten())
        if cls_cal.get('gate_scale', False):
            scs_all = torch.maximum(scs_all, scs_all.median())
        if cls_cal.get('subtract_off', True):
            offs_all = (offs_all - ref).clamp_min(0.0)
            if float(img_cal_alpha) != 1.0:
                offs_all = float(img_cal_alpha) * offs_all
        else:
            offs_all = torch.zeros_like(offs_all)
        fused_all = (fused_all - offs_all) / scs_all
        ptr = 0
        for gtb, labb, (gh, gw) in zip(cls_cal['gt'], cls_cal['label'],
                                       cls_cal['ghgw']):
            nb = gtb.shape[0]
            _emit(fused_all[ptr:ptr + nb], gtb.to(device), labb.to(device),
                  gh, gw)
            ptr += nb

    if img_cal_stats and cal_digest:
        d_sum = sum(cal_digest[0::2])
        d_sq = sum(cal_digest[1::2])
        print(f'[cal-digest] batches={len(cal_digest) // 2} '
              f'n={sum(cal_digest_n)} sum={d_sum:.10f} sq={d_sq:.10f}',
              flush=True)

    if img_cal_stats and cal_stat_buf:
        # per-class distribution of the offsets the calibration removes.  This is
        # the statistic a label-free gate has to be built on, so it is dumped
        # rather than guessed.
        o = torch.cat(cal_stat_buf).float()
        med = o.median()
        print(f'[cal-stats] n={o.numel()} '
              f'median={float(med):.6f} '
              f'mad={float((o - med).abs().median()):.6f} '
              f'std={float(o.std(unbiased=False)):.6f} '
              f'q10={float(torch.quantile(o, 0.10)):.6f} '
              f'q90={float(torch.quantile(o, 0.90)):.6f} '
              f'mean={float(o.mean()):.6f} '
              f'frac_gt_med={float((o > med).float().mean()):.4f} '
              f'frac_zero={float((o <= 1e-6).float().mean()):.4f}',
              flush=True)

    gt_px = torch.cat(gt_list_px, dim=0)[:, 0].cpu().numpy()
    pr_px = torch.cat(pr_list_px, dim=0)[:, 0].cpu().numpy()
    gt_sp = torch.cat(gt_list_sp).flatten().cpu().numpy()
    pr_sp = torch.cat(pr_list_sp).flatten().cpu().numpy()

    if (agg in ('dad', 'cfar') or dad_post != 0.0 or cfar_post != 0.0) and verbose:
        _nm = int(dad_stat.get('miss', 0))
        _nn = max(int(dad_stat.get('n', 0)), 1)
        _rate = _nm / _nn
        print(f'[dad-miss] {item_name}: {_nm}/{_nn} = {_rate:.6f} neighbour '
              f'lookups landed outside the class block (falling back to the '
              f'class median).  The blocks are certified by CLS purity, so a '
              f'small rate just measures how often CLS retrieval crosses '
              f'classes; a rate near 1 means the blocks are wrong.', flush=True)
        if _rate > 0.02:
            print(f'[dad-miss] WARNING: {_rate:.2%} of retrievals left the class '
                  f'block of {item_name!r}; the DAO denominator of those '
                  f'positions is a fallback, not a real statistic.', flush=True)
        if _rate > 0.5:
            raise SystemExit(
                f'[dad-blocks] {_rate:.2%} of the bank-side lookups fell outside '
                f'the class block of {item_name!r} -- this is the wrong-block '
                f'signature, not a retrieval quirk.  The block layout is '
                f'recovered by _bank_class_order and certified by CLS purity; '
                f'check the [bank-order] lines above and pass an explicit '
                f'--bank_item_order if the bank was built in another order.')

    aupro_px = compute_pro(gt_px, pr_px)
    g, p = gt_px.ravel(), pr_px.ravel()
    res = dict(
        auroc_sp=float(roc_auc_score(gt_sp, pr_sp)),
        ap_sp=float(average_precision_score(gt_sp, pr_sp)),
        f1_sp=float(f1_score_max(gt_sp, pr_sp)),
        auroc_px=float(roc_auc_score(g, p)),
        ap_px=float(average_precision_score(g, p)),
        f1_px=float(f1_score_max(g, p)),
        aupro_px=float(aupro_px),
        mean_normal_score=float(np.mean(norm_scores)) if norm_scores else float('nan'),
        n_images=int(len(gt_sp)),
    )
    if clip_stat is not None and clip_stat.get('n'):
        _n = clip_stat['n']
        res['clip_n'] = int(_n)
        res['clip_frac'] = float(clip_stat.get('clip', 0)) / _n
        res['clip_ratio_mean'] = float(clip_stat.get('ratio_sum', 0.0)) / _n
        res['clip_ratio_max'] = float(clip_stat.get('ratio_max', 0.0))
        print(f'[radius-clip] {item_name}: {res["clip_frac"]:.4%} of the '
              f'{_n} (patch,layer) coordinates exceeded R; mean ||a||/R = '
              f'{res["clip_ratio_mean"]:.4f}, max = {res["clip_ratio_max"]:.3f}',
              flush=True)
    return res


# ============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_path', type=str, required=True)
    ap.add_argument('--bank_path', type=str, required=True)
    ap.add_argument('--encoder_name', type=str, default='dinov3_vitb16')
    ap.add_argument('--encoder_weight', type=str,
                    default='/home/amax/project/foundad/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth')
    ap.add_argument('--save_dir', type=str, default='./saved_curvot')
    ap.add_argument('--save_name', type=str, default='debug')
    ap.add_argument('--item_list', nargs='+', default=['auto'])
    ap.add_argument('--layout', type=str, default='folder',
                    choices=['folder', 'realiad'],
                    help="folder = <data_path>/<item>/{train,test} (MVTec / VisA "
                         "1cls / 3D-ADAM); realiad = Real-IAD (realiad_1024 + "
                         "realiad_jsons).  The Real-IAD test dataset returns the "
                         "same (img, gt, label, path) tuple, so every mechanism "
                         "below applies unchanged.")
    ap.add_argument('--k_image', type=int, default=150)
    ap.add_argument('--resize_mask', type=int, default=256)
    ap.add_argument('--max_ratio', type=float, default=0.01)
    ap.add_argument('--use_positional_bank', type=int, default=1)
    ap.add_argument('--pos_radius', type=int, default=1)
    ap.add_argument('--layer_weights', nargs='+', type=float, default=None)
    ap.add_argument('--image_size', type=int, default=512)
    ap.add_argument('--crop_size', type=int, default=448)
    ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--num_workers', type=int, default=4)
    ap.add_argument('--gpu_bank_cache_mb', type=int, default=0)
    ap.add_argument('--normalize_bank', type=int, default=1)
    ap.add_argument('--load_mmap', type=int, default=1)
    ap.add_argument('--knn_chunk_imgs', type=int, default=32)
    ap.add_argument('--global_chunk_imgs', type=int, default=64)

    # ---- method switches ----
    ap.add_argument('--use_aptr', type=int, default=0)
    ap.add_argument('--ot_k0', type=int, default=128)
    ap.add_argument('--ot_kr', type=int, default=24)
    ap.add_argument('--ot_mass', type=float, default=0.8)
    ap.add_argument('--ot_reg', type=float, default=0.05)
    ap.add_argument('--ot_lam_pos', type=float, default=0.05)
    ap.add_argument('--ot_iter', type=int, default=120)
    ap.add_argument('--coarse_grid', type=int, default=7)

    ap.add_argument('--use_lonm', type=int, default=0)
    ap.add_argument('--lonm_M', type=int, default=24)
    ap.add_argument('--lonm_dmax', type=int, default=4)
    ap.add_argument('--lonm_energy', type=float, default=0.9)
    ap.add_argument('--lonm_ridge', type=float, default=1e-4)
    ap.add_argument('--lonm_trust', type=float, default=0.9)
    ap.add_argument('--lonm_quadratic', type=int, default=1)
    ap.add_argument('--lonm_ridge_rel', type=float, default=0.0)
    ap.add_argument('--lonm_res_rank', type=int, default=0)
    ap.add_argument('--lonm_curv_gain', type=float, default=1.0)
    ap.add_argument('--lonm_norm', type=int, default=0,
                    help='1 = divide the manifold distance by the leave-one-out local '
                         'scale of the normal candidates (local non-conformity)')
    ap.add_argument('--lonm_scale_q', type=float, default=0.9)
    ap.add_argument('--lonm_radial', type=int, default=0,
                    help='1 = isotropic ||y||^2 curvature basis instead of the full '
                         'quadratic monomials (needs far fewer local samples)')
    ap.add_argument('--lonm_scale_beta', type=float, default=0.0,
                    help='shrink the local scale towards the image median: '
                         'scale_eff = scale + beta * median(scale).  Pure '
                         'pointwise division is scale-invariant and amplifies '
                         'noise in near-flat regions (P-AP collapse); beta>0 '
                         'adds the missing absolute reference.')
    ap.add_argument('--lonm_norm_mode', type=str, default='ratio',
                    choices=['ratio', 'sat', 'raw', 'sub', 'maha'],
                    help='ratio: num/scale ; sat: num/(num+scale) in [0,1) ; '
                         'raw: num itself (pure distance to the local normal '
                         'subspace, no calibration) ; sub: max(num-scale, 0), i.e. '
                         'subtract the local noise floor instead of dividing by it ; '
                         'maha: per-DIRECTION normalisation of the normal-space '
                         'residual by the local residual spectrum (needs --lonm_maha 1)')
    ap.add_argument('--lonm_maha', type=int, default=0,
                    help='1 = Mahalanobis-style directional normalisation: each '
                         'principal direction of the local normal residual bundle is '
                         'divided by its OWN standard deviation, instead of dividing '
                         'everything by one scalar (which collapsed in 13 configs)')
    ap.add_argument('--lonm_maha_ridge', type=float, default=0.1,
                    help='relative ridge on the per-direction variances')
    ap.add_argument('--lonm_maha_rank', type=int, default=0,
                    help='number of normal-residual directions to use (0 = M-1-dmax)')
    ap.add_argument('--anc_beta', type=float, default=0.0,
                    help='contribution (2) candidate: neighbourhood COHERENCE '
                         'calibration of the ① local chart.  The M nearest bank '
                         'patches are averaged with equal weight, so the chart '
                         'anchor z0 is a plain centroid; ANoCo (CVPR 2026) builds '
                         'its normal manifold only from references compatible with '
                         'the query\'s single best match, because a raw top-K set '
                         'can be locally plausible yet mutually incoherent once one '
                         'reference image is itself defective (this project\'s '
                         'diagnostic B, and the measured cause of the upper-tail '
                         'false positives that cap pixel-AP).  Here each neighbour '
                         'is weighted by softmax(anc_beta * cos(anchor, f_j)) and '
                         'the centroid / covariance become the WEIGHTED ones.  '
                         '0.0 = bit-identical to uniform.  The sign is the open '
                         'question: beta > 0 sharpens the chart onto near-identical '
                         'patches, beta < 0 broadens it onto the full spread of '
                         'normal appearance in the window.  Requires --use_lonm 1.')
    ap.add_argument('--lonm_fast', type=int, default=0,
                    help='1 = form the LONM window similarities by shifting the '
                         'QUERY and re-aligning the offsets (the bank is read '
                         'once per layer/image) instead of gathering the '
                         '[Kr*Kp, C] candidate block once per offset.  The '
                         'arithmetic is the same products in a different order: '
                         'measured 4e-10 relative, printed metrics identical '
                         '(test_agg.py / test_window_fast.py).  Real-IAD '
                         'rho=4/K=900 is ~875 GB of traffic per layer/image on '
                         'the gather path vs ~7 GB here.  Requires --use_lonm 1.')
    ap.add_argument('--lonm_unbounded', type=int, default=0,
                    help='1 = remove the finite coordinate radius, i.e. the '
                         'R = inf limit of the reference set: alpha == 1 for '
                         'every query and the score is the orthogonal residual '
                         'to the local affine span.  This is the radius control '
                         'of Section IV-A.  Default 0 leaves every stored result '
                         'bit-identical.  Requires --use_lonm 1.')
    ap.add_argument('--lonm_clipstat', type=int, default=0,
                    help='1 = report, per class, the fraction of (patch,layer) '
                         'query coordinates that exceeded the fitted radius R '
                         '(and the mean/max of ||a||/R).  Diagnostic only; has '
                         'no effect on any score.  Requires --use_lonm 1.')
    ap.add_argument('--baseline', type=str, default='none',
                    choices=['none', 'subspacead', 'padim', 'patchcore', 'anoco'],
                    help='matched-feature baseline to score with instead of '
                         'RAD/LONM (see baseline_scores.py).  Same frozen '
                         'features, same normal bank, same output processing '
                         'and evaluator as the paired reruns; only the '
                         'reference construction and readout change.  '
                         'none = the configured RAD/LONM path, unchanged.')
    ap.add_argument('--baseline_d', type=int, default=64,
                    help='subspace dimension for --baseline subspacead '
                         '(retained principal components) and padim '
                         '(retained feature dimensions per position).')
    ap.add_argument('--baseline_power', type=float, default=2.0,
                    help='subspacead residual exponent: 2 = the squared '
                         'orthogonal projection residual of the published '
                         'method, 1 = the unsquared form used by the '
                         'LFR-vs-subspace limit in the paper.')
    ap.add_argument('--baseline_padim_shrink', type=str, default='published',
                    choices=['published', 'lw', 'ridge'],
                    help='padim covariance regularisation.  published = an '
                         'absolute ridge, defaulting to the paper\'s own '
                         '0.01*I; lw = Ledoit-Wolf shrinkage towards mu*I '
                         '(scale free, no free parameter); ridge = an explicit '
                         'absolute + relative ridge.  The three regimes score '
                         'very differently here, so all three are run and '
                         'reported rather than one being chosen.')
    ap.add_argument('--baseline_padim_ridge_rel', type=float, default=0.0,
                    help='padim ridge as a fraction of the mean covariance '
                         'diagonal (used by published / ridge).')
    ap.add_argument('--baseline_padim_ridge_abs', type=float, default=0.01,
                    help='padim absolute ridge (used by published / ridge).  '
                         'The default is the value in the published code.')
    ap.add_argument('--baseline_padim_sqrt', type=int, default=1,
                    help='1 = the published readout, the square root of the '
                         'Mahalanobis quadratic form; 0 = the squared form.')
    ap.add_argument('--baseline_pc_ratio', type=float, default=0.01,
                    help='patchcore coreset ratio of the category bank.')
    ap.add_argument('--baseline_pc_k', type=int, default=9,
                    help='patchcore neighbourhood size b for the reweighting.')
    ap.add_argument('--baseline_pc_rw', type=float, default=1.0,
                    help='patchcore reweighting constant w.')
    ap.add_argument('--baseline_pc_cap', type=int, default=2000,
                    help='absolute cap on the patchcore coreset size per '
                         'category and layer (0 = no cap).  Greedy k-center is '
                         'O(m*n); without a cap a VisA category needs a ~5.6k '
                         'point coreset and about 20 h per category.')
    ap.add_argument('--baseline_pc_proj', type=int, default=64,
                    help='patchcore coreset random-projection dimension (0 = '
                         'select in the full 768-d feature space).')
    ap.add_argument('--baseline_pc_subcap', type=int, default=120000,
                    help='uniform cap on the patchcore coreset selection '
                         'candidates (0 = use every patch).')
    ap.add_argument('--baseline_anoco_lambda', type=float, default=1.0,
                    help='ANoCo stability weight, relative to the bipartite '
                         'degree (lam_i = lambda * D_ii).  The published note '
                         'does not state a value, so it is fixed a priori '
                         'rather than tuned on the test set.')
    ap.add_argument('--baseline_anoco_anchor', type=int, default=1,
                    help='1 = use ANoCo\'s anchor-consistency rule when '
                         'selecting the reference neighbours, 0 = the full '
                         'candidate pool (their "Bipartite" ablation row).')


    ap.add_argument('--use_transport_window', type=int, default=0)
    ap.add_argument('--max_shift', type=int, default=3)
    ap.add_argument('--use_unmatched', type=int, default=0,
                    help='1 = fuse the per-query-cell unmatched partial-OT mass '
                         '(a direct anomaly cue, only available when mass < 1)')
    ap.add_argument('--unmatched_w', type=float, default=0.05)
    ap.add_argument('--use_centroid', type=int, default=0,
                    help="0-th order geometry control. 1 = score the local "
                         "centroid of the M nearest normal candidates with "
                         "1 - cos (RAD-style angular statistic), 2 = the same "
                         "centroid with ||q - z0||. The pair differs only in the "
                         "score form, which is what separates 'the absolute "
                         "distance itself helps' from 'a reconstruction is not a "
                         "unit vector so the angular form throws away the radial "
                         "part'.")
    ap.add_argument('--img_cal', type=str, default='none',
                    choices=['none', 'sub_med', 'sub_q', 'znorm',
                             'matched', 'matched_z', 'matched_pit',
                             'matched_s', 'matched_sg', 'matched_sgo'],
                    help="per-image null calibration applied to the [B,L] patch "
                         "score block BEFORE the class-wide pooled pixel AP. The "
                         "RAD protocol pools every test image of a class into one "
                         "ranking, so an inter-image offset decides whether one "
                         "image's normal tail outranks another image's defect. "
                         "'sub_med'/'sub_q' remove a per-image offset, 'matched' "
                         "takes the offset from the patches partial OT "
                         "confidently matched, 'matched_pit' maps every score to "
                         "the fraction of matched (normal) patches below it, "
                         "i.e. to that image's own null CDF -- also a per-image "
                         "monotone transform, but one that aligns the normal "
                         "tails as well as the centres (needs --use_aptr 1). "
                         "The SCALE family is the one that was missing: the "
                         "diagnosis says the mismatch is a scale mismatch, but "
                         "the original fix removed a level, and a level carries "
                         "defect evidence on MVTec metal_nut/transistor. "
                         "'matched_s' divides by the plan-vouched null width, "
                         "'matched_sg' divides by max(tau_i, median_j tau_j) so "
                         "a clean image is never amplified, and 'matched_sgo' "
                         "combines the gated width with the gated offset.")
    ap.add_argument('--img_cal_q', type=float, default=0.5,
                    help="sub_q: lowest fraction used as the normal floor; "
                         "matched: quantile of the unmatched-mass field below "
                         "which a patch counts as confidently matched; "
                         "img_cal_ref=q: the reference quantile of the "
                         "per-image offsets.")
    ap.add_argument('--img_cal_ref', type=str, default='none',
                    choices=['none', 'median', 'q'],
                    help="reference level of the per-image offsets. 'none' keeps "
                         "the plain per-image subtraction; 'median'/'q' remove "
                         "only the offset EXCEEDING that reference, i.e. "
                         "o <- max(0, o - ref). This is self-gating: classes "
                         "whose images already share a normal level are left "
                         "untouched instead of being damaged, which is what "
                         "plain 'matched' does to metal_nut/transistor/pill "
                         # argparse runs `help` through %-formatting, so every
                         # literal percent sign has to be doubled.  A single
                         # bare "%" here (as this string used to have) makes
                         # `--help` itself raise
                         #   ValueError: unsupported format character %r
                         # for the whole parser, i.e. no flag can be
                         # introspected until it is fixed.
                         "(-46%%/-34%%/-29%% on MVTec-15).")
    ap.add_argument('--img_cal_alpha', type=float, default=1.0,
                    help="scale the applied offset by this factor (1.0 = full "
                         "correction, 0.0 = off) to screen partial variants.")
    ap.add_argument('--img_cal_scope', type=str, default='batch',
                    choices=['batch', 'class'],
                    help="the population the img_cal_ref reference is taken "
                         "over.  'batch' (default, bit-exact with every result "
                         "produced so far) uses one batch of batch_size images, "
                         "which is an arbitrary subset of the class and injects "
                         "a between-batch shift into a per-CLASS pooled AP. "
                         "'class' buffers the fused maps and resolves the "
                         "reference once for the whole class, which is the scope "
                         "the metric actually uses -- only then is a uniform "
                         "shift a true no-op, i.e. only then does the mechanism "
                         "leave a scale-consistent class untouched as claimed.")
    ap.add_argument('--img_cal_red', type=str, default='median',
                    choices=['median', 'mean', 'q'],
                    help="which statistic of the plan-vouched matched patches is "
                         "used as the per-image normal level.  'median' (the "
                         "default) aligns the normal CENTRE; 'q' uses "
                         "--img_cal_red_q and aligns the normal TAIL, which is "
                         "the part that actually competes with another image's "
                         "defects in a pooled pixel-AP.")
    ap.add_argument('--img_cal_red_q', type=float, default=0.9,
                    help="quantile of the matched scores used when "
                         "--img_cal_red q.")
    ap.add_argument('--img_cal_scale_q', type=float, default=0.95,
                    help="high quantile defining the null WIDTH tau = (Q_q - "
                         "median) of the plan-vouched matched scores, used by "
                         "the matched_s* scale family.")
    ap.add_argument('--img_cal_stats', type=int, default=0,
                    help="1 = print the per-class distribution of the removed "
                         "offsets (median/mad/std/q10/q90) -- the statistic a "
                         "label-free gate would have to be built on.")
    ap.add_argument('--layer_fuse', type=str, default='mean',
                    choices=['mean', 'geo', 'min', 'zfuse', 'rankfuse',
                             'geo_cls', 'zfuse_cls', 'gls'],
                    help='how the 4 layer score maps are combined.  mean/geo/'
                         'min are the original forms.  zfuse (per-image '
                         'standardisation) and rankfuse (per-image rank) are '
                         'scale invariant but delete the between-image scale, '
                         'which IS the signal -- measured at -0.24 on MVTec-6. '
                         'geo_cls / zfuse_cls / gls are the class-scoped '
                         'replacements: the per-layer statistic is taken over '
                         'the CLASS, so the layer combination does not depend '
                         'on one dataset\'s score geometry.  gls is the '
                         'Gauss-Markov (BLUE) combination w ~ Sigma^-1 1 with '
                         'the 4x4 covariance estimated from the class\'s own '
                         'unlabelled patch scores: no labels, no free '
                         'hyper-parameter to tune per dataset.')
    ap.add_argument('--agg', type=str, default='min', choices=['min', 'ia', 'dad', 'cfar'],
                    help="how the reference bank is reduced for each patch. "
                         "'min' is RAD's hard max over (reference image x "
                         "window offset).  'ia' is the interval average of MuSc "
                         "(ICLR 2024): best match inside the window per "
                         "reference image, then the mean of the K best images, "
                         "which is far less sensitive to a single reference "
                         "image that happens to look like the defect.  "
                         "ia with --ia_K 1 is bit-exactly 'min'.")
    ap.add_argument('--cfar_comb', type=str, default='logp',
                    choices=['logp', 'neg'],
                    help="how --agg cfar turns the position-wise p-value into a "
                         "score: 'logp' = -log(p), 'neg' = 1-p.  Both are "
                         "monotone in p, so the ranking is identical -- 'neg' "
                         "only exists so mean_normal_score is interpretable.")
    ap.add_argument('--cfar_post', type=float, default=0.0,
                    help="stack the position-wise CFAR factor on top of the "
                         "configured score: s <- s * (-log p)**cfar_post.  0 "
                         "disables.  This is the ①+② combination.")
    ap.add_argument('--ia_K', type=int, default=0,
                    help="K for --agg ia.  0 = 10%% of the bank size "
                         "(MuSc's recommended regime).")

    # ---- band-pass probe (the frequency axis) -----------------------------
    ap.add_argument('--band_mode', type=str, default='off',
                    choices=['off', 'swt', 'hp'],
                    help="frequency-domain filtering of the patch-grid score "
                         "map.  'swt' = one-level UNDECIMATED Haar, four "
                         "subbands that each stay at h x w, recombined with "
                         "--band_w.  'hp' = X - beta*blur3(X) with --band_beta. "
                         "'off' (default) is bit-identical to not filtering.")
    ap.add_argument('--band_w', type=_parse_band_w, default=(1.0, 1.0, 1.0, 1.0),
                    help="swt subband weights a,h,v,d.  Because A+H+V+D == "
                         "2*x00:  (1,1,1,1) = identity;  (1,0,0,0) = 2x2 box "
                         "low-pass at the input's own level;  (0,1,1,1) = pure "
                         "high-pass (the map minus its 2x2 local mean);  "
                         "(t,1,1,1) interpolates identity (t=1) to pure "
                         "high-pass (t=0).")
    ap.add_argument('--band_beta', type=float, default=0.0,
                    help="hp mode: X - band_beta * blur3(X).  0.0 = identity.")

    # ---- DAD: dimensionality-aware deviation (DAO / SLOF) ------------------
    ap.add_argument('--dad_k', type=int, default=8,
                    help="number of nearest normal candidates in the DAO ratio")
    ap.add_argument('--dad_lid', type=str, default='est', choices=['est', 'one'],
                    help="'est' = DAO (exponent = the neighbour's own LID); "
                         "'one' = SLOF (exponent fixed at 1).  The DAO paper "
                         "calls the second the theoretically unjustified "
                         "assumption that every neighbour has LID == 1, so this "
                         "pair is the ablation that isolates the contribution.")
    ap.add_argument('--dad_comb', type=str, default='dao',
                    choices=['dao', 'slof', 'mul', 'r', 'med'],
                    help="'dao' = the DAO ratio itself (scale free); "
                         "'slof' = the same ratio with the exponent fixed to 1, "
                         "i.e. SLOF; "
                         "'mul' = kdist(q) * DAO (keeps the absolute level, so "
                         "a wholly anomalous image still scores high); "
                         "'r' = kdist(q) alone, the k-NN control; "
                         "'med' = kdist(q) / the class's per-position MEDIAN "
                         "bank kdist -- the low-variance version of the same "
                         "denominator, and the one that survives LID ~ 20.")
    ap.add_argument('--dad_floor', type=float, default=0.0,
                    help="floor the per-neighbour denominator at "
                         "dad_floor * (per-position class median kdist).  0 "
                         "keeps the strict paper formula; 1.0 bounds the ratio "
                         "and is what the LID~20 explosion calls for.")
    ap.add_argument('--dad_metric', type=str, default='l2',
                    choices=['l2', 'cos'],
                    help="metric used for the k-NN distances.  l2 is the true "
                         "metric on the unit sphere; 'cos' folds sqrt(2-2s) to "
                         "1-s, which scales every LID by 2 -- the same "
                         "metric-vs-reconstruction lesson as contribution (1).")
    ap.add_argument('--dad_refn', type=int, default=0,
                    help="cap on the number of anchor images that can act as "
                         "bank-side neighbours (0 = exact, use all).")
    ap.add_argument('--dad_dir', type=str, default='./cache_dad',
                    help="where the per-class bank-side (kdist, LID) tables are "
                         "cached; '' disables the disk cache.")
    ap.add_argument('--dad_post', type=float, default=0.0,
                    help="stack ② on top of whatever score the configured path "
                         "produced: s <- s * DAO**dad_post.  0 disables.  This "
                         "is how ① (reference point) and ② (statistic) combine, "
                         "since --agg dad replaces RAD's score instead.")
    ap.add_argument('--bank_item_order', nargs='+', default=None,
                    help="the class order the bank was BUILT with (i.e. the "
                         "--item_list passed to build_bank_multilayer.py).  "
                         "The bank checkpoint records no metadata, so the "
                         "per-class row blocks are recovered automatically and "
                         "verified by CLS block purity; pass this only to "
                         "override, or if auto-recovery reports it is unsure.")

    ap.add_argument('--pose_angle', type=float, default=0.0)
    ap.add_argument('--pose_shift', type=int, default=0)
    ap.add_argument('--limit_imgs', type=int, default=0)
    ap.add_argument('--shot_k', type=int, default=0,
                    help='0 = full training set (default); k>0 = k normal images per '
                         'class are kept in the memory bank (few-shot protocol)')
    ap.add_argument('--shot_seed', type=int, default=-1,
                    help='-1 = take the first k training images of each class '
                         '(deterministic); >=0 = draw the k images at random with '
                         'this seed (repeat over several seeds and average)')
    ap.add_argument('--skip_bank_check', type=int, default=0,
                    help='1 = 允许用 --item_list 子集跑（bank 仍是全集，仅跳过行数自检）')
    args = ap.parse_args()

    setup_seed(1)
    os.makedirs(os.path.join(args.save_dir, args.save_name), exist_ok=True)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    item_list = resolve_item_list(args.data_path, args.item_list,
                                  layout=args.layout, print_fn=print)
    print('item_list =', item_list)
    print('config =', json.dumps({k: v for k, v in vars(args).items()
                                  if k not in ('encoder_weight',)}, ensure_ascii=False))

    data_transform, gt_transform = get_data_transforms(args.image_size, args.crop_size)

    train_img_paths, test_data_list, train_data_list = [], [], []
    for i, item in enumerate(item_list):
        if args.layout == 'realiad':
            tr = RealIADDataset(root=args.data_path, category=item,
                                transform=data_transform,
                                gt_transform=gt_transform, phase='train')
            te = RealIADDataset(root=args.data_path, category=item,
                                transform=data_transform,
                                gt_transform=gt_transform, phase='test')
            train_data_list.append(tr)
            test_data_list.append(te)
            train_img_paths += list(tr.img_paths)
            continue
        tr = ImageFolder(root=os.path.join(args.data_path, item, 'train'),
                         transform=data_transform)
        tr.classes = item
        tr.class_to_idx = {item: i}
        tr.samples = [(s[0], i) for s in tr.samples]
        train_data_list.append(tr)
        test_data_list.append(MVTecDataset(
            root=os.path.join(args.data_path, item), transform=data_transform,
            gt_transform=gt_transform, phase='test'))
        train_img_paths += [p for p, _ in tr.samples]
    print('train image number (for bank):', len(train_img_paths))

    encoder = load_dinov3_model(args.encoder_name,
                                pretrained_weight_path=args.encoder_weight)
    encoder = encoder.to(device)
    for p in encoder.parameters():
        p.requires_grad = False
    encoder.eval()

    bank_ckpt, load_mode = load_bank(args.bank_path, use_mmap=bool(args.load_mmap))
    layer_indices = bank_ckpt['layers']
    cls_banks = bank_ckpt['cls_banks']
    patch_banks = bank_ckpt['patch_banks']
    print(f'bank layers {layer_indices} [{load_mode}]')

    try:
        n_bank, bank_L = validate_bank_consistency(
            patch_banks, len(train_img_paths), item_list, args.data_path, args.bank_path,
            args.crop_size, getattr(encoder, 'patch_size', None),
            max(args.k_image, args.ot_k0))
    except SystemExit as exc:
        if args.skip_bank_check:
            n_bank = int(patch_banks[0].shape[0])
            bank_L = int(patch_banks[0].shape[1])
            print('[bank 自检已跳过 --skip_bank_check=1]')
        else:
            raise
    print(f'bank ok: N={n_bank}, L={bank_L}')

    # ---- per-class bank row ranges -----------------------------------------
    # The checkpoint holds ONE bank covering every class of the dataset (and
    # does not record the class order), while the CLS top-K retrieval is what
    # separates the classes at run time.  DAD's bank-side null (kdist / LID of
    # every normal patch) must nonetheless be built per class: a patch of class
    # A at position p has its true neighbours among class A's images at
    # position p, and pooling classes would inflate kdist(o) and silently
    # deflate the ratio.
    #
    # These are GLOBAL row ranges over ALL classes on disk, so a subset run
    # (--item_list grid) still resolves grid's true offset.  Deriving them from
    # the --item_list subset -- or from the alphabetical disk order -- is
    # wrong, see _bank_class_order for the measurements.
    _, _bank_counts, _, _bank_rows_of = _bank_class_order(
        args.data_path, args.layout, args.bank_path, item_list, cls_banks,
        device, explicit=args.bank_item_order, verbose=True)
    class_rows_all = {it: torch.arange(int(_bank_rows_of[it][0]),
                                       int(_bank_rows_of[it][1]),
                                       dtype=torch.long)
                      for it in item_list}
    _tot = sum(_bank_counts.values())
    if _tot != int(patch_banks[0].shape[0]):
        print(f'[class-blocks] WARNING: classes on disk account for {_tot} '
              f'images but the bank has {int(patch_banks[0].shape[0])} rows; '
              f'the blocks are verified by CLS purity, not by this count.')

    # ---- few-shot restriction: keep only the first k training rows of each class
    # Uses the same GLOBAL blocks, so it is correct for subset runs too (an
    # equivalent block derived from the --item_list subset was off by the rows
    # of every class that precedes the requested ones).
    allowed_idx = None
    if args.shot_k and args.shot_k > 0:
        rng = np.random.RandomState(args.shot_seed) if args.shot_seed >= 0 else None
        keep = []
        for item in item_list:
            lo, hi = _bank_rows_of[item]
            kk = min(args.shot_k, int(hi - lo))
            if rng is None:
                pick = list(range(lo, lo + kk))
            else:
                pick = sorted(rng.choice(np.arange(lo, hi), size=kk,
                                         replace=False).tolist())
            keep += pick
        allowed_idx = torch.as_tensor(keep, dtype=torch.long, device=device)
        print(f'few-shot: {args.shot_k}-shot/class (seed={args.shot_seed}) -> '
              f'{len(keep)} bank rows (of {int(patch_banks[0].shape[0])})')

    total_bank_bytes = sum(b.numel() * b.element_size() for b in patch_banks)
    if args.gpu_bank_cache_mb < 0:
        budget = total_bank_bytes
    elif args.gpu_bank_cache_mb > 0:
        budget = args.gpu_bank_cache_mb * 1024 ** 2
    else:
        free, total = torch.cuda.mem_get_info()
        budget = min(int(0.55 * free), total_bank_bytes)
    budget = max(budget, min(total_bank_bytes, 4 * max(args.k_image, args.ot_k0)
                             * patch_banks[0].shape[1] * patch_banks[0].shape[2] * 4))
    bank_cache = GPUPatchBankCache(patch_banks, device, budget_bytes=budget,
                                   normalize=bool(args.normalize_bank))
    print(f'patch-bank {total_bank_bytes / 2**30:.2f} GiB | GPU cache '
          f'{budget / 2**30:.2f} GiB | cap/layer {bank_cache.cap}')

    agg = dict(auroc_sp=[], ap_sp=[], f1_sp=[], auroc_px=[], ap_px=[], f1_px=[],
               aupro_px=[], mean_normal_score=[])
    per_class = {}
    t_all = time.time()
    for _ci, (item, test_data) in enumerate(zip(item_list, test_data_list)):
        dl = DataLoader(test_data, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers)
        t_cls = time.time()
        print(f'--- {item}: {len(test_data)} test images ---', flush=True)
        _dad_rows = class_rows_all.get(item)
        if _dad_rows is None and (args.agg in ('dad', 'cfar')
                                  or args.dad_post != 0.0
                                  or args.cfar_post != 0.0):
            raise SystemExit(
                f'[class-blocks] cannot locate the bank rows of class {item!r}; '
                f'--agg dad / --dad_post would read another class\'s statistics')
        if _dad_rows is not None and allowed_idx is not None:
            _keep = set(int(v) for v in allowed_idx.detach().cpu().tolist())
            _m = torch.tensor([int(v) in _keep for v in _dad_rows.tolist()],
                              dtype=torch.bool)
            _dad_rows = _dad_rows[_m]
        r = eval_curvot(
            encoder, layer_indices, cls_banks, patch_banks, dl, device,
            k_image=args.k_image, resize_mask=args.resize_mask,
            max_ratio=args.max_ratio,
            use_positional_bank=bool(args.use_positional_bank),
            pos_radius=args.pos_radius, layer_weights=args.layer_weights,
            bank_cache=bank_cache, knn_chunk_imgs=args.knn_chunk_imgs,
            global_chunk_imgs=args.global_chunk_imgs,
            use_aptr=bool(args.use_aptr), ot_k0=args.ot_k0, ot_kr=args.ot_kr,
            ot_mass=args.ot_mass, ot_reg=args.ot_reg, ot_lam_pos=args.ot_lam_pos,
            ot_iter=args.ot_iter, coarse_grid=args.coarse_grid,
            use_lonm=bool(args.use_lonm), lonm_M=args.lonm_M,
            lonm_dmax=args.lonm_dmax, lonm_energy=args.lonm_energy,
            lonm_ridge=args.lonm_ridge, lonm_trust=args.lonm_trust,
            lonm_quadratic=bool(args.lonm_quadratic),
            lonm_ridge_rel=args.lonm_ridge_rel, lonm_res_rank=args.lonm_res_rank,
            lonm_curv_gain=args.lonm_curv_gain, lonm_norm=bool(args.lonm_norm),
            lonm_scale_q=args.lonm_scale_q,
            lonm_radial=bool(args.lonm_radial),
            lonm_norm_mode=args.lonm_norm_mode,
            lonm_scale_beta=args.lonm_scale_beta,
            lonm_maha=bool(args.lonm_maha),
            lonm_maha_ridge=args.lonm_maha_ridge,
            lonm_maha_rank=args.lonm_maha_rank,
            anc_beta=args.anc_beta,
            lonm_fast=bool(args.lonm_fast),
            lonm_unbounded=bool(args.lonm_unbounded),
            lonm_clipstat=bool(args.lonm_clipstat),
            baseline=args.baseline, baseline_d=args.baseline_d,
            baseline_power=args.baseline_power,
            baseline_padim_shrink=args.baseline_padim_shrink,
            baseline_padim_ridge_rel=args.baseline_padim_ridge_rel,
            baseline_padim_ridge_abs=args.baseline_padim_ridge_abs,
            baseline_padim_sqrt=args.baseline_padim_sqrt,
            baseline_pc_ratio=args.baseline_pc_ratio,
            baseline_pc_k=args.baseline_pc_k, baseline_pc_rw=args.baseline_pc_rw,
            baseline_pc_cap=args.baseline_pc_cap,
            baseline_pc_proj=args.baseline_pc_proj,
            baseline_pc_subcap=args.baseline_pc_subcap,
            baseline_anoco_lambda=args.baseline_anoco_lambda,
            baseline_anoco_anchor=bool(args.baseline_anoco_anchor),
            use_transport_window=bool(args.use_transport_window),
            max_shift=args.max_shift,
            use_unmatched=bool(args.use_unmatched),
            unmatched_w=args.unmatched_w,
            layer_fuse=args.layer_fuse,
            img_cal=args.img_cal, img_cal_q=args.img_cal_q,
            img_cal_ref=args.img_cal_ref, img_cal_alpha=args.img_cal_alpha,
            img_cal_scope=args.img_cal_scope, img_cal_stats=args.img_cal_stats,
            img_cal_red=args.img_cal_red, img_cal_red_q=args.img_cal_red_q,
            img_cal_scale_q=args.img_cal_scale_q,
            agg=args.agg, ia_K=args.ia_K,
            dad_k=args.dad_k, dad_lid=args.dad_lid, dad_comb=args.dad_comb,
            dad_floor=args.dad_floor, cfar_comb=args.cfar_comb,
            cfar_post=args.cfar_post,
            band_mode=args.band_mode, band_w=args.band_w,
            band_beta=args.band_beta,
            dad_metric=args.dad_metric, dad_refn=args.dad_refn,
            dad_dir=args.dad_dir, dad_rows=_dad_rows, item_name=str(item),
            dad_post=args.dad_post,
            use_centroid=args.use_centroid,
            pose_angle=args.pose_angle, pose_shift=args.pose_shift,
            limit_imgs=args.limit_imgs, allowed_idx=allowed_idx)
        per_class[item] = r
        for k in agg:
            agg[k].append(r[k])
        print(f'{item}: I-AUROC:{r["auroc_sp"]:.4f} I-AP:{r["ap_sp"]:.4f} '
              f'I-F1:{r["f1_sp"]:.4f} P-AUROC:{r["auroc_px"]:.4f} '
              f'P-AP:{r["ap_px"]:.4f} P-F1:{r["f1_px"]:.4f} '
              f'P-AUPRO:{r["aupro_px"]:.4f} normS:{r["mean_normal_score"]:.4f} '
              f'[{time.time() - t_cls:.1f}s]', flush=True)

    mean = {k: float(np.nanmean(v)) for k, v in agg.items()}
    print('Mean: ' + ', '.join(f'{k}:{v:.4f}' for k, v in mean.items()))
    print(f'total time: {time.time() - t_all:.1f}s | '
          f'peak GPU mem: {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB')

    out = {'args': {k: v for k, v in vars(args).items()}, 'mean': mean,
           'per_class': per_class}
    with open(os.path.join(args.save_dir, args.save_name, 'metrics.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print('saved ->', os.path.join(args.save_dir, args.save_name, 'metrics.json'))


if __name__ == '__main__':
    main()
