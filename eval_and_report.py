#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_and_report.py (with HD95 + ASSD + Boundary F-score)
Evaluate a trained generator (pix2pix-style or plain UNet) on BraTS slices and
export metrics (IoU, Dice, HD95, ASSD, Boundary F-score), CSVs, LaTeX table,
and qualitative montages.

Usage (example)
--------------
python eval_and_report.py \
  --lists ./splits_50 --split val \
  --gen unet --checkpoint outputs/pix2pix_unet/best_unet.pt \
  --outdir reports/unet_val --bf-tol 2.0 --spacing 1 1
"""
from pathlib import Path
import argparse, re, math, csv
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List

from dataset_brats import BrainTumorMRIData
from models_pix2pix import EncDecGAN, UNetGAN, TLUNetGAN

# ----------------------------- helpers -----------------------------
H5_RE = re.compile(r"volume_(\d+)_slice_(\d+)\.h5$")

def parse_ids(p: Path) -> Tuple[int, int]:
    m = H5_RE.search(p.name)
    if not m:
        return (-1, -1)
    return int(m.group(1)), int(m.group(2))

@torch.no_grad()
def iou_per_class(pred: np.ndarray, gt: np.ndarray, C: int) -> Dict[int, float]:
    out = {}
    for c in range(C):
        p = (pred == c)
        t = (gt == c)
        inter = (p & t).sum()
        union = (p | t).sum()
        out[c] = (inter / (union + 1e-8)) if union > 0 else (1.0 if inter==0 else 0.0)
    return out

@torch.no_grad()
def dice_per_class(pred: np.ndarray, gt: np.ndarray, C: int) -> Dict[int, float]:
    out = {}
    for c in range(C):
        p = (pred == c).astype(np.float32)
        t = (gt == c).astype(np.float32)
        inter = 2.0 * (p*t).sum()
        denom = p.sum() + t.sum()
        out[c] = (inter / (denom + 1e-8)) if denom > 0 else 1.0
    return out

# ----------- Surfaces & distance metrics (HD95 / ASSD / BF) -----------
from scipy.ndimage import distance_transform_edt, binary_erosion, generate_binary_structure

def _surface(mask: np.ndarray) -> np.ndarray:
    """Binary 2D surface (1-pixel boundary)."""
    b = mask.astype(bool)
    if not b.any():
        return np.zeros_like(b, dtype=bool)
    inner = binary_erosion(b, structure=generate_binary_structure(2,1), border_value=0)
    return b ^ inner

def _surface_distances(mask_pred: np.ndarray, mask_true: np.ndarray, spacing=(1.0,1.0)) -> np.ndarray:
    """All point-to-surface distances from pred<->true surfaces, concatenated."""
    s1 = _surface(mask_pred)
    s2 = _surface(mask_true)
    if not s1.any() and not s2.any():
        return np.array([0.0], dtype=np.float32)
    if not s1.any(): s1 = s2
    if not s2.any(): s2 = s1
    dt1 = distance_transform_edt(~s1, sampling=spacing)
    dt2 = distance_transform_edt(~s2, sampling=spacing)
    d12 = dt2[s1]  # pred->gt
    d21 = dt1[s2]  # gt->pred
    return np.concatenate([d12, d21]).astype(np.float32)

def hd95_binary(pred_bin: np.ndarray, gt_bin: np.ndarray, spacing=(1.0,1.0)) -> float:
    if pred_bin.max() == 0 and gt_bin.max() == 0:
        return 0.0
    d = _surface_distances(pred_bin>0, gt_bin>0, spacing=spacing)
    return float(np.percentile(d, 95.0)) if d.size else 0.0

def assd_binary(pred_bin: np.ndarray, gt_bin: np.ndarray, spacing=(1.0,1.0)) -> float:
    """Average Symmetric Surface Distance."""
    if pred_bin.max() == 0 and gt_bin.max() == 0:
        return 0.0
    d = _surface_distances(pred_bin>0, gt_bin>0, spacing=spacing)
    return float(np.mean(d)) if d.size else 0.0

def bfscore_binary(pred_bin: np.ndarray, gt_bin: np.ndarray, spacing=(1.0,1.0), tol: float = 2.0) -> float:
    """
    Boundary F-score with tolerance `tol`.
    Precision: fraction of predicted boundary points within tol of GT boundary.
    Recall:    fraction of GT boundary points within tol of predicted boundary.
    """
    s_pred = _surface(pred_bin>0)
    s_gt   = _surface(gt_bin>0)
    # handle empty boundaries
    if not s_pred.any() and not s_gt.any():
        return 1.0
    if not s_pred.any() or not s_gt.any():
        return 0.0

    dt_gt   = distance_transform_edt(~s_gt, sampling=spacing)  # distance to GT boundary
    dt_pred = distance_transform_edt(~s_pred, sampling=spacing)

    # precision: predicted boundary close to GT boundary
    hit_p = (dt_gt[s_pred] <= tol).sum()
    tot_p = s_pred.sum()
    precision = float(hit_p / (tot_p + 1e-8))

    # recall: GT boundary close to predicted boundary
    hit_r = (dt_pred[s_gt] <= tol).sum()
    tot_r = s_gt.sum()
    recall = float(hit_r / (tot_r + 1e-8))

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall + 1e-8)

def hd95_per_class(pred: np.ndarray, gt: np.ndarray, C: int, spacing=(1.0,1.0)) -> Dict[int, float]:
    return {c: hd95_binary(pred==c, gt==c, spacing=spacing) for c in range(C)}

def assd_per_class(pred: np.ndarray, gt: np.ndarray, C: int, spacing=(1.0,1.0)) -> Dict[int, float]:
    return {c: assd_binary(pred==c, gt==c, spacing=spacing) for c in range(C)}

def bf_per_class(pred: np.ndarray, gt: np.ndarray, C: int, spacing=(1.0,1.0), tol: float = 2.0) -> Dict[int, float]:
    return {c: bfscore_binary(pred==c, gt==c, spacing=spacing, tol=tol) for c in range(C)}

# ---------------------- visualization (montage) ----------------------
def colorize(lbl: np.ndarray, colors=((1,0,0),(1,1,0),(0,1,0))):
    h,w = lbl.shape; rgb = np.zeros((h,w,3), np.float32)
    for i,c in enumerate(colors,1): rgb[lbl==i] = np.array(c,np.float32)
    return rgb

def overlay(gray: np.ndarray, lbl: np.ndarray, alpha=0.35):
    base = np.repeat(gray[...,None], 3, 2)
    col = colorize(lbl)
    a = (lbl>0).astype(np.float32)*alpha
    out = (1-a[...,None])*base + a[...,None]*col
    return np.clip(out,0,1)

def save_montage(examples: List[Tuple[np.ndarray,np.ndarray,np.ndarray]], out_path: Path, cols=4):
    import math
    n = len(examples)
    rows = math.ceil(n / cols)
    fig = plt.figure(figsize=(3.6*cols, 3.2*rows))
    k = 1
    for (g,gt,pd) in examples:
        for img, title in [(g,'T1ce'), (colorize(gt),'GT'), (colorize(pd),'Pred'),
                           (overlay(g,gt),'Overlay GT'), (overlay(g,pd),'Overlay Pred')]:
            ax = fig.add_subplot(rows, cols, k); k += 1
            if img.ndim==2: ax.imshow(img, cmap='gray')
            else: ax.imshow(img)
            ax.set_title(title, fontsize=10); ax.axis('off')
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches='tight')
    plt.close(fig)

# ---------------------- model factory ----------------------
def build_generator(name, ckpt_path: Path, device, in_ch=1, out_ch=4, freeze_ratio=0.75):
    if name == 'baseline':
        G = EncDecGAN(in_ch, out_ch, base=64, use_skip=False).to(device)
        state = torch.load(ckpt_path, map_location=device)
        G.load_state_dict(state.get('G', state.get('model', state)))
    elif name == 'unet':
        G = UNetGAN(in_ch, out_ch, base=64).to(device)
        state = torch.load(ckpt_path, map_location=device)
        G.load_state_dict(state.get('G', state.get('model', state)))
    elif name == 'tl_unet':
        G = TLUNetGAN(in_ch, out_ch, base=64, freeze_ratio=freeze_ratio).to(device)
        state = torch.load(ckpt_path, map_location=device)
        G.load_state_dict(state.get('G', state.get('model', state)), strict=False)
    else:
        raise ValueError(f"Unknown generator: {name}")
    G.eval()
    return G

# ---------------------- main eval ----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lists', type=str, required=True, help='Dir containing train/val/test.txt')
    ap.add_argument('--split', type=str, choices=['train','val','test'], default='val')
    ap.add_argument('--gen', type=str, choices=['baseline','unet','tl_unet'], default='unet')
    ap.add_argument('--checkpoint', type=str, required=True)
    ap.add_argument('--outdir', type=str, required=True)
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--num-workers', type=int, default=4)
    ap.add_argument('--spacing', type=float, nargs=2, default=(1.0,1.0), help='pixel spacing: (sx, sy)')
    ap.add_argument('--bf-tol', type=float, default=2.0, help='Boundary F-score tolerance (same unit as spacing)')  # NEW
    ap.add_argument('--vis-k', type=int, default=6)
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lists_dir = Path(args.lists)
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    vis_dir = outdir / 'montage'; vis_dir.mkdir(exist_ok=True)

    # dataset / loader
    files_txt = lists_dir / f"{args.split}.txt"
    ds = BrainTumorMRIData([Path(x) for x in files_txt.read_text().splitlines() if x.strip()],
                           return_onehot=False, joint_transform=None)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers, pin_memory=True)
    C = 4
    G = build_generator(args.gen, Path(args.checkpoint), device)

    # collectors
    slice_rows = []  # per-slice metrics
    by_volume: Dict[int, Dict[str, list]] = {}

    with torch.no_grad():
        for batch in dl:
            x = batch['image'].to(device)  # (N,1,H,W)
            y = batch['mask'].to(device)   # (N,H,W)
            logits = G(x)
            pred = torch.argmax(logits, dim=1)

            x_np = x.cpu().numpy()
            y_np = y.cpu().numpy()
            p_np = pred.cpu().numpy()

            for i, pth in enumerate(batch['path']):
                vol_id, sl_id = parse_ids(Path(pth))
                C_iou  = iou_per_class(p_np[i], y_np[i], C)
                C_dice = dice_per_class(p_np[i], y_np[i], C)
                C_hd   = hd95_per_class(p_np[i], y_np[i], C, spacing=tuple(args.spacing))
                C_assd = assd_per_class(p_np[i], y_np[i], C, spacing=tuple(args.spacing))                      # NEW
                C_bf   = bf_per_class(p_np[i], y_np[i], C, spacing=tuple(args.spacing), tol=args.bf_tol)       # NEW

                mean_iou_fg  = float(np.mean([C_iou[c]  for c in range(1,C)]))
                mean_dice_fg = float(np.mean([C_dice[c] for c in range(1,C)]))
                mean_hd_fg   = float(np.mean([C_hd[c]   for c in range(1,C)]))
                mean_assd_fg = float(np.mean([C_assd[c] for c in range(1,C)]))                                  # NEW
                mean_bf_fg   = float(np.mean([C_bf[c]   for c in range(1,C)]))                                  # NEW

                row = {
                    "path": pth, "volume_id": vol_id, "slice_id": sl_id,
                    **{f"iou_c{c}":   C_iou[c]  for c in range(C)},
                    **{f"dice_c{c}":  C_dice[c] for c in range(C)},
                    **{f"hd95_c{c}":  C_hd[c]   for c in range(C)},
                    **{f"assd_c{c}":  C_assd[c] for c in range(C)},                                            # NEW
                    **{f"bf_c{c}":    C_bf[c]   for c in range(C)},                                            # NEW
                    "iou_fg":   mean_iou_fg, "dice_fg":  mean_dice_fg,
                    "hd95_fg":  mean_hd_fg, "assd_fg":  mean_assd_fg, "bf_fg": mean_bf_fg                      # NEW
                }
                slice_rows.append(row)

                b = by_volume.setdefault(vol_id, {"iou_fg":[], "dice_fg":[], "hd95_fg":[],
                                                  "assd_fg":[], "bf_fg":[],  # NEW
                                                  "gray":[], "gt":[], "pred":[]})
                b["iou_fg"].append(mean_iou_fg)
                b["dice_fg"].append(mean_dice_fg)
                b["hd95_fg"].append(mean_hd_fg)
                b["assd_fg"].append(mean_assd_fg)  # NEW
                b["bf_fg"].append(mean_bf_fg)      # NEW
                b["gray"].append(x_np[i,0]); b["gt"].append(y_np[i]); b["pred"].append(p_np[i])

    # ---- write per-slice CSV ----
    sl_csv = outdir / "metrics_slices.csv"
    if slice_rows:
        keys = list(slice_rows[0].keys())
        with sl_csv.open("w", newline='', encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(slice_rows)

    # ---- aggregate per-volume ----
    vol_rows = []
    vol_scores = []
    for vid, d in by_volume.items():
        iou  = float(np.mean(d["iou_fg"]))
        dice = float(np.mean(d["dice_fg"]))
        hd   = float(np.mean(d["hd95_fg"]))
        assd = float(np.mean(d["assd_fg"]))  # NEW
        bf   = float(np.mean(d["bf_fg"]))    # NEW
        vol_rows.append({"volume_id": vid, "iou_fg": iou, "dice_fg": dice, "hd95_fg": hd,
                         "assd_fg": assd, "bf_fg": bf})  # NEW
        vol_scores.append((vid, iou, dice, hd, assd, bf, d))  # NEW

    # write per-volume CSV
    vol_csv = outdir / "metrics_volumes.csv"
    with vol_csv.open("w", newline='', encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["volume_id","iou_fg","dice_fg","hd95_fg","assd_fg","bf_fg"])  # NEW
        w.writeheader(); w.writerows(vol_rows)

    # ---- summary (mean±std) and LaTeX ----
    def vec(key):
        return np.array([r[key] for r in vol_rows], dtype=np.float64)
    ious  = vec("iou_fg"); dices = vec("dice_fg"); hds = vec("hd95_fg")
    assds = vec("assd_fg"); bfs  = vec("bf_fg")  # NEW
    def mpm(x): return f"{np.mean(x):.3f} $\\pm$ {np.std(x):.3f}" if x.size>0 else "n/a"

    tex = outdir / "summary_table.tex"
    with tex.open("w", encoding="utf-8") as f:
        f.write("\\begin{tabular}{lccccc}\\hline\\hline\n")
        f.write("Method & IoU$_{fg}$ & Dice$_{fg}$ & HD95$_{fg}$ & ASSD$_{fg}$ & BF$_{fg}$\\\\\\hline\n")
        f.write(f"{args.gen} & {mpm(ious)} & {mpm(dices)} & {mpm(hds)} & {mpm(assds)} & {mpm(bfs)}\\\\\\hline\\hline\n")
        f.write("\\end{tabular}\n")

    # ---- montages (unchanged) ----
    vol_scores.sort(key=lambda x: x[1], reverse=True)  # by IoU desc
    K = min(args.vis_k, len(vol_scores))
    def pick_rep(case_d):
        areas = [ (i, (case_d["gt"][i]>0).sum()) for i in range(len(case_d["gt"])) ]
        idx = max(areas, key=lambda t:t[1])[0] if areas else 0
        return case_d["gray"][idx], case_d["gt"][idx], case_d["pred"][idx]
    best_cases  = [ pick_rep(d) for *_, d in vol_scores[:K] ]
    worst_cases = [ pick_rep(d) for *_, d in vol_scores[-K:][::-1] ]
    save_montage(best_cases,  outdir / "montage_best.png",  cols=K)
    save_montage(worst_cases, outdir / "montage_worst.png", cols=K)

    print(f"Done. CSVs & LaTeX in: {outdir}")
    print(f"- Slices:   {sl_csv}")
    print(f"- Volumes:  {vol_csv}")
    print(f"- LaTeX:    {tex}")
    print(f"- Montages: {outdir/'montage_best.png'}, {outdir/'montage_worst.png'}")

if __name__ == "__main__":
    main()
