#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_unet.py
-------------
U-Net training for BraTS slices (T1ce input) with options:
- Loss: CrossEntropy (default) or Dice (multi-class, one-hot targets)
- LR scheduler: ReduceLROnPlateau
- Early stopping on best validation mean-foreground Dice
- TensorBoard logging
- AMP support
- Validation visualization: save PNGs with prediction/GT overlays
"""
from pathlib import Path
import argparse
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

try:
    import albumentations as A
except Exception:
    A = None

from dataset_brats import BrainTumorMRIData

# ----------------- U-Net -----------------
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)

class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_ch, out_ch))
    def forward(self, x): return self.net(x)

class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = DoubleConv(in_ch, out_ch)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size(2) - x1.size(2)
        diffX = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [diffX//2, diffX - diffX//2, diffY//2, diffY - diffY//2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=4, base_ch=32):
        super().__init__()
        self.inc   = DoubleConv(in_channels, base_ch)
        self.down1 = Down(base_ch, base_ch*2)
        self.down2 = Down(base_ch*2, base_ch*4)
        self.down3 = Down(base_ch*4, base_ch*8)
        self.up1   = Up(base_ch*8 + base_ch*4, base_ch*4)
        self.up2   = Up(base_ch*4 + base_ch*2, base_ch*2)
        self.up3   = Up(base_ch*2 + base_ch,   base_ch)
        self.outc  = nn.Conv2d(base_ch, num_classes, kernel_size=1)
    def forward(self, x):
        x1 = self.inc(x); x2 = self.down1(x1); x3 = self.down2(x2); x4 = self.down3(x3)
        x  = self.up1(x4, x3); x = self.up2(x, x2); x = self.up3(x, x1)
        return self.outc(x)

# ----------------- Losses -----------------
class DiceLoss(nn.Module):
    """
    Multi-class soft Dice loss on one-hot targets (N,C,H,W).
    Excludes background from the average by default.
    """
    def __init__(self, eps: float = 1e-6, include_bg: bool = False):
        super().__init__()
        self.eps = eps
        self.include_bg = include_bg
    def forward(self, logits: torch.Tensor, target_onehot: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=1)
        dims = (0, 2, 3)
        inter = (probs * target_onehot).sum(dims)                          # (C,)
        denom = (probs * probs).sum(dims) + target_onehot.sum(dims)        # (C,)
        dice  = (2 * inter + self.eps) / (denom + self.eps)                # (C,)
        return 1.0 - (dice.mean() if self.include_bg else dice[1:].mean()) # skip bg

# ----------------- Metrics -----------------
@torch.no_grad()
def dice_per_class(logits, target, eps=1e-6):
    """
    logits: (N,C,H,W), target: (N,H,W) int labels 0..C-1
    returns: (dict per-class dice, mean foreground dice)
    """
    C = logits.shape[1]
    probs = F.softmax(logits, dim=1)
    dices = {}
    fg_dices = []
    for c in range(C):
        p = probs[:, c]            # (N,H,W)
        t = (target == c).float()  # (N,H,W)
        inter = (p * t).sum(dim=(1,2)) * 2.0
        denom = (p * p).sum(dim=(1,2)) + t.sum(dim=(1,2)) + eps
        d = (inter / denom).mean().item()
        dices[c] = d
        if c != 0: fg_dices.append(d)
    mean_fg = float(np.mean(fg_dices)) if fg_dices else 0.0
    return dices, mean_fg

# ----------------- Visualization -----------------
def colorize_labels(lbl: np.ndarray, colors=((1,0,0),(1,1,0),(0,1,0))):
    """lbl: (H,W) in {0,1,2,3}; returns RGB (H,W,3) in 0..1 (no blending)."""
    h, w = lbl.shape
    rgb = np.zeros((h,w,3), dtype=np.float32)
    for i, col in enumerate(colors, start=1):
        rgb[lbl == i] = np.array(col, dtype=np.float32)
    return rgb

def overlay(base_gray: np.ndarray, lbl: np.ndarray, alpha=0.35, colors=((1,0,0),(1,1,0),(0,1,0))):
    """Blend colored labels onto gray base image."""
    base_rgb = np.repeat(base_gray[..., None], 3, axis=2)
    color = colorize_labels(lbl, colors=colors)
    a = (lbl > 0).astype(np.float32) * alpha
    out = (1 - a[..., None]) * base_rgb + a[..., None] * color
    return np.clip(out, 0, 1)

def save_val_visuals(x, y_lbl, y_pred_lbl, save_dir: Path, epoch: int, max_samples: int = 8):
    """
    x: (N,1,H,W) float in [0,1]
    y_lbl / y_pred_lbl: (N,H,W) long
    Saves: input | GT | Pred | Overlay GT | Overlay Pred
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    N = min(max_samples, x.size(0))
    for i in range(N):
        base = x[i,0].detach().cpu().numpy()
        gt   = y_lbl[i].detach().cpu().numpy()
        pd   = y_pred_lbl[i].detach().cpu().numpy()
        ov_gt = overlay(base, gt)
        ov_pd = overlay(base, pd)
        fig = plt.figure(figsize=(12, 3))
        ax1 = plt.subplot(1,5,1); ax1.imshow(base, cmap='gray'); ax1.set_title('T1ce'); ax1.axis('off')
        ax2 = plt.subplot(1,5,2); ax2.imshow(colorize_labels(gt)); ax2.set_title('GT'); ax2.axis('off')
        ax3 = plt.subplot(1,5,3); ax3.imshow(colorize_labels(pd)); ax3.set_title('Pred'); ax3.axis('off')
        ax4 = plt.subplot(1,5,4); ax4.imshow(ov_gt); ax4.set_title('Overlay GT'); ax4.axis('off')
        ax5 = plt.subplot(1,5,5); ax5.imshow(ov_pd); ax5.set_title('Overlay Pred'); ax5.axis('off')
        fig.tight_layout()
        out_path = save_dir / f"epoch{epoch:03d}_idx{i:02d}.png"
        fig.savefig(out_path, dpi=140, bbox_inches='tight')
        plt.close(fig)

# ----------------- Train / Eval -----------------
def train_one_epoch_ce(model, loader, optimizer, device, scaler=None):
    model.train()
    total = 0.0
    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)   # (N,1,H,W)
        y = batch["mask"].to(device, non_blocking=True)    # (N,H,W)
        optimizer.zero_grad(set_to_none=True)
        if scaler is not None and scaler.is_enabled():
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                logits = model(x)
                loss = F.cross_entropy(logits, y, ignore_index=-100)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
        else:
            logits = model(x); loss = F.cross_entropy(logits, y, ignore_index=-100)
            loss.backward(); optimizer.step()
        total += loss.item() * x.size(0)
    return total / len(loader.dataset)

def train_one_epoch_dice(model, loader, optimizer, device, scaler=None, dice_loss=None):
    model.train()
    total = 0.0
    assert dice_loss is not None
    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)    # (N,1,H,W)
        y_oh = batch["mask"].to(device, non_blocking=True)  # (N,3,H,W) one-hot (no background)
        # build full one-hot with background channel
        bg = (y_oh.sum(dim=1, keepdim=True) <= 0.5).float() # (N,1,H,W)
        y_oh_full = torch.cat([bg, y_oh], dim=1)            # (N,4,H,W)
        optimizer.zero_grad(set_to_none=True)
        if scaler is not None and scaler.is_enabled():
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                logits = model(x)
                loss = dice_loss(logits, y_oh_full)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
        else:
            logits = model(x); loss = dice_loss(logits, y_oh_full)
            loss.backward(); optimizer.step()
        total += loss.item() * x.size(0)
    return total / len(loader.dataset)

@torch.no_grad()
def evaluate_common(model, loader, device):
    """Compute CE loss for reporting + dice metrics, and cache first batch for visualization."""
    model.eval()
    total = 0.0
    all_fg = []
    per_sum = None
    n = 0
    x_first = y_first = pred_first = None
    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        y = batch["mask"].to(device, non_blocking=True)
        logits = model(x)
        # convert one-hot targets to labels if needed
        if y.ndim == 4:  # (N,3,H,W)
            bg = (y.sum(dim=1) <= 0.5)
            lbl = torch.argmax(y, dim=1) + 1
            lbl[bg] = 0
            y_lbl = lbl
        else:
            y_lbl = y

        loss = F.cross_entropy(logits, y_lbl, ignore_index=-100)
        total += loss.item() * x.size(0)

        dices, mean_fg = dice_per_class(logits, y_lbl)
        if per_sum is None: per_sum = {k: 0.0 for k in dices}
        for k, v in dices.items(): per_sum[k] += v
        all_fg.append(mean_fg)
        n += 1

        if x_first is None:
            x_first = x.detach().cpu()
            y_first = y_lbl.detach().cpu()
            pred_first = torch.argmax(logits, dim=1).detach().cpu()

    avg_loss = total / len(loader.dataset)
    avg_fg = float(np.mean(all_fg)) if all_fg else 0.0
    per_avg = {k: v / n for k, v in per_sum.items()} if per_sum else {}
    return avg_loss, avg_fg, per_avg, x_first, y_first, pred_first

# ----------------- Utils -----------------
def read_list(txt: Path):
    with txt.open('r', encoding='utf-8') as f:
        return [Path(line.strip()) for line in f if line.strip()]

def build_joint_transform(crop=240):
    if A is None: return None
    return A.Compose([
        A.PadIfNeeded(args.crop, args.crop, border_mode=0, pad_val=0, pad_val_mask=0),
        A.RandomCrop(crop, crop),
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
    ])

# ----------------- Main -----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lists', type=str, required=True, help='Dir containing train.txt/val.txt')
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--num-workers', type=int, default=4)
    ap.add_argument('--amp', action='store_true')
    ap.add_argument('--logdir', type=str, default='runs/unet_t1ce')
    ap.add_argument('--outdir', type=str, default='outputs/unet_t1ce')
    ap.add_argument('--crop', type=int, default=240)
    ap.add_argument('--t1ce-idx', type=int, default=1)
    ap.add_argument('--patience', type=int, default=10)
    ap.add_argument('--loss', type=str, choices=['ce', 'dice'], default='ce', help='Loss type')
    ap.add_argument('--vis-samples', type=int, default=8, help='How many validation images to save per epoch')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lists_dir = Path(args.lists).resolve()
    outdir = Path(args.outdir).resolve(); outdir.mkdir(parents=True, exist_ok=True)
    vis_dir = outdir / 'vis'; vis_dir.mkdir(exist_ok=True)

    train_files = read_list(lists_dir/'train.txt')
    val_files   = read_list(lists_dir/'val.txt')

    jt = build_joint_transform(args.crop)
    if args.loss == 'dice':
        train_ds = BrainTumorMRIData(train_files, t1ce_idx=args.t1ce_idx, return_onehot=True,  joint_transform=jt)
        val_ds   = BrainTumorMRIData(val_files,   t1ce_idx=args.t1ce_idx, return_onehot=True,  joint_transform=None)
    else:
        train_ds = BrainTumorMRIData(train_files, t1ce_idx=args.t1ce_idx, return_onehot=False, joint_transform=jt)
        val_ds   = BrainTumorMRIData(val_files,   t1ce_idx=args.t1ce_idx, return_onehot=False, joint_transform=None)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

    model = UNet(in_channels=1, num_classes=4, base_ch=32).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    writer = SummaryWriter(log_dir=args.logdir)

    train_step = (lambda *a, **kw: train_one_epoch_dice(*a, **kw, dice_loss=DiceLoss(include_bg=False))
                  ) if args.loss == 'dice' else train_one_epoch_ce

    best_fg = -math.inf
    bad_epochs = 0

    for epoch in range(1, args.epochs+1):
        tr_loss = train_step(model, train_loader, optimizer, device, scaler=scaler)
        val_loss, val_fg, per_cls, x_first, y_first, pred_first = evaluate_common(model, val_loader, device)
        scheduler.step(val_loss)

        writer.add_scalar('loss/train', tr_loss, epoch)
        writer.add_scalar('loss/val', val_loss, epoch)
        writer.add_scalar('dice_fg/val', val_fg, epoch)
        for k, v in per_cls.items(): writer.add_scalar(f'dice_class_{k}/val', v, epoch)
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)

        print(f"[Epoch {epoch:03d}] train_loss={tr_loss:.4f} val_loss={val_loss:.4f} "
              f"val_fg_dice={val_fg:.4f} per_class={per_cls} lr={optimizer.param_groups[0]['lr']:.2e}")

        # 保存验证可视化
        if x_first is not None:
            save_val_visuals(x_first, y_first, pred_first, save_dir=vis_dir, epoch=epoch, max_samples=args.vis_samples)

        # 早停：以前景 Dice 均值为准
        if val_fg > best_fg + 1e-6:
            best_fg = val_fg
            bad_epochs = 0
            torch.save({'model': model.state_dict(), 'epoch': epoch, 'val_fg_dice': best_fg}, outdir / 'best.pt')
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"Early stopping at epoch {epoch}. Best val_fg_dice={best_fg:.4f}.")
                break

    writer.close()
    print(f"Done. Best val_fg_dice={best_fg:.4f}. Checkpoints & visuals in {outdir}")

if __name__ == '__main__':
    main()
