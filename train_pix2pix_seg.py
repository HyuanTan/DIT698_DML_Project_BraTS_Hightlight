#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_pix2pix_seg.py
--------------------
Conditional GAN (pix2pix-style) for segmentation on BraTS slices (T1ce -> mask).
Implements three generators (baseline/unet/tl_unet) and a shared PatchGAN70.
Loss: L_total = L_GAN + λ1*L1(or Dice/Focal) + λfm*FeatureMatching (optional).
Primary metric: IoU (per-class + mean). Also reports Dice.

Usage (examples)
---------------
# Use UNet generator + default losses (GAN + L1)
python train_pix2pix_seg.py --lists ./splits_50 --gen unet --epochs 50 --outdir outputs/pix2pix_unet

# Baseline encoder-decoder (no skip) + GAN+Dice
python train_pix2pix_seg.py --lists ./splits_50 --gen baseline --aux-loss dice --lambda-aux 1.0

# Transfer-learning U-Net encoder (freeze 75%) + PatchGAN(shared), add FeatureMatching
python train_pix2pix_seg.py --lists ./splits_50 --gen tl_unet --freeze-ratio 0.75 --lambda-fm 10.0
"""
from pathlib import Path
import argparse, math, numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset_brats import BrainTumorMRIData
from models_pix2pix import EncDecGAN, UNetGAN, TLUNetGAN, PatchGAN70

from albumentations import Compose, PadIfNeeded, RandomCrop, HorizontalFlip, RandomRotate90

# ---------- helpers ----------
def read_list(txt: Path):
    with txt.open('r', encoding='utf-8') as f:
        return [Path(line.strip()) for line in f if line.strip()]

def onehot_from_labels(lbl: torch.Tensor, C: int) -> torch.Tensor:
    # lbl: (N,H,W) in {0..C-1}; return (N,C,H,W) one-hot
    return F.one_hot(lbl.long(), num_classes=C).permute(0,3,1,2).float()

@torch.no_grad()
def iou_per_class(pred, target, C):
    """
    pred/target: (N,H,W) integer labels
    returns dict{c:iou}, mean over foreground classes (1..C-1)
    """
    ious, fg = {}, []
    for c in range(C):
        p = (pred == c); t = (target == c)
        inter = (p & t).sum().item()
        union = (p | t).sum().item()
        i = inter / (union + 1e-8)
        ious[c] = i
        if c != 0: fg.append(i)
    return ious, float(np.mean(fg) if fg else 0.0)

@torch.no_grad()
def dice_per_class(pred, target, C):
    dices, fg = {}, []
    for c in range(C):
        p = (pred == c).float(); t = (target == c).float()
        inter = (2*(p*t).sum().item())
        denom = p.sum().item() + t.sum().item() + 1e-8
        d = inter/denom
        dices[c] = d
        if c != 0: fg.append(d)
    return dices, float(np.mean(fg) if fg else 0.0)

# ---------- losses ----------
class FeatureMatchingLoss(nn.Module):
    """Sum of L1 over intermediate discriminator features (wrt real vs fake)."""
    def __init__(self): super().__init__(); self.l1 = nn.L1Loss()
    def forward(self, feats_real, feats_fake):
        loss = 0.0
        for fr, ff in zip(feats_real, feats_fake):
            loss += self.l1(fr, ff)
        return loss

def bce_with_logits(pred, target):  # PatchGAN real/fake
    return F.binary_cross_entropy_with_logits(pred, target)

def build_generator(name, in_ch=1, out_ch=4, base=64, freeze_ratio=0.75):
    if name == 'baseline': return EncDecGAN(in_ch, out_ch, base, use_skip=False)
    if name == 'unet':     return UNetGAN(in_ch, out_ch, base)
    if name == 'tl_unet':  return TLUNetGAN(in_ch, out_ch, base, freeze_ratio)
    raise ValueError(f'Unknown gen: {name}')

# ---------- visualization ----------
import numpy as np, matplotlib.pyplot as plt
def colorize(lbl, colors=((1,0,0),(1,1,0),(0,1,0))):
    h,w = lbl.shape; rgb = np.zeros((h,w,3), dtype=np.float32)
    for i,c in enumerate(colors,1): rgb[lbl==i] = np.array(c, np.float32)
    return rgb
def overlay(gray, lbl, alpha=0.35):
    base = np.repeat(gray[...,None], 3, 2); col = colorize(lbl); a = (lbl>0).astype(np.float32)*alpha
    out = (1-a[...,None])*base + a[...,None]*col; return np.clip(out,0,1)
def save_vis(x, y, pred, save_dir: Path, epoch: int, max_n=8, tag='G'):
    save_dir.mkdir(parents=True, exist_ok=True)
    N = min(max_n, x.size(0))
    for i in range(N):
        g = x[i,0].cpu().numpy(); gt = y[i].cpu().numpy(); pd = pred[i].cpu().numpy()
        fig = plt.figure(figsize=(12,3))
        for j,(img,title) in enumerate([
            (g, 'Input T1ce'),
            (colorize(gt), 'GT'),
            (colorize(pd), f'{tag} Pred'),
            (overlay(g, gt), 'Overlay GT'),
            (overlay(g, pd), 'Overlay Pred')]):
            ax = plt.subplot(1,5,j+1)
            ax.imshow(img if j else img, cmap=('gray' if j==0 else None))
            ax.set_title(title); ax.axis('off')
        fig.tight_layout()
        fig.savefig(save_dir / f"epoch{epoch:03d}_idx{i:02d}.png", dpi=140, bbox_inches='tight')
        plt.close(fig)

# ---------- training ----------
class PatchGANWithFeatures(nn.Module):
    """Wrap D to return intermediate features for Feature Matching."""
    def __init__(self, D: PatchGAN70):
        super().__init__()
        self.blocks = nn.ModuleList(D.net[:-1])  # all but last conv
        self.last   = D.net[-1]
    def forward(self, x_cond, y):
        if y.shape[2:] != x_cond.shape[2:]:
            y = F.interpolate(y, size=x_cond.shape[2:], mode='bilinear', align_corners=False)
        z = torch.cat([x_cond, y], dim=1)
        feats = []
        h = z
        for m in self.blocks:
            h = m(h)
            if isinstance(m, nn.LeakyReLU):  # collect after conv+bn+act
                feats.append(h)
        logit = self.last(h)
        return logit, feats

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lists', type=str, required=True)
    ap.add_argument('--gen', type=str, choices=['baseline','unet','tl_unet'], default='unet')
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--lrG', type=float, default=2e-4)
    ap.add_argument('--lrD', type=float, default=2e-4)
    ap.add_argument('--num-workers', type=int, default=4)
    ap.add_argument('--amp', action='store_true')
    ap.add_argument('--outdir', type=str, default='outputs/pix2pix')
    ap.add_argument('--logdir', type=str, default='runs')
    ap.add_argument('--lambda-aux', type=float, default=1.0, help='λ1 for aux loss (L1/Dice/Focal)')
    ap.add_argument('--lambda-fm', type=float, default=0.0, help='λfm for Feature Matching')
    ap.add_argument('--aux-loss', choices=['l1','dice','focal'], default='l1')
    ap.add_argument('--freeze-ratio', type=float, default=0.75, help='for tl_unet encoder')
    ap.add_argument('--crop', type=int, default=240)
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lists_dir = Path(args.lists)
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    vis_dir = outdir / 'vis'; vis_dir.mkdir(exist_ok=True)
    logdir = outdir / args.logdir; logdir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(logdir)

    # dataset
    try:
        pad = PadIfNeeded(args.crop, args.crop, border_mode=0, value=0, mask_value=0)
    except TypeError:
        pad = PadIfNeeded(args.crop, args.crop, border_mode=0, pad_val=0, pad_val_mask=0)
    jt = Compose([pad,
                  RandomCrop(args.crop,args.crop),
                  HorizontalFlip(p=0.5),
                  RandomRotate90(p=0.5)])
    train_ds = BrainTumorMRIData(read_list(lists_dir/'train.txt'), return_onehot=False, joint_transform=jt)
    val_ds   = BrainTumorMRIData(read_list(lists_dir/'val.txt'),   return_onehot=False, joint_transform=None)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=args.num_workers, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    C = 4  # classes (0 bg + 1/2/3)
    G = build_generator(args.gen, in_ch=1, out_ch=C, freeze_ratio=args.freeze_ratio).to(device)
    D_raw = PatchGAN70(in_ch=1, out_ch=C).to(device)
    D = PatchGANWithFeatures(D_raw).to(device)

    optG = torch.optim.Adam(G.parameters(), lr=args.lrG, betas=(0.5, 0.999))
    optD = torch.optim.Adam(D_raw.parameters(), lr=args.lrD, betas=(0.5, 0.999))
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    fm_loss = FeatureMatchingLoss() if args.lambda_fm > 0 else None
    l1 = nn.L1Loss()

    def aux_crit(y_pred_logits, y_lbl, x_gray):
        # unified interface for aux loss: operate on predictions vs GT
        if args.aux_loss == 'l1':
            # use softmax probs vs one-hot GT
            probs = torch.softmax(y_pred_logits, 1)
            y_oh = onehot_from_labels(y_lbl, C).to(probs.dtype)
            return l1(probs, y_oh)
        elif args.aux_loss == 'dice':
            probs = torch.softmax(y_pred_logits, 1)
            y_oh = onehot_from_labels(y_lbl, C).to(probs.dtype)
            inter = (probs * y_oh)[:,1:].sum(dim=(0,2,3)) * 2
            denom = (probs*probs)[:,1:].sum(dim=(0,2,3)) + y_oh[:,1:].sum(dim=(0,2,3)) + 1e-6
            dice = (inter/denom).mean()
            return 1 - dice
        else:  # focal (multi-class)
            gamma=2.0; alpha=0.25
            logp = F.log_softmax(y_pred_logits, 1)
            y_oh = onehot_from_labels(y_lbl, C)
            p = torch.exp(logp)
            focal = -alpha*((1-p)**gamma)*logp*y_oh
            return focal[:,1:].mean()  # ignore bg

    best_iou = -math.inf; bad_epochs = 0; patience = 10
    iteration = 0
    for epoch in range(1, args.epochs+1):
        G.train(); D_raw.train()
        running = {'g':0.0,'d':0.0,'aux':0.0,'gan':0.0,'fm':0.0}
        for batch in train_dl:
            iteration += 1
            x = batch['image'].to(device)        # (N,1,H,W)
            y = batch['mask'].to(device)         # (N,H,W)
            y_oh = onehot_from_labels(y, C).to(x.dtype)  # (N,C,H,W)

            # --------- Update D ---------
            optD.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=args.amp):
                # fake
                logits_fake = G(x)                      # (N,C,H,W)
                probs_fake  = torch.softmax(logits_fake, 1)
                D_fake, _   = D(x, probs_fake.detach())
                d_loss_fake = bce_with_logits(D_fake, torch.zeros_like(D_fake))
                # real
                D_real, _   = D(x, y_oh)
                d_loss_real = bce_with_logits(D_real, torch.ones_like(D_real))
                d_loss = 0.5*(d_loss_fake + d_loss_real)
            scaler.scale(d_loss).backward()
            scaler.step(optD)

            # --------- Update G ---------
            optG.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=args.amp):
                logits_fake = G(x)
                probs_fake  = torch.softmax(logits_fake, 1)
                D_fake, feats_fake = D(x, probs_fake)
                gan_loss = bce_with_logits(D_fake, torch.ones_like(D_fake))
                aux = aux_crit(logits_fake, y, x)
                total = gan_loss + args.lambda_aux * aux
                if fm_loss is not None:
                    with torch.no_grad():
                        D_real, feats_real = D(x, y_oh)
                    fm = fm_loss(feats_real, feats_fake)
                    total = total + args.lambda_fm * fm
                else:
                    fm = torch.tensor(0.0, device=x.device)
            scaler.scale(total).backward()
            scaler.step(optG)
            scaler.update()

            running['d']   += d_loss.item()*x.size(0)
            running['g']   += total.item()*x.size(0)
            running['aux'] += aux.item()*x.size(0)
            running['gan'] += gan_loss.item()*x.size(0)
            running['fm']  += fm.item()*x.size(0)
            print(f"iteration:{iteration}, train_loss_encoder:{running['d']/iteration:.4f}, train_loss_decoder:{running['g']/iteration:.4f}")

        ntr = len(train_dl.dataset)
        for k in running: running[k] /= ntr

        # --------- Validation ---------
        G.eval()
        tot_iou_fg, tot_dice_fg, n = 0.0, 0.0, 0
        with torch.no_grad():
            for i, batch in enumerate(val_dl):
                x = batch['image'].to(device)
                y = batch['mask'].to(device)
                logits = G(x)
                pred = torch.argmax(logits, 1)
                _, iou_fg = iou_per_class(pred.cpu(), y.cpu(), C)
                _, dic_fg = dice_per_class(pred.cpu(), y.cpu(), C)
                tot_iou_fg += iou_fg; tot_dice_fg += dic_fg; n += 1
                if i==0:  # save first batch visuals
                    save_vis(x.cpu(), y.cpu(), pred.cpu(), vis_dir, epoch, max_n=8,
                             tag=args.gen.upper())
        iou_fg = tot_iou_fg/max(1,n)
        dice_fg = tot_dice_fg/max(1,n)

        # logs
        writer.add_scalar('train/G_total', running['g'], epoch)
        writer.add_scalar('train/D', running['d'], epoch)
        writer.add_scalar('train/GAN', running['gan'], epoch)
        writer.add_scalar('train/AUX', running['aux'], epoch)
        writer.add_scalar('train/FM', running['fm'], epoch)
        writer.add_scalar('val/IoU_fg', iou_fg, epoch)
        writer.add_scalar('val/Dice_fg', dice_fg, epoch)

        print(f"[Epoch {epoch:03d}] G={running['g']:.4f} D={running['d']:.4f} "
              f"GAN={running['gan']:.4f} AUX={running['aux']:.4f} FM={running['fm']:.4f} "
              f"IoU_fg={iou_fg:.4f} Dice_fg={dice_fg:.4f}")

        # early stop on IoU (primary)
        if iou_fg > best_iou + 1e-6:
            best_iou = iou_fg; bad_epochs = 0
            torch.save({'G': G.state_dict(), 'best_iou': best_iou, 'epoch': epoch},
                       outdir / f'best_{args.gen}.pt')
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"Early stopping. Best IoU_fg={best_iou:.4f}")
                break

    writer.close()
    print(f"Done. Best IoU_fg={best_iou:.4f}. Models & visuals at: {outdir}")

if __name__ == "__main__":
    main()
