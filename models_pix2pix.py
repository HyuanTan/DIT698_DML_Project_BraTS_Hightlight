#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
models_pix2pix.py
- Generators:
  A) EncDecGAN: plain encoder-decoder (few/no skips)  -> "baseline"
  B) UNetGAN   : U-Net generator with full skip       -> "unet"
  C) TLUNetGAN : U-Net generator w/ pretrained encoder (partial fine-tune) -> "tl_unet"
- Discriminator:
  PatchGAN70: 70x70 PatchGAN (shared across methods)
"""
from typing import List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34

# ------------------ blocks ------------------
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, norm=True):
        super().__init__()
        layers = [nn.Conv2d(in_ch, out_ch, k, s, p, bias=not norm)]
        if norm: layers += [nn.BatchNorm2d(out_ch)]
        layers += [nn.LeakyReLU(0.2, inplace=True)]
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

class DeconvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=4, s=2, p=1, norm=True):
        super().__init__()
        layers = [nn.ConvTranspose2d(in_ch, out_ch, k, s, p, bias=not norm)]
        if norm: layers += [nn.BatchNorm2d(out_ch)]
        layers += [nn.ReLU(inplace=True)]
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

# ------------------ A) EncDecGAN ------------------
class EncDecGAN(nn.Module):
    """
    Simple encoder-decoder; optional tiny #skips (default none) to act as the
    'true baseline' per your spec.
    """
    def __init__(self, in_ch=1, out_ch=4, base=64, use_skip=False):
        super().__init__()
        self.use_skip = use_skip
        # encoder
        self.e1 = ConvBlock(in_ch,   base,   s=2)
        self.e2 = ConvBlock(base,    base*2, s=2)
        self.e3 = ConvBlock(base*2,  base*4, s=2)
        self.e4 = ConvBlock(base*4,  base*8, s=2)
        # decoder
        self.d3 = DeconvBlock(base*8, base*4)
        self.d2 = DeconvBlock(base*4, base*2)
        self.d1 = DeconvBlock(base*2, base)
        # self.out = nn.Conv2d(base, out_ch, 1)
        self.out = nn.ConvTranspose2d(base, out_ch, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        x1 = self.e1(x); x2 = self.e2(x1); x3 = self.e3(x2); x4 = self.e4(x3)
        y  = self.d3(x4)
        if self.use_skip: y = y + x3
        y  = self.d2(y)
        if self.use_skip: y = y + x2
        y  = self.d1(y)
        if self.use_skip: y = y + x1
        return self.out(y)  # logits (N,C,H,W)

# ------------------ B) UNetGAN ------------------
class UNetBlockDown(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )
    def forward(self, x): return self.net(x)

class UNetBlockUp(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=False):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if dropout: layers += [nn.Dropout(0.5)]
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

class UNetGAN(nn.Module):
    """pix2pix-style U-Net generator with full skip connections."""
    def __init__(self, in_ch=1, out_ch=4, base=64):
        super().__init__()
        # down
        self.d1 = nn.Sequential(nn.Conv2d(in_ch, base, 4, 2, 1), nn.LeakyReLU(0.2, True))
        self.d2 = UNetBlockDown(base, base*2)
        self.d3 = UNetBlockDown(base*2, base*4)
        self.d4 = UNetBlockDown(base*4, base*8)
        self.d5 = UNetBlockDown(base*8, base*8)
        # up
        self.u4 = UNetBlockUp(base*8, base*8, dropout=True)
        self.u3 = UNetBlockUp(base*8*2, base*4, dropout=True)
        self.u2 = UNetBlockUp(base*4*2, base*2)
        self.u1 = UNetBlockUp(base*2*2, base)
        self.out = nn.ConvTranspose2d(base*2, out_ch, 4, 2, 1)

    def forward(self, x):
        e1 = self.d1(x); e2 = self.d2(e1); e3 = self.d3(e2); e4 = self.d4(e3); e5 = self.d5(e4)
        y  = self.u4(e5); y = torch.cat([y, e4], dim=1)
        y  = self.u3(y);  y = torch.cat([y, e3], dim=1)
        y  = self.u2(y);  y = torch.cat([y, e2], dim=1)
        y  = self.u1(y);  y = torch.cat([y, e1], dim=1)
        return self.out(y)  # logits

# ------------------ C) Transfer-learning UNet encoder ------------------
class TLUNetGAN(nn.Module):
    """
    U-Net generator with a pretrained ResNet-34 encoder (ImageNet).
    Fine-tune part of encoder layers per freeze_ratio.
    """
    def __init__(self, in_ch=1, out_ch=4, base=64, freeze_ratio=0.75):
        super().__init__()
        # map 1->3 for resnet
        self.stem = nn.Conv2d(in_ch, 3, kernel_size=1)
        res = resnet34(weights='IMAGENET1K_V1')
        # encoder features
        self.enc1 = nn.Sequential(res.conv1, res.bn1, res.relu)  # /2
        self.pool = res.maxpool                                   # /4
        self.enc2 = res.layer1                                    # /4
        self.enc3 = res.layer2                                    # /8
        self.enc4 = res.layer3                                    # /16
        self.enc5 = res.layer4                                    # /32
        # decoder (simple)
        self.up4 = DeconvBlock(512, 256)   # -> /16
        self.up3 = DeconvBlock(256+256, 128)  # -> /8
        self.up2 = DeconvBlock(128+128, 64)   # -> /4
        self.up1 = DeconvBlock(64+64, base)   # -> /2
        self.out = nn.ConvTranspose2d(base+64, out_ch, 4, 2, 1)  # -> /1
        # freeze early encoder
        total = sum(1 for _ in res.parameters())
        k = int(total * freeze_ratio)
        for i, p in enumerate(res.parameters()):
            p.requires_grad = (i >= k)

    def forward(self, x):
        x  = self.stem(x)
        e1 = self.enc1(x)         # H/2
        e2 = self.pool(e1)        # H/4
        e2 = self.enc2(e2)
        e3 = self.enc3(e2)        # H/8
        e4 = self.enc4(e3)        # H/16
        e5 = self.enc5(e4)        # H/32
        y  = self.up4(e5)         # /16
        y  = torch.cat([y, e4], 1)
        y  = self.up3(y)          # /8
        y  = torch.cat([y, e3], 1)
        y  = self.up2(y)          # /4
        y  = torch.cat([y, e2], 1)
        y  = self.up1(y)          # /2
        y  = torch.cat([y, e1], 1)
        return self.out(y)        # /1, logits

# ------------------ PatchGAN 70x70 ------------------
class PatchGAN70(nn.Module):
    """
    Discriminator that receives concatenated condition and output:
    x_cond: (N,1,H,W), y: (N,C,H,W)  -> concat along channels.
    """
    def __init__(self, in_ch=1, out_ch=4, base=64, spectral=False):
        super().__init__()
        C = in_ch + out_ch
        Conv = nn.utils.spectral_norm if spectral else (lambda m: m)
        layers = []
        def add(cin, cout, s):
            layers.append(Conv(nn.Conv2d(cin, cout, 4, s, 1, bias=False)))
            layers.append(nn.BatchNorm2d(cout))
            layers.append(nn.LeakyReLU(0.2, True))
        layers.append(Conv(nn.Conv2d(C, base, 4, 2, 1)))  # no BN first
        layers.append(nn.LeakyReLU(0.2, True))
        add(base, base*2, 2)
        add(base*2, base*4, 2)
        add(base*4, base*8, 1)
        layers.append(Conv(nn.Conv2d(base*8, 1, 4, 1, 1)))  # (N,1,H',W') logits
        self.net = nn.Sequential(*layers)

    def forward(self, x_cond, y):
        z = torch.cat([x_cond, y], dim=1)
        return self.net(z)
