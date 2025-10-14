#!/usr/bin/env python3
"""
dataset_brats.py
----------------
PyTorch Dataset for BraTS-style H5 slices (T1ce input).
"""
from pathlib import Path
from typing import Any, Callable, Optional, Sequence, Dict
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset

def robust_norm(x: np.ndarray, lo_p=0.5, hi_p=99.5) -> np.ndarray:
    lo, hi = np.percentile(x, [lo_p, hi_p])
    x = np.clip(x, lo, hi)
    denom = (hi - lo) if hi > lo else 1.0
    return (x - lo) / denom

def to_onehot3_from_labels(lbl: np.ndarray) -> np.ndarray:
    oh = np.zeros(lbl.shape + (3,), dtype=np.float32)
    oh[..., 0] = (lbl == 1)
    oh[..., 1] = (lbl == 2)
    oh[..., 2] = (lbl == 3) | (lbl == 4)
    return oh

def to_label_from_onehot3(oh: np.ndarray, thr: float = 0.5) -> np.ndarray:
    fg = (oh.max(axis=-1) > thr)
    cls = np.argmax(oh, axis=-1) + 1
    cls[~fg] = 0
    return cls.astype(np.int64)

class BrainTumorMRIData(Dataset):
    def __init__(
        self,
        files: Sequence[Path | str],
        t1ce_idx: int = 1,
        return_onehot: bool = False,
        intensity_norm: bool = True,
        joint_transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        img_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        mask_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        image_dtype: torch.dtype = torch.float32,
        mask_dtype_onehot: torch.dtype = torch.float32,
    ):
        self.files = [Path(f) for f in files]
        self.t1ce_idx = t1ce_idx
        self.return_onehot = return_onehot
        self.intensity_norm = intensity_norm
        self.joint_transform = joint_transform
        self.img_transform = img_transform
        self.mask_transform = mask_transform
        self.image_dtype = image_dtype
        self.mask_dtype_onehot = mask_dtype_onehot

    def __len__(self) -> int:
        return len(self.files)

    def _load_h5(self, p: Path):
        with h5py.File(p, "r") as f:
            image = f["image"][:]
            mask  = f["mask"][:]
        return image, mask

    def _prepare_xy(self, image, mask):
        x = image[..., self.t1ce_idx]
        if self.intensity_norm:
            x = robust_norm(x)

        if mask.ndim == 2:
            mask_oh = to_onehot3_from_labels(mask)
        else:
            mask_oh = mask.astype(np.float32)

        if self.return_onehot:
            y = mask_oh.transpose(2,0,1)
        else:
            y = to_label_from_onehot3(mask_oh)
        return x, y

    def __getitem__(self, index: int):
        path = self.files[index]
        image, mask = self._load_h5(path)
        x, y = self._prepare_xy(image, mask)

        if self.joint_transform is not None:
            sample = {"image": x, "mask": y}
            out = self.joint_transform(**sample) if hasattr(self.joint_transform, "__call__") else self.joint_transform(sample)
            x, y = out["image"], out["mask"]

        x_t = torch.from_numpy(x).unsqueeze(0).to(self.image_dtype)
        if self.return_onehot:
            y_t = torch.from_numpy(y).to(self.mask_dtype_onehot)
        else:
            y_t = torch.from_numpy(y).long()

        if self.img_transform is not None:
            x_t = self.img_transform(x_t)
        if self.mask_transform is not None:
            y_t = self.mask_transform(y_t)

        return {"image": x_t, "mask": y_t, "path": str(path)}

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}(n={len(self)}, t1ce_idx={self.t1ce_idx}, "
                f"return_onehot={self.return_onehot}, intensity_norm={self.intensity_norm})")

