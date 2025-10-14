#!/usr/bin/env python3
"""
prepare_splits.py
-----------------
Group BraTS-style H5 slice files by VOLUME and split them into train/val/test.
Writes file lists (absolute paths) and metadata for reproducibility.

Filename pattern assumed:
    volume_<id>_slice_<k>.h5
"""
import argparse
import re
import random
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple, Optional, Sequence

H5_RE = re.compile(r'^volume_(\d+)_slice_(\d+)\.h5$')


def group_by_volume(root: Path):
    print(f'Grouping .h5 files in {root} by volume id...')
    groups = defaultdict(list)
    for p in root.glob('*.h5'):
        # print(f'Found file: {p.name}') # volume_135_slice_93.h5
        m = H5_RE.fullmatch(p.name)
        if not m:
            continue
        vid = int(m.group(1))
        groups[vid].append(p.resolve())
    return {k: sorted(v) for k, v in sorted(groups.items())}

def _largest_remainder_counts(n: int, shares: Sequence[float]) -> List[int]:
    raw = [n * s for s in shares]
    base = [int(x) for x in raw]
    rem = n - sum(base)
    order = sorted(range(len(shares)), key=lambda i: raw[i] - base[i], reverse=True)
    for k in range(rem):
        base[order[k]] += 1
    return base

def split_volumes_flexible(volume_ids: List[int], *, seed: int = 45,
                           ratios: Tuple[float, float, float] = (0.70, 0.15, 0.15),
                           n_select: Optional[int] = None, sample_frac: Optional[float] = None,
                           shuffle_before_split: bool = True):
    ids = list(volume_ids)
    rng = random.Random(seed)
    r_tr, r_val, r_te = ratios
    s = r_tr + r_val + r_te
    r_tr, r_val, r_te = r_tr/s, r_val/s, r_te/s

    N = len(ids)
    if n_select is not None:
        if not (1 <= n_select <= N):
            raise ValueError(f"n_select must be in [1, {N}], got {n_select}")
        selected = sorted(rng.sample(ids, n_select))
    elif sample_frac is not None:
        if not (0 < sample_frac <= 1.0):
            raise ValueError("sample_frac must be in (0,1].")
        k = max(1, int(round(N * sample_frac)))
        selected = sorted(rng.sample(ids, k))
    else:
        selected = sorted(ids)

    alloc = selected[:]
    if shuffle_before_split:
        rng.shuffle(alloc)

    n_tr, n_val, n_te = _largest_remainder_counts(len(alloc), [r_tr, r_val, r_te])
    train_ids = sorted(alloc[:n_tr])
    val_ids   = sorted(alloc[n_tr:n_tr+n_val])
    test_ids  = sorted(alloc[n_tr+n_val:])
    assert len(train_ids)+len(val_ids)+len(test_ids) == len(selected)
    return train_ids, val_ids, test_ids, selected

def write_list(paths: List[Path], out_txt: Path):
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    with out_txt.open('w', encoding='utf-8') as f:
        for p in paths:
            f.write(str(p) + '\n')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('data_dir', type=str, help='Directory containing *.h5 files')
    ap.add_argument('--out', type=str, required=True, help='Output dir for filelists and metadata')
    ap.add_argument('--seed', type=int, default=45)
    ap.add_argument('--ratios', type=float, nargs=3, default=(0.7, 0.15, 0.15), help='train val test ratios')
    ap.add_argument('--n-select', type=int, default=None, help='Select exactly N volumes before splitting')
    ap.add_argument('--sample-frac', type=float, default=None, help='Select a fraction of volumes before splitting')
    args = ap.parse_args()

    data_dir = Path(args.data_dir).expanduser().resolve()
    print(f'Reading data from: {data_dir}')
    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    groups = group_by_volume(data_dir)
    if not groups:
        raise SystemExit(f'No .h5 files found in {data_dir} matching pattern volume_<id>_slice_<k>.h5')
    volume_ids = list(groups.keys())
    print(f'Found {len(volume_ids)} volumes.')

    train_ids, val_ids, test_ids, selected_ids = split_volumes_flexible(
        volume_ids, seed=args.seed, ratios=tuple(args.ratios),
        n_select=args.n_select, sample_frac=args.sample_frac
    )
    print(f'Using {len(selected_ids)} volumes -> train={len(train_ids)} val={len(val_ids)} test={len(test_ids)}')

    def gather(ids):
        files = []
        for vid in ids:
            files.extend(groups[vid])
        return sorted(files)

    train_files = gather(train_ids)
    val_files   = gather(val_ids)
    test_files  = gather(test_ids)

    write_list(train_files, out_dir/'train.txt')
    write_list(val_files,   out_dir/'val.txt')
    write_list(test_files,  out_dir/'test.txt')

    meta = out_dir/'meta'
    meta.mkdir(exist_ok=True)
    write_list([Path(str(v)) for v in selected_ids], meta/'selected_volume_ids.txt')
    write_list([Path(str(v)) for v in train_ids],    meta/'train_volume_ids.txt')
    write_list([Path(str(v)) for v in val_ids],      meta/'val_volume_ids.txt')
    write_list([Path(str(v)) for v in test_ids],     meta/'test_volume_ids.txt')

    print(f'Wrote filelists to: {out_dir}')

if __name__ == '__main__':
    main()

