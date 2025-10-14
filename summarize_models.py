#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
summarize_models.py
-------------------
Merge multiple `metrics_volumes.csv` (from eval_and_report.py) and produce:
1) A LaTeX table of mean±SD per metric for each method (best highlighted).
2) Pairwise stats (A vs B) for each metric: n, mean diff, 95% bootstrap CI,
   paired t-test p-value, Wilcoxon signed-rank p-value.
3) CSV exports for downstream plotting.

Usage
-----
python summarize_models.py \
  --csv unet=reports/unet_val/metrics_volumes.csv \
  --csv baseline=reports/baseline_val/metrics_volumes.csv \
  --csv tl_unet=reports/tl_unet_val/metrics_volumes.csv \
  --outdir reports/summary --n-boot 10000 --seed 42

Notes
-----
- We align on the intersection of volume_id across all methods to keep pairing fair.
- Metrics expected in each CSV: iou_fg, dice_fg, hd95_fg, assd_fg, bf_fg.
- Higher-is-better: IoU/Dice/BF；Lower-is-better: HD95/ASSD。
- In pairwise outputs, we convert differences so that **positive means the first
  method is better** for *all* metrics (i.e., we negate diffs for lower-better metrics).
"""
from __future__ import annotations
from pathlib import Path
import argparse
import itertools
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from scipy import stats

HIGHER_BETTER = {"iou_fg": True, "dice_fg": True, "bf_fg": True,
                 "hd95_fg": False, "assd_fg": False}
METRIC_ORDER = ["iou_fg", "dice_fg", "hd95_fg", "assd_fg", "bf_fg"]
METRIC_LABEL = {
    "iou_fg":  r"IoU$_{fg}$",
    "dice_fg": r"Dice$_{fg}$",
    "hd95_fg": r"HD95$_{fg}$",
    "assd_fg": r"ASSD$_{fg}$",
    "bf_fg":   r"BF$_{fg}$",
}

def parse_method_arg(arg: str) -> Tuple[str, Path]:
    if "=" not in arg:
        raise ValueError("Use --csv name=path/to/metrics_volumes.csv")
    name, path = arg.split("=", 1)
    return name.strip(), Path(path.strip()).resolve()

def read_metrics(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "volume_id" not in df.columns:
        raise ValueError(f"{path} must contain 'volume_id' column.")
    keep = ["volume_id"] + [m for m in METRIC_ORDER if m in df.columns]
    return df[keep].copy()

def format_mean_sd(x: np.ndarray) -> str:
    if x.size == 0: return "n/a"
    return f"{np.nanmean(x):.3f} $\\pm$ {np.nanstd(x):.3f}"

def bootstrap_ci_mean_diff(diffs: np.ndarray, n_boot: int = 10000,
                           seed: int = 42, alpha: float = 0.05) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    N = len(diffs)
    if N == 0:
        return (np.nan, np.nan)
    boots = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, N, size=N)
        boots[b] = np.nanmean(diffs[idx])
    lo = np.nanpercentile(boots, 100*alpha/2)
    hi = np.nanpercentile(boots, 100*(1 - alpha/2))
    return float(lo), float(hi)

def latex_bold_best(row_vals: Dict[str, float], higher_better: bool) -> Dict[str, str]:
    """
    Given raw means (not mean±sd strings), return strings with \textbf{} for the best.
    """
    out = {}
    if not row_vals:
        return out
    # choose best ignoring NaNs
    items = [(k, v) for k, v in row_vals.items() if not np.isnan(v)]
    if not items:
        return {k: "n/a" for k in row_vals}
    best_val = (max if higher_better else min)(v for _, v in items)
    for k, v in row_vals.items():
        if np.isnan(v):
            out[k] = "n/a"
        else:
            s = f"{v:.3f}"
            if abs(v - best_val) < 1e-12:
                s = r"\textbf{" + s + "}"
            out[k] = s
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", action="append", required=True,
                    help="name=path/to/metrics_volumes.csv (repeatable)")
    ap.add_argument("--outdir", type=str, required=True)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    # 1) Read and align
    method_paths = dict(parse_method_arg(s) for s in args.csv)
    tables: Dict[str, pd.DataFrame] = {name: read_metrics(p) for name, p in method_paths.items()}
    # intersection of volume_id
    common = None
    for df in tables.values():
        vids = set(df["volume_id"].tolist())
        common = vids if common is None else (common & vids)
    common = sorted(common) if common else []
    if not common:
        raise SystemExit("No overlapping volume_id across methods; cannot do paired tests.")
    for k, df in tables.items():
        tables[k] = df[df["volume_id"].isin(common)].sort_values("volume_id").reset_index(drop=True)

    # 2) Descriptive stats (mean±sd) per method
    desc_rows = []
    raw_means_for_bold: Dict[str, Dict[str, float]] = {m: {} for m in METRIC_ORDER}
    for name, df in tables.items():
        row = {"method": name, "n": len(df)}
        for m in METRIC_ORDER:
            if m in df.columns:
                vals = df[m].to_numpy(dtype=np.float64)
                row[m] = format_mean_sd(vals)
                raw_means_for_bold[m][name] = float(np.nanmean(vals))
            else:
                row[m] = "n/a"
                raw_means_for_bold[m][name] = np.nan
        desc_rows.append(row)
    desc_df = pd.DataFrame(desc_rows)
    desc_df.to_csv(outdir / "summary_descriptives.csv", index=False)

    # 3) LaTeX summary table (mean±sd) with best bold (by mean only)
    lines = []
    cols = "l" + "c"*len(METRIC_ORDER)
    lines.append(f"\\begin{{tabular}}{{{cols}}}\\hline\\hline")
    header = "Method & " + " & ".join(METRIC_LABEL[m] for m in METRIC_ORDER) + r"\\\hline"
    lines.append(header)
    for _, r in desc_df.iterrows():
        name = r["method"]
        cells = []
        for m in METRIC_ORDER:
            # bold best by mean number only for readability
            mean_map = raw_means_for_bold[m]
            hb = HIGHER_BETTER[m]
            bold_map = latex_bold_best(mean_map, higher_better=hb)
            # embed bold into the mean±sd string by replacing the mean prefix
            cur = str(r[m])
            if name in bold_map and bold_map[name] != "n/a" and cur != "n/a":
                # replace the leading mean with bolded mean
                # cur like "0.842 ± 0.051"
                parts = cur.split(" ")
                parts[0] = bold_map[name]
                cur = " ".join(parts)
            cells.append(cur)
        lines.append(name + " & " + " & ".join(cells) + r"\\")
    lines.append(r"\hline\hline")
    lines.append(r"\end{tabular}")
    (outdir / "summary_table.tex").write_text("\n".join(lines), encoding="utf-8")

    # 4) Pairwise tests per metric
    pair_rows_all: Dict[str, List[Dict]] = {m: [] for m in METRIC_ORDER}
    names = list(tables.keys())
    for a, b in itertools.combinations(names, 2):
        dfa, dfb = tables[a], tables[b]
        assert (dfa["volume_id"].to_list() == dfb["volume_id"].to_list()), "volume_id mismatch after alignment"
        for m in METRIC_ORDER:
            if (m not in dfa.columns) or (m not in dfb.columns):
                continue
            xa = dfa[m].to_numpy(np.float64)
            xb = dfb[m].to_numpy(np.float64)
            # paired diffs in "performance" units: positive => A better
            if HIGHER_BETTER[m]:
                diffs = xa - xb
            else:
                diffs = -(xa - xb)  # invert: smaller is better -> negate so that positive still "A better"

            n = diffs.size
            mean_diff = float(np.nanmean(diffs))
            ci_lo, ci_hi = bootstrap_ci_mean_diff(diffs, n_boot=args.n_boot, seed=args.seed)

            # paired t-test / wilcoxon (handle fallback if all diffs==0)
            try:
                t_p = float(stats.ttest_rel(xa, xb, nan_policy="omit").pvalue)
            except Exception:
                t_p = float("nan")
            try:
                # Wilcoxon requires non-zero diffs; if zero, p=1
                if np.allclose(diffs, 0, atol=1e-12):
                    w_p = 1.0
                else:
                    # scipy wilcoxon returns statistic, pvalue
                    w_p = float(stats.wilcoxon(xa, xb, zero_method="wilcox", alternative="two-sided").pvalue)
            except Exception:
                w_p = float("nan")

            pair_rows_all[m].append({
                "A": a, "B": b, "n": n,
                "mean_diff(>0:A better)": mean_diff,
                "boot_CI95_lo": ci_lo, "boot_CI95_hi": ci_hi,
                "ttest_p": t_p, "wilcoxon_p": w_p,
            })

    # write per-metric CSV + LaTeX
    for m in METRIC_ORDER:
        rows = pair_rows_all[m]
        if not rows: 
            continue
        dfm = pd.DataFrame(rows)
        dfm.to_csv(outdir / f"pairwise_{m}.csv", index=False)

        # LaTeX small table
        lines = []
        lines.append("\\begin{tabular}{lcccc}\n\\hline\\hline")
        lines.append(f"Pair & $\\Delta$ {METRIC_LABEL[m]} & 95\\% CI & t p & Wilcoxon p\\\\\\hline")
        for r in rows:
            pair = f"{r['A']} vs {r['B']}"
            delta = f"{r['mean_diff(>0:A better)']:.3f}"
            ci = f"[{r['boot_CI95_lo']:.3f}, {r['boot_CI95_hi']:.3f}]"
            tp = f"{r['ttest_p']:.3e}" if not np.isnan(r['ttest_p']) else "n/a"
            wp = f"{r['wilcoxon_p']:.3e}" if not np.isnan(r['wilcoxon_p']) else "n/a"
            lines.append(f"{pair} & {delta} & {ci} & {tp} & {wp}\\\\")
        lines.append("\\hline\\hline\n\\end{tabular}")
        (outdir / f"pairwise_{m}.tex").write_text("\n".join(lines), encoding="utf-8")

    print(f"Done. Outputs in: {outdir}")
    print(" - summary_descriptives.csv")
    print(" - summary_table.tex (mean±SD with bold best)")
    print(" - pairwise_{metric}.csv / .tex for each metric")

if __name__ == "__main__":
    main()
