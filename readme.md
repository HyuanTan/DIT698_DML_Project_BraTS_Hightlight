# Install
Base on `dml`:
```
pip install h5py albumentations>=1.4.0 opencv-python-headless
```

# Data preprocess
We unzip the `archive.zip` and rename the folder to `brats2020-training-data`
## Visualize
Details can be seen in  `project.ipynb`


## Preprocess
```bash
python prepare_splits.py ./brats2020-training-data/BraTS2020_training_data/content/data/ --out ./splits_10 --seed 45 --ratios 0.8 0.1 0.1 --n-select 10
```

# Train

```bash
# Baseline
python train_pix2pix_seg.py --lists ./splits_10 --gen baseline --outdir outputs/baseline --batch-size 2 --num-workers 2 --epochs 1

# UNet 生成器（推荐）
python train_pix2pix_seg.py --lists ./splits_10 --gen unet --outdir outputs/unet --batch-size 2 --num-workers 2 --epochs 1

# 预训练Encoder + 微调
python train_pix2pix_seg.py --lists ./splits_10 --gen tl_unet --freeze-ratio 0.75 --lambda-fm 10.0 --outdir outputs/tl_unet --batch-size 2 --num-workers 2 --epochs 1
```


# Evaluate

```
python eval_and_report.py --lists ./splits_10 --split val --gen unet --checkpoint outputs/unet/best_unet.pt --outdir outputs/unet/unet_val

python eval_and_report.py --lists ./splits_10 --split val --gen baseline --checkpoint outputs/baseline/best_baseline.pt --outdir outputs/baseline/baseline_val

python eval_and_report.py --lists ./splits_10 --split val --gen tl_unet --checkpoint outputs/tl_unet/best_tl_unet.pt --outdir outputs/tl_unet/baseline_val

```

## summarize
```
python summarize_models.py --csv baseline=outputs/baseline/baseline_val/metrics_volumes.csv --csv unet=outputs/unet/unet_val/metrics_volumes.csv --csv tl_unet=outputs/tl_unet/tl_unet_val/metrics_volumes.csv --outdir outputs/summary_val --n-boot 10000 --seed 42
```
