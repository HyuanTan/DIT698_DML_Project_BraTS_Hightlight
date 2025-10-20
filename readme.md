# Install
Base on `dml`:
```
pip install -U segmentation-models-pytorch torchmetrics --quiet

pip install h5py
```

# Data preprocess
We unzip the `archive.zip` and rename the folder to `brats2020-training-data`. Put `brats2020-training-data` and `BraTS2D_segmentation**.ipynb` in the same folder.

# Train&Evaluate
Details can be seen in  `BraTS2D_segmentation**.ipynb`

`BraTS2D_segmentation_googlecloud.ipynb` use the whole dataset [Brain Tumor Segmentation(BraTS2020)](https://www.kaggle.com/datasets/awsaf49/brats2020-training-data/data). We train and test the model using **GoogleCloud** Compute Engine, `GPU L4`.


`BraTS2D_segmentation_localGPU.ipynb` use half of the [Brain Tumor Segmentation(BraTS2020)](https://www.kaggle.com/datasets/awsaf49/brats2020-training-data/data)  dataset as a subset and train on `RTX4060Ti`.