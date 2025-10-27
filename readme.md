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

Update Note:

*2025-10-19*\
`BraTS2D_segmentation_googlecloud.ipynb` use the whole dataset [Brain Tumor Segmentation(BraTS2020)](https://www.kaggle.com/datasets/awsaf49/brats2020-training-data/data). We train and test the model using **GoogleCloud** Compute Engine, `GPU L4`.(**Training Finished**)


`BraTS2D_segmentation_localGPU.ipynb` use half of the [Brain Tumor Segmentation(BraTS2020)](https://www.kaggle.com/datasets/awsaf49/brats2020-training-data/data)  dataset as a subset and train on `RTX4060Ti`.(**Training Finished**)

*2025-10-26*\
Train on **GoogleCloud** Compute Engine, `GPU L4` using whole dataset.(**Training Not Finished**)
**Update BraTS2D_segmentation.ipynb**：
- Using same cost function instead of copy many times
- Add Learning Rate Scheduler and Scaler to all model
- Add early stop

*2025-10-27*\
`BraTS2D_segmentation_googlecloud_resume.ipynb` is base on `BraTS2D_segmentation_googlecloud.ipynb`, using the whole dataset [Brain Tumor Segmentation(BraTS2020)](https://www.kaggle.com/datasets/awsaf49/brats2020-training-data/data) and train/test the model using **GoogleCloud** Compute Engine, `GPU L4`.

`BraTS2D_segmentation_googlecloud_resume.ipynb` resumed Transfer-learning UNet encoder from 16 to 20 epochs and optimize visual layout. The **final version for report**. (**Training Finished**)