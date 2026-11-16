# Fetal Ultrasound Segmentation

Research internship project on fetal ultrasound image segmentation, feature extraction, and machine-learning-based analysis. The notebooks cover the full pipeline: organising the raw dataset, exploring it, and training two U-Net-based segmentation models (TensorFlow and PyTorch).

All notebooks were developed in Google Colab. Each file has an **Open in Colab** badge at the top.

## Notebooks

Run them in this order:

| # | Notebook | What it does | Main libraries |
|---|----------|--------------|----------------|
| 1 | [`DataPreprocessing.ipynb`](DataPreprocessing.ipynb) | Mounts Google Drive, unzips the nested dataset archives (Trans-thalamic, Trans-cerebellum, Trans-ventricular, etc.), separates images from annotation formats (COCO, YOLO, PASCAL-VOC, …), cleans filenames, and pairs each image with its segmentation mask. | `os`, `zipfile`, `shutil`, `re`, `PIL`, `matplotlib` |
| 2 | [`DataAnalysis.ipynb`](DataAnalysis.ipynb) | Exploratory data analysis: class distribution (benign / malignant / normal), sample visualisation, image size and brightness statistics. | `pandas`, `numpy`, `opencv`, `seaborn`, `matplotlib` |
| 3 | [`segmentation.ipynb`](segmentation.ipynb) | **Attention U-Net** built from scratch in TensorFlow/Keras (256×256 input, binary mask) with deep supervision and a combined loss. Evaluated per image with Dice, IoU, BCE loss, Hausdorff distance and boundary F-score. | `tensorflow`, `scikit-learn`, `scipy`, `pandas` |
| 4 | [`Unet.ipynb`](Unet.ipynb) | **U-Net with a pre-trained ResNeXt50-32x4d encoder** (ImageNet weights) via `segmentation-models-pytorch`. Albumentations augmentation, Dice + BCE combined loss, AdamW (lr 1e-4) with ReduceLROnPlateau, 20 epochs. Reports pixel accuracy, AUC-ROC and threshold analysis. | `torch`, `segmentation-models-pytorch`, `albumentations`, `opencv` |

## Results so far

| Model | Metric | Value |
|-------|--------|-------|
| Attention U-Net (TensorFlow) | Mean Dice | 0.39 |
| | Mean IoU | 0.24 |
| U-Net + ResNeXt50 (PyTorch) | Pixel accuracy | 98.5% |
| | AUC-ROC (10 validation images) | 0.78 |
| | Validation Dice | ≈0.25 |

High pixel accuracy combined with modest Dice/IoU is expected here: the fetal structure covers a small part of each image, so most pixels are background. Dice and IoU are the more meaningful measures, and improving them is the next step.

## Getting started

1. Open a notebook with its **Open in Colab** badge.
2. Put the dataset in your Google Drive (the notebooks expect it under `MyDrive/Ultrasound_Dataset/` or as an uploaded `Datasets.rar`; adjust the paths in the first cells if yours differ).
3. Use a GPU runtime (**Runtime → Change runtime type → GPU**) for notebooks 3 and 4.
4. Run the cells from top to bottom.

The dataset itself is not included in this repository.

## Repository structure

```
├── DataPreprocessing.ipynb   # 1. dataset extraction, cleaning, image–mask pairing
├── DataAnalysis.ipynb        # 2. exploratory data analysis
├── segmentation.ipynb        # 3. Attention U-Net (TensorFlow)
├── Unet.ipynb                # 4. U-Net + ResNeXt50 encoder (PyTorch)
└── README.md
```

## Author

Yashvi Tyagi
