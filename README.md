# Outdoor Scene Image Classification

## Project Overview
This project aims to build a deep learning model to classify outdoor scene images into six categories: **buildings, forest, glacier, mountain, sea, and street**. The goal is to complete an end-to-end (E2E) deep learning pipeline, including data exploration, model training, and evaluation.

## Project Structure
```
data/
 ├─ raw/                         # Original Kaggle dataset
 └─ processed/                   # Split into train/val/test
notebooks/
 ├─ dataset_info.ipynb
 ├─ day5_dataloader_debug.ipynb  # Data pipeline verification
src/
 ├─ data_preprocessing.py
 ├─ model.py                     # CNN Baseline architecture
 └─ train.py                     # Training & Validation pipeline
README.md
```

## Dataset
The dataset is based on the Intel Image Classification from Kaggle.
- Training data: `data/raw/seg_train`
- Test data: `data/raw/seg_test`
- Prediction data: `data/raw/seg_pred`
- Link: `https://www.kaggle.com/datasets/puneet6060/intel-image-classification/data`

## Exploratory Data Analysis (EDA)
An exploratory data analysis was performed to understand class distribution and data quality.
- Class Distribution: Six classes with ~2,200–2,500 images per class.
- Balance: The imbalance ratio (largest/smallest) is 1.14, indicating a well-balanced dataset.
- Data Quality: Visual inspection confirmed no obvious mislabeling or corrupted files.

## Data Preprocessing
Automated script to split raw training data into `train/val/test` sets.
- Resolution: $224 \times 224$ pixels.
- Split Ratio: 70% Train / 15% Val / 15% Test.

Command:
```bash
python src/data_preprocessing.py --raw_dir data/raw/seg_train --out_dir data/processed --val_ratio 0.15 --test_ratio 0.15 --img_size 224 --batch_size 32 --copy_mode copy
```

## Model Architecture: CNN Baseline
Implemented a modular CNN baseline in `src/model.py`.
- 4 Convolutional Blocks: Uses `Conv2d -> BatchNorm -> ReLU -> MaxPool`.
- Engineering Choice: Set `bias=False` in Conv layers since they are followed by BatchNorm, reducing redundant parameters.
- Global Pooling: Used `AdaptiveAvgPool2d((1, 1))` for input size flexibility.
- Regularization: Integrated `Dropout(p=0.3)` in the classifier to prevent overfitting.

## Training & Validation Results
Initial training was performed for 10 epochs to establish a performance benchmark.

| Metric | Epoch 1 | Epoch 5 | Epoch 10 |
| :--- | :---: | :---: | :---: |
| **Train Loss** | 0.9506 | 0.5845 | 0.4762 |
| **Train Acc** | 62.77% | 78.49% | **83.17%** |
| **Val Acc** | 63.68% | 81.67% | **81.96%** |

### Observation: 
The model shows healthy convergence. The small gap between Train and Val accuracy indicates strong generalization.