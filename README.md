# Outdoor Scene Image Classification

## Project Overview
This project aims to build a deep learning model to classify outdoor scene images
into six categories: buildings, forest, glacier, mountain, sea, and street.
The goal is to complete an end-to-end (E2E) deep learning pipeline, including
data exploration, model training, and evaluation.

## Dataset
The dataset is based on the Intel Image Classification dataset from Kaggle.
Images are organized into class-specific folders and split into training,
testing, and prediction sets.
- Training data: `data/raw/seg_train`
- Test data: `data/raw/seg_test`
- Prediction data: `data/raw/seg_pred`

## Exploratory Data Analysis (EDA)
An exploratory data analysis was performed on the training dataset to
understand the class distribution and data quality.

### Class Distribution
The training dataset contains six classes with approximately 2,200–2,500 images
per class. The ratio between the largest and smallest class is about 1.14,
indicating that the dataset is reasonably balanced.
No class re-weighting or over-sampling is required.

### Sample Images
Random samples from each class were visually inspected.
No obvious mislabeling or corrupted images were found.

## Model (Planned)
The next steps include building a convolutional neural network (CNN) using
PyTorch, applying data augmentation, and evaluating the model on the test set.

## Project Structure
```
data/
 ├─ raw/
 │   ├─ seg_train/
 │   ├─ seg_test/
 │   └─ seg_pred/
 └─ processed/
notebooks/
 ├─ dataset_info.ipynb
README.md
```



Please download Intel Image Classification dataset from Kaggle and extract it into data/raw/
https://www.kaggle.com/datasets/puneet6060/intel-image-classification/data