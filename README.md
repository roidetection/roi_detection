# Region of Interest Detection in Melanocytic Skin Tumor Whole Slide Images

## Updates:
* 10/04/2021: Submitted to ISBI 2022. See visualization results on TCGA-SKCM in folder TCGA_visual.

## Overview
Automated region of interest detection in histopathological image analysis is a challenging and important topic with tremendous potential impact in clinical practice. In this paper, we aim to address one important question: How to perform automated region of interest detection in melanocytic skin tumor (*melanoma* and *nevus*) whole slide images? We work with a dataset that contains 161 melanocytic skin tumor images, including *melanomas* (skin cancer) and *nevi* (benign moles), and propose a patch-based region of interest detection method, which achieves better performance than other unsupervised and supervised methods on the melanocytic skin tumor images.

![plot](./pics/task.jpg)

## Setup

### 1. Prerequisites
- Linux (Tested on Ubuntu 18.04)
- NVIDIA GPU (Tested on Nvidia GeForce RTX 3090 on local workstations)
- CUDA (Tested on CUDA 11.3)
- torch>=1.7.1

### 2. Code Base Structure
The code base structure is explained below: 
- **create_patches.py**: extract patches from whole slide images (.svs). If annotations are provided, patches inside annotated regions will be generated. If annotations are not available, contour detection will be performed and all patches inside contour will be generated.
- **split_patches.py**: split annotated patches into training, testing and validation.
- **train.py**: train patch classification model on annotated patches.
- **score.py**: compute predicted scores for all patches from WSI with trained model.
- **visual.py**: generate visualization maps.

You need to generate a csv file that contains 'slide_id', 'data_split', 'label' for training the model.

### 3. Training and Detection
Here are example commands for training patch classification model and performing ROI detection.

#### Train Patch Classification Model
Step 1: extracting patches from whole slide images with annotation files (.xml). Depending on the annotations, the extracted patches may belong to different classes.
```
python create_patches.py --source PATH_TO_WSI --save_dir PATH_TO_SAVE_ANNOTATED_PATCHES --xml_dir PATH_TO_XML --patch --xml
```
Step 2: saving patches to corresponding directories (train/val/test) based on csv file.
```
python split_patches.py --data_dir PATH_TO_SAVE_MEL/NEVI_PATCHES/patches --other_patches_dir PATH_TO_SAVE_OTHER_PATCHES/patches --csv_path PATH_TO_CSV --feat_dir PATH_TO_SAVE_FEATURES
```
Step 3: train patch classification model.
```
python train.py --exp_name 'pcla_3class' --data_folder PATH_TO_SAVE_FEATURESs --batch_size 256 --n_epochs 20
```
#### Region of Interest Detection
Step 1: extracting patches from whole slide images without using annotations. (Testing)
```
python create_patches.py --source PATH_TO_WSI --save_dir PATH_TO_ALL_PATCHES --patch --seg
```
Step 2: calculate predicted scores for all extracted patches.
```
python score.py --exp_name 'pcla_3class' --auto_skip --model_load TRAINED_MODEL --csv_path PATH_TO_CSV --patch_path PATH_TO_ALL_PATCHES --results_dir PATH_TO_SAVE_RESULTS --classification_save_dir PATH_TO_SAVE_CLASSIFICATION_RESULTS
```
Step 3: generate overlay map.
```
python visual.py --exp_name 'pcla_3class' --csv_path PATH_TO_CSV --wsi_dir PATH_TO_WSI --results_dir PATH_TO_SAVE_RESULTS --xml_dir PATH_TO_GROUND_TRUTH_LABELS
```
By setting `--heatmap` or `--boundary`, other two types of visulization results can also be generated.
## Reproducibility
The melanocytic skin tumor dataset will be made public in the future. To reproduce the results on TCGA-SKCM dataset, the pretrained model is available in model folder.

## Issues
- Please report all issues on the public forum.

## Acknowledgments
- This code of patch extraction is inspired by [CLAM](https://github.com/mahmoodlab/CLAM).

