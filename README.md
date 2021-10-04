# Region of Interest Detection in Melanocytic Skin Tumor Whole Slide Images

## Updates:
* 10/04/2021: Submitted to ISBI 2022. See visualization results on TCGA-SKCM at [following link]()

## Overview
Automated region of interest detection in histopathological image analysis is a challenging and important topic with tremendous potential impact in clinical practice. In this paper, we aim to address one important question: How to perform automated region of interest detection in melanocytic skin tumor (*melanoma* and *nevus*) whole slide images? We work with a dataset that contains 161 melanocytic skin tumor images, including *melanomas* (skin cancer) and *nevi* (benign moles), and propose a patch-based region of interest detection method, which achieves better performance than other unsupervised and supervised methods on the melanocytic skin tumor images.

<img src="https://github.com/roidetection/roi_detection/blob/master/main_fig.jpg" width="1024"/>

## Setup

### 1. Prerequisites
- Linux (Tested on Ubuntu 18.04)
- NVIDIA GPU (Tested on Nvidia GeForce RTX 3090 on local workstations)
- CUDA (Tested on CUDA 11.3)
- torch>=1.7.1

### 2. Code Base Structure
The code base structure is explained below: 
- **create_patches.py**: extract patches from whole slide images (.svs). If annotations are provided, patches inside annotated regions will be generated. If annotations are not available, contour detection will be performed and all patches inside contour will be generated.
- **extract_patches.py**: split annotated patches into training, testing and validation.
- **method.py**: train patch classification model on annotated patches.
- **score.py**: compute predicted scores for all patches from WSI with trained model.
- **visual.py**: generate visualization maps.

The directory structure for your multimodal dataset should look similar to the following:
```bash
./
├── data
      └── PROJECT
            ├── INPUT A (e.g. Image)
                ├── image_001.png
                ├── image_002.png
                ├── ...
            ├── INPUT B (e.g. Graph)
                ├── image_001.pkl
                ├── image_002.pkl
                ├── ...
            └── INPUT C (e.g. Genomic)
                └── genomic_data.csv
└── checkpoints
        └── PROJECT
            ├── TASK X (e.g. Survival Analysis)
                ├── path
                    ├── ...
                ├── ...
            └── TASK Y (e.g. Grade Classification)
                ├── path
                    ├── ...
                ├── ...
```

Depending on which modalities you are interested in combining, you must: (1) write your own function for aligning multimodal data in **make_splits.py**, (2) create your DatasetLoader in **data_loaders.py**, (3) modify the **options.py** for your data and task. Models will be saved to the **checkpoints** directory, with each model for each task saved in its own directory. At the moment, the only supervised learning tasks implemented are survival outcome prediction and grade classification.

### 3. Training and Evaluation
Here are example commands for training unimodal + multimodal networks.

#### Survival Model for Input A
Example shown below for training a survival model for mode A and saving the model checkpoints + predictions at the end of each split. In this example, we would create a folder called "CNN_A" in "./checkpoints/example/" for all the models in cross-validation. It assumes that "A" is defined as a mode in **dataset_loaders.py** for handling modality-specific data-preprocessing steps (random crop + flip + jittering for images), and that there is a network defined for input A in **networks.py**. "surv" is already defined as a task for training networks for survival analysis in **options.py, networks.py, train_test.py, train_cv.py**.

```
python train_cv.py --exp_name surv --dataroot ./data/example/ --checkpoints_dir ./checkpoints/example/ --task surv --mode A --model_name CNN_A --niter 0 --niter_decay 50 --batch_size 64 --reg_type none --init_type max --lr 0.002 --weight_decay 4e-4 --gpu_ids 0
```
To obtain test predictions on only the test splits in your cross-validation, you can replace "train_cv" with "test_cv".
```
python test_cv.py --exp_name surv --dataroot ./data/example/ --checkpoints_dir ./checkpoints/example/ --task surv --mode input_A --model input_A_CNN --niter 0 --niter_decay 50 --batch_size 64 --reg_type none --init_type max --lr 0.002 --weight_decay 4e-4 --gpu_ids 0
```

#### Grade Classification Model for Input A + B
Example shown below for training a grade classification model for fusing modes A and B. Similar to the previous example, we would create a folder called "Fusion_AB" in "./checkpoints/example/" for all the models in cross-validation. It assumes that "AB" is defined as a mode in **dataset_loaders.py** for handling multiple inputs A and B at the same time. "grad" is already defined as a task for training networks for grade classification in **options.py, networks.py, train_test.py, train_cv.py**.
```
python train_cv.py --exp_name surv --dataroot ./data/example/ --checkpoints_dir ./checkpoints/example/ --task grad --mode AB --model_name Fusion_AB --niter 0 --niter_decay 50 --batch_size 64 --reg_type none --init_type max --lr 0.002 --weight_decay 4e-4 --gpu_ids 0
```

## Reproducibility
To reporduce the results in our paper and for exact data preprocessing, implementation, and experimental details please follow the instructions here: [./data/TCGA_GBMLGG/](https://github.com/mahmoodlab/PathomicFusion/tree/master/data/TCGA_GBMLGG). Processed data and trained models can be downloaded [here](https://drive.google.com/drive/folders/1swiMrz84V3iuzk8x99vGIBd5FCVncOlf?usp=sharing).

## Issues
- Please report all issues on the public forum.

## Acknowledgments
- This code of patch extraction is inspired by [CLAM](https://github.com/huangzhii/SALMON).

