# [UnderReview] DiCLIP: Diffusion Model Enhances CLIP’s Dense Knowledge for Weakly Supervised Semantic Segmentation

<!-- We propose DiCLIP, a novel WSSS framework, which leverages the generative diffusion model to enhance CLIP's dense knowledge across two modalities. -->

## News

* **If you find this work helpful, please give us a :star2: to receive the updation !**
* **` Mar. 28th, 2025`:** DiCLIP is Submitted.
* **Code will be available once DiCLIP is accepted...** 🔥🔥🔥

## Overview

We propose DiCLIP, a novel WSSS framework, which leverages the generative diffusion model to enhance CLIP's dense knowledge across vision and text modalities

<p align="middle">
<img src="/sources/overview.png" alt="DiCLIP pipeline" width="600px">
</p>


## Data Preparation

### PASCAL VOC 2012

#### 1. Download

``` bash
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
```
#### 2. Segmentation Labels

The augmented annotations are from [SBD dataset](http://home.bharathh.info/pubs/codes/SBD/download.html). The download link of the augmented annotations at
[DropBox](https://www.dropbox.com/s/oeu149j8qtbs1x0/SegmentationClassAug.zip?dl=0). After downloading ` SegmentationClassAug.zip `, you should unzip it and move it to `VOCdevkit/VOC2012/`. 

``` bash
VOCdevkit/
└── VOC2012
    ├── Annotations
    ├── ImageSets
    ├── JPEGImages
    ├── SegmentationClass
    ├── SegmentationClassAug
    └── SegmentationObject
```

### MSCOCO 2014

#### 1. Download
``` bash
wget http://images.cocodataset.org/zips/train2014.zip
wget http://images.cocodataset.org/zips/val2014.zip
```

#### 2. Segmentation Labels

To generate VOC style segmentation labels for COCO, you could use the scripts provided at this [repo](https://github.com/alicranck/coco2voc), or just download the generated masks from [Google Drive](https://drive.google.com/file/d/147kbmwiXUnd2dW9_j8L5L0qwFYHUcP9I/view?usp=share_link).

``` bash
COCO/
├── JPEGImages
│    ├── train2014
│    └── val2014
└── SegmentationClass
     ├── train2014
     └── val2014
```

## Requirement

Please refer to the requirements.txt. 

## Train DiCLIP
``` bash
### train voc
bash run_train.sh scripts/train_voc.py [gpu_device] [gpu_number] [master_port]  train_voc

### train coco
bash run_train.sh scripts/train_coco.py [gpu_devices] [gpu_numbers] [master_port] train_coco
```

## Evaluate DiCLIP
``` bash
### eval voc seg and LAM
bash run_evaluate_voc.sh tools/infer_lam.py [gpu_device] [gpu_number] [infer_set] [checkpoint_path]

### eval coco seg
bash run_evaluate_seg_coco.sh tools/infer_seg_coco.py [gpu_device] [gpu_number] [infer_set] [checkpoint_path]
```

## Main Results

* **Quantitative Results**
  
Semantic performance on VOC and COCO. Logs are available now. Checkpoints will be available soon.
| Dataset | Backbone |  Val  | Test | Log | Weight |
|:-------:|:--------:|:-----:|:----:|:---:|:------:|
|   PASCAL VOC   |   ViT-B  | 78.8  | 78.9 | [log](logs/voc_train.log) | Checkpoint    |
|   MS COCO      |   ViT-B  |  48.7 |   -  | [log](logs/coco_train.log)| Checkpoint    |

* **Qualitative Results**

1. CAM Comparison
<p align="middle">
<img src="/sources/cam.png" alt="DiCLIP results" width="1200px">
</p>

2. VOC Segmentation
<p align="middle">
<img src="/sources/voc.png" alt="DiCLIP results" width="1200px">
</p>

3. COCO Segmentation
<p align="middle">
<img src="/sources/coco.png" alt="DiCLIP results" width="1200px">
</p>

## Citation 
Please cite our work if you find it helpful to your reseach. :two_hearts:
