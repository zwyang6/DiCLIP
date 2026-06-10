# [TIP2026] DiCLIP: Diffusion Model Enhances CLIP’s Dense Knowledge for Weakly Supervised Semantic Segmentation

## News

* **` Mar. 28th, 2025`:** DiCLIP is Submitted.
* **` May. 4th, 2026`:** DiCLIP is Accepted by IEEE Transactions on Image Processing!
* **`Jun. 10th, 2026`:** **All Code, Data, and Checkpoints are released!** 🤗🤗🤗
* **If you find this work helpful, please give us a :star2: to receive the updation !**

## Overview

<p align="middle">
<img src="/sources/main_fig.png" alt="DiCLIP pipeline" width="900px">
</p>

Weakly Supervised Semantic Segmentation (WSSS) with image-level labels typically leverages Class Activation Maps (CAMs) to achieve pixel-level predictions. Recently, Contrastive Language-Image Pre-training (CLIP) has been introduced to generate CAMs in WSSS. However, previous WSSS methods solely adopt CLIP’s vision-language paired property for dense localization, neglecting its inherently limited dense knowledge across both visual and text modalities, which renders CAM generation suboptimal. In this work, we propose DiCLIP, a novel WSSS framework that leverages the generative diffusion model to enhance CLIP’s dense knowledge across two modalities. Specifically, Visual Correlation Enhancement (VCE) and Text Semantic Augmentation (TSA) modules are proposed for dense prediction enhancement. To improve the spatial awareness of visual features, our VCE module utilizes diffusion’s reliable spatial consistency to mitigate the over-smoothing issue in CLIP’s attention. It designs the Attention Clustering Refinement (ACR) module to reliably extract diverse correlation maps from the diffusion model. The correlation maps act as a diversity bias for CLIP’s self-attention, recursively pushing its visual features towards a more discriminative dense distribution. To augment the semantics of text embeddings, our TSA module argues that a single text modality is insufficient to encompass the variability of visual categories. Thus, we leverage diffusion’s generative power to maintain a dynamic key-value cache model, shifting CAM gen- eration from a patch-text matching mechanism to a novel visual knowledge retrieval paradigm. With these enhancements, DiCLIP not only outperforms state-of-the-art methods on PASCAL VOC and MS COCO but also significantly reduces training costs.


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

## Construction of KV Cache

#### 1. Generate Stable Diffusion Images

To construct the SD-derived KV cache, single-class images for the VOC and COCO categories are required. You can either:

- Follow the instructions in `maintain_kv_cache/SD_generate_imgs/README.md` to generate the images from scratch; or
- Directly use our pre-generated image set available [HERE](https://drive.google.com/file/d/1EMfbMTzaY6sz3iwfeApVbUDNQwDgGijv/view?usp=drive_link).

#### 2. Maintain the KV Cache

We also provide the pre-built KV cache in `datasets/dif_voc`. If you would like to construct your own cache, please refer to:

```bash
python maintain_kv_cache/generate_kv_cache.py
```

to generate a customized KV cache from the synthesized images.

## Train DiCLIP
``` bash
### train voc
bash run_train_voc.sh scripts/train_voc.py [gpu_device] [gpu_number] [master_port]  train_voc

### train coco
bash run_train_coco.sh scripts/train_coco.py [gpu_devices] [gpu_numbers] [master_port] train_coco
```

## Evaluate DiCLIP
``` bash
### eval voc LAM
bash ./infer_lam_voc.sh tools/infer_lam.py [gpu_device] [gpu_number] [infer_set] [checkpoint_path]

### eval voc seg
bash ./infer_voc_coco.sh tools/infer_seg_voc.py [gpu_device] [gpu_number] [infer_set] [checkpoint_path]

### eval coco seg
bash ./infer_seg_coco.sh tools/infer_seg_coco.py [gpu_device] [gpu_number] [infer_set] [checkpoint_path]
```

## Main Results

* **Quantitative Results**
  
Semantic performance on VOC and COCO. Logs are available now. Checkpoints will be available soon.
| Dataset | Backbone |  Val  | Test | Log | Weight |
|:-------:|:--------:|:-----:|:----:|:---:|:------:|
|   PASCAL VOC   |   ViT-B  | 78.8  | 78.9 | [log](logs/voc_train.log) | [Checkpoint](https://drive.google.com/drive/folders/17Y7b6P2OODQVDwUfBNZNgmJzMkbYgQEV?usp=drive_link)    |
|   MS COCO      |   ViT-B  |  48.7 |   -  | [log](logs/coco_train.log)| [Checkpoint](https://drive.google.com/drive/folders/1GYqF_LCGW3UBXUGGbs4AMLvfcstnyvN3?usp=drive_link)    |

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
```bash
@article{yang2026diclip,
  title={DiCLIP: Diffusion Model Enhances CLIP’s Dense Knowledge for Weakly Supervised Semantic Segmentation},
  author={Yang, Zhiwei and Song, Pengfei and Meng, Yucong and Fu, Kexue and Wang, Shuo and Song, Zhijian},
  journal={IEEE Transactions on Image Processing},
  year={2026},
  publisher={IEEE}
}
```

## Acknowledgement
This repo is built upon [ExCEL](https://github.com/zwyang6/ExCEL) and [Tip-Adapter](https://github.com/gaopengcuhk/Tip-Adapter). Many thanks to their brilliant works!!!
