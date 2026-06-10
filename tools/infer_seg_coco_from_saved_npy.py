import argparse
import os
import sys
import logging
sys.path.append("./")

from collections import OrderedDict
import imageio.v2 as imageio
import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from datasets import coco
from model.model_diclip import DiCLIP_model
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import evaluate, imutils
from utils.dcrf import DenseCRF
from utils.pyutils import format_tabs_multi_metircs, setup_logger, convert_test_seg2RGB
from utils.reload import reload_cpt

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", default="/data/PROJECTS/ExCEL+/ExCEL_Plus/00_sota/coco/checkpoints/model_iter_100000.pth", type=str, help="model_path")
parser.add_argument("--model", default="ExCEL_ViT-B/16", type=str, help="custom clip")
parser.add_argument("--cache_file", default="/data/PROJECTS/Diffusion_CLIP/DiT_base_diffusion/datasets/dif_coco/diffusion_v2_1_kv_cache_with_clustering_5_centroids_per_cls_with_bkg_coco.pth", type=str, help="custom clip")
parser.add_argument("--dataset_name", default="ms_coco", type=str, help="custom clip")
parser.add_argument("--num_attri", default=224, type=int, help="number of attribution tokens")
parser.add_argument("--embedding_dim", default=256, type=int, help="number of attribution tokens")
parser.add_argument("--in_channels", default=768, type=int, help="number of attribution tokens")
parser.add_argument("--crf_post", default=True, type=lambda x: x.lower() in ["true", "1", "yes"], help="take cam as seg")
parser.add_argument("--resize_size", default=320, type=int, help="resize the long side")
parser.add_argument("--scales", default=[0.7, 1.0, 1.2, 1.5], help="multi_scales for seg")
# parser.add_argument("--scales", default=[0.7, 1.0, 1.1, 1.2, 1.5], help="multi_scales for seg")

#! TO DO
## infer valset or testset: The test datafolder is different from valtrain folder
parser.add_argument("--infer_set", default="val", type=str, help="infer_set")
parser.add_argument("--data_folder", default='/data/Datasets/MSCOCO2014/', type=str, help="dataset folder")
parser.add_argument("--pooling", default="gmp", type=str, help="pooling method")
parser.add_argument("--list_folder", default='datasets/coco', type=str, help="train/val/test list file")
parser.add_argument("--pretrained", default=True, type=bool, help="use imagenet pretrained weights")
parser.add_argument("--num_classes", default=81, type=int, help="number of classes")
parser.add_argument("--ignore_index", default=255, type=int, help="random index")

def _validate(data_loader=None, args=None):


    with torch.no_grad(), torch.cuda.device(0):


        gts, seg_pred = [], []

        for idx, data in tqdm(enumerate(data_loader), total=len(data_loader), ncols=100, ascii=" >="):

            name, inputs, labels, cls_label = data

            inputs = inputs.cuda()
            labels = labels.cuda()
            cls_label = cls_label.cuda()
            _, _, h, w = inputs.shape
            gts += list(labels.cpu().numpy().astype(np.int16))

            logit_name = args.logits_dir + "/" + name[0] + ".npy"

            logit = np.load(logit_name, allow_pickle=True).item()
            logit = logit['msc_seg']

            logit = torch.FloatTensor(logit)#[None, ...]
            logit = F.interpolate(logit, size=(h, w), mode="bilinear", align_corners=False)
            seg_pred += list(torch.argmax(logit, dim=1).cpu().numpy().astype(np.int16))

    seg_score = evaluate.scores(gts, seg_pred, num_classes=args.num_classes)
    logging.info('raw_seg_score:')
    metrics_tab = format_tabs_multi_metircs([seg_score], ["confusion","precision","recall",'iou'], cat_list=coco.class_list)
    logging.info("\n"+metrics_tab)
    
    return seg_score

def validate(args=None):

    val_dataset = coco.CocoSegDataset(
        root_dir=args.data_folder,
        name_list_dir=args.list_folder,
        split=args.infer_set,
        stage=args.infer_set,
        aug=False,
        ignore_index=args.ignore_index,
        num_classes=args.num_classes,
    )

    val_loader = DataLoader(val_dataset,
                            batch_size=1,
                            shuffle=False,
                            num_workers=8,
                            pin_memory=False,
                            drop_last=False)




    seg_score = _validate(data_loader=val_loader, args=args)
    torch.cuda.empty_cache()
    
    return True

if __name__ == "__main__":

    args = parser.parse_args()
    base_dir = args.model_path.split("checkpoints/")[0] + f'/{args.infer_set}/'
    cpt_name = args.model_path.split("checkpoints/")[-1].replace('.pth','')
    args.logits_dir = '/data/PROJECTS/Diffusion_CLIP/DiT_base_diffusion/00_exp/00_coco_sota_47.0/codes/coco_train_coco_5_cluster_18-11-51-21/val/val_model_iter_90000_segs/logits'
    
    os.makedirs(os.path.join(base_dir, f"{args.infer_set}_{cpt_name}_segs"), exist_ok=True)
    setup_logger(filename=os.path.join(base_dir, f"{args.infer_set}_{cpt_name}_segs/results.log"))
    print(args)
    validate(args=args)
