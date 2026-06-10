import argparse
import os
import sys
sys.path.append("./")
from collections import OrderedDict
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import logging
import numpy as np
import torch
import torch.nn.functional as F
from datasets import voc, coco
from model.model_diclip import DiCLIP_model
from model.rar import recursive_refinement_with_diff_attn
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm
from utils import evaluate, imutils
from utils.camutils import cure_attr_map_flip, cam_to_label_iseg, cams_to_affinity_label, get_mask_by_radius
from utils.affutils import refine_cams_with_aff, refine_cams_with_bkg_weclip
from utils.visual_self_attn import visual_self_attn
from utils.pyutils import setup_logger, format_tabs_multi_metircs
from utils.dcrf import DenseCRF
from utils.PAR import PAR

import joblib
import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()

parser.add_argument("--model_path", default="./00_sota/voc/checkpoints/model_sota.pth", type=str, help="model_path")
parser.add_argument("--model", default="DiCLIP_ViT-B/16", type=str, help="custom clip")
parser.add_argument("--dataset_name", default="pascal_voc", type=str, help="custom clip")
parser.add_argument("--cache_file", default="./datasets/dif_voc/diffusion_v2_1_kv_cache_with_clustering_10_centroids_per_cls_with_bkg.pth", type=str, help="custom clip")
parser.add_argument("--adapter_size", default=312, type=int, help="adapter_size")
parser.add_argument("--embedding_dim", default=256, type=int, help="number of attribution tokens")
parser.add_argument("--in_channels", default=768, type=int, help="number of attribution tokens")
parser.add_argument("--resize_size", default=320, type=int, help="resize the long side")

parser.add_argument("--infer_set", default="val", type=str, help="infer_set")
parser.add_argument("--training_free", default=True, type=lambda x: x.lower() in ["true", "1", "yes"], help="take cam as seg")
parser.add_argument("--refine_with_aff", default=True, type=lambda x: x.lower() in ["true", "1", "yes"], help="refine_cam_with_multiscale")
parser.add_argument("--crf_post", default=False, type=lambda x: x.lower() in ["true", "1", "yes"], help="take cam as seg")
parser.add_argument("--save_cls_specific_cam", default=True, type=lambda x: x.lower() in ["true", "1", "yes"], help="save the cam figs")
parser.add_argument("--save_cam", default=False, type=lambda x: x.lower() in ["true", "1", "yes"], help="save the cam figs")

#! TO DO
####refine_raw_CAM 2 masks with multiscales, if False output cam_seeds, else output refined_masks
parser.add_argument("--data_folder", default='/data/Datasets/VOC/VOC2012/', type=str, help="dataset folder")
parser.add_argument("--list_folder", default='datasets/voc', type=str, help="train/val/test list file")
parser.add_argument("--num_classes", default=21, type=int, help="number of classes")

parser.add_argument("--ignore_index", default=255, type=int, help="random index")
parser.add_argument("--bkg_thre", default=0.5, type=float, help="work_dir")

parser.add_argument("--nproc_per_node", default=8, type=int, help="nproc_per_node")
parser.add_argument("--local_rank", default=0, type=int, help="local_rank")
parser.add_argument('--backend', default='nccl')

def validate(args=None):

    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend=args.backend, )
    val_dataset = voc.VOC12SegDataset(
        root_dir=args.data_folder,
        name_list_dir=args.list_folder,
        split=args.infer_set,
        stage=args.infer_set,
        aug=False,
        ignore_index=args.ignore_index,
        num_classes=args.num_classes,
    )

    model = DiCLIP_model(
                        clip_model=args.model, embedding_dim=args.embedding_dim, in_channels=args.in_channels, \
                        adapter_size=args.adapter_size, \
                        num_classes=args.num_classes, cache_file=args.cache_file,\
                        img_size=args.resize_size, mode=args.infer_set, device='cuda')

    if not args.training_free:
        trained_state_dict = torch.load(args.model_path, map_location="cpu")
        new_state_dict = OrderedDict()
        for k, v in trained_state_dict.items():
            k = k.replace('module.', '')
            # new_state_dict[k] = v
            if 'encoder.visual.positional_embedding' not in k:
                new_state_dict[k] = v
        model.load_state_dict(state_dict=new_state_dict, strict=False)

    model.to(torch.device(args.local_rank))
    model.eval()

    model = DistributedDataParallel(model, device_ids=[args.local_rank],)
    n_gpus = dist.get_world_size()
    split_dataset = [torch.utils.data.Subset(val_dataset, np.arange(i, len(val_dataset), n_gpus)) for i in range (n_gpus)]
    val_loader = DataLoader(split_dataset[args.local_rank], batch_size=1, shuffle=False, num_workers=2, pin_memory=False)
    par = PAR(num_iter=20, dilations=[1,2,4,8,12,24]).cuda()

    results = build_validation(model=model, par=par, val_loader=val_loader, device='cuda', args=args)
    torch.cuda.empty_cache()

    if args.crf_post:
        crf_score = crf_proc()
    return True

def build_validation(model=None, par=None, val_loader=None, device='cuda', args=None):

    gts, sms_attr_aff = [], [],
    color_map = plt.get_cmap("jet")

    model.eval()

    with torch.no_grad():
        for _, data in tqdm(enumerate(val_loader), total=len(val_loader), ncols=100, ascii=" >="):
            name, inputs, labels, cls_labels = data
            img = imutils.denormalize_img(inputs)[0].permute(1,2,0).numpy()
            inputs = inputs.to(device, non_blocking=True)            
            inputs  = F.interpolate(inputs, size=[args.resize_size, args.resize_size], mode='bilinear', align_corners=False)

            cls_labels = cls_labels.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            labels_re = F.interpolate(labels.unsqueeze(0), size=[args.resize_size, args.resize_size], mode='nearest')[0]
            
            label_aff = cams_to_affinity_label(labels_re)
            label_aff[label_aff==255] = 0 
            label_aff[label_aff!=0] = 0 

            return_dynamic = False if args.training_free else True
            attr_maps_raw, attn_weights, diff_attn, segs, attn_pred, _ = model(inputs, return_dynamic)

            # visual_self_attn(diff_attn_down[0,0,...],imutils.denormalize_img(inputs),labels_re,name=name,save_root='./w_outputs/xxx')

            for i, attr_map in enumerate(attr_maps_raw):

                cls_label = cls_labels[i]
                attn_weight = attn_weights[:,i,...]
                refined_attr_maps_raw, cls_lst = refine_cams_with_aff(attr_map, attn_weight, cls_label, size=inputs.shape[2:], seg_attn=None, caa_thre=0.88)
                # refined_attr_maps_raw, cls_lst = refine_cams_with_aff2(attr_map, attn_weight, cls_label, size=inputs.shape[2:], seg_attn=None, caa_thre=0.79, cluster_attn=diff_attn)
                pred_mask, valid_lam = refine_cams_with_bkg_weclip(refined_attr_maps_raw, inputs[i], cls_lst, par, labels.shape[-2:])
                cams = valid_lam[1:,...]

                ###TODO recursive affinity refinment
                refined_attr_maps_raw = torch.stack(refined_attr_maps_raw,dim=0).flatten(-2,-1).permute(1,0).unsqueeze(0)
                valid_lam, pred_mask = recursive_refinement_with_diff_attn(cls_labels[i], refined_attr_maps_raw, diff_attn[i], mask_size=labels.shape[-2:],iter=10)
                cams = valid_lam[0,1:][cls_lst,...]

                if args.save_cam:
                    resized_attr_maps = cams # exclude bkg
                    cam_np = torch.max(resized_attr_maps, dim=0)[0].cpu().numpy()
                    cam_rgb = color_map(cam_np)[:,:,:3] * 255
                    alpha = 0.5
                    cam_rgb = alpha*cam_rgb + (1-alpha)*img
                    if not args.save_cls_specific_cam:
                        imageio.imsave(os.path.join(args.cam_dir, name[0] + ".jpg"), cam_rgb.astype(np.uint8))
                    else:
                        for cam,idx in zip(resized_attr_maps, cls_lst):
                            cam_np = cam.cpu().numpy()
                            cam_rgb = color_map(cam_np)[:,:,:3] * 255
                            alpha = 0.6
                            cam_rgb = alpha*cam_rgb + (1-alpha)*img
                            imageio.imsave(os.path.join(args.cs_cam_dir, name[0] + f"_{voc.class_list[idx+1]}.jpg"), cam_rgb.astype(np.uint8))

                    imageio.imsave(args.segs_rgb_dir + "/" + name[0] + ".png", imutils.encode_cmap(np.squeeze(pred_mask.cpu().numpy())).astype(np.uint8))

                sms_attr_aff += list(pred_mask.cpu().numpy().astype(np.int16))        
                gts += list(labels.cpu().numpy().astype(np.int16))

                if args.crf_post:
                    keys_gt = cls_lst
                    valid_lam = cams
                    np.save(args.logits_dir + "/" + name[0] + '.npy', {"valid_lam":valid_lam.cpu().numpy(),"keys_gt":keys_gt.cpu().numpy()})

    attr_aff_score = evaluate.scores(gts, sms_attr_aff, num_classes=args.num_classes)
    model.train()
    cat_list=voc.class_list if 'voc' in args.dataset_name else coco.class_list
    tab_results = format_tabs_multi_metircs([attr_aff_score], ["confusion","precision","recall",'iou'], cat_list=cat_list)
    logging.info(f'Training_free:{args.training_free}, LAM_score:')
    logging.info("\n"+tab_results)

    return tab_results


def crf_proc():
    print("crf post-processing...")

    txt_name = os.path.join(args.list_folder, args.infer_set) + '.txt'
    with open(txt_name) as f:
        name_list = [x for x in f.read().split('\n') if x]

    images_path = os.path.join(args.data_folder, 'JPEGImages',)
    labels_path = os.path.join(args.data_folder, 'SegmentationClassAug')

    post_processor = DenseCRF(
        iter_max=10,
        pos_xy_std=1,
        pos_w=3,
        bi_xy_std=67,
        bi_rgb_std=3,
        bi_w=4,
    )

    def _job(i):

        name = name_list[i]
        logit_name = args.logits_dir + "/" + name + ".npy"
        logit_ = np.load(logit_name, allow_pickle=True).item()
        lams = logit_['valid_lam']
        keys = logit_['keys_gt']


        image_name = os.path.join(images_path, name + ".jpg")
        image = imageio.imread(image_name).astype(np.float32)
        label_name = os.path.join(labels_path, name + ".png")

        if "test" in args.infer_set:
            label = image[:,:,0]
        else:
            label = imageio.imread(label_name)

        prob = lams

        image = image.astype(np.uint8)
        prob = post_processor(image, prob)
        pred = np.argmax(prob, axis=0)
        keys = np.pad(keys+1, (1, 0), mode='constant')
        pred_crf = keys[pred].astype(np.uint8)
        imageio.imsave(args.segs_crf_rgb_dir + "/" + name + ".png", imutils.encode_cmap(np.squeeze(pred_crf)).astype(np.uint8))

        return pred_crf,label

    n_jobs = int(os.cpu_count() * 0.6)
    results = joblib.Parallel(n_jobs=n_jobs, verbose=10, pre_dispatch="all")([joblib.delayed(_job)(i) for i in range(len(name_list))])
    preds, gts = zip(*results)

    crf_score = evaluate.scores(gts, preds)
    logging.info('crf_seg_score:')
    metrics_tab_crf = format_tabs_multi_metircs([crf_score], ["confusion","precision","recall",'iou'], cat_list=voc.class_list)
    logging.info("\n"+ metrics_tab_crf)

    return crf_score

if __name__ == "__main__":

    args = parser.parse_args()
    base_dir = args.model_path.split("checkpoints/")[0] + f'/{args.infer_set}/'
    cpt_name = args.model_path.split("checkpoints/")[-1].replace('.pth','')

    if args.training_free:
        tag = 'lam_training_free/aff_lam' if args.refine_with_aff else 'lam_training_free/seeds_lam'
    else:
        tag = 'lam_optimized/aff_lam' if args.refine_with_aff else 'lam_optimized/seeds_lam'

    if args.crf_post:
        args.logits_dir = os.path.join(base_dir, f"{args.infer_set}_{cpt_name}_{tag}_logits")
        args.segs_dir = os.path.join(base_dir, f"{args.infer_set}_{cpt_name}_{tag}_segs/seg_preds")
        args.segs_rgb_dir = os.path.join(base_dir, f"{args.infer_set}_{cpt_name}_{tag}_segs/seg_preds_rgb")
        args.segs_crf_rgb_dir = os.path.join(base_dir, f"{args.infer_set}_{cpt_name}_{tag}_segs/segcrf_preds_rgb")
        os.makedirs(args.logits_dir, exist_ok=True)
        os.makedirs(args.segs_dir, exist_ok=True)
        os.makedirs(args.segs_rgb_dir, exist_ok=True)
        os.makedirs(args.segs_crf_rgb_dir, exist_ok=True)

    if args.save_cls_specific_cam:
        args.cs_cam_dir = os.path.join(base_dir, f"{args.infer_set}_{cpt_name}_{tag}_class_specific_img")
        args.segs_rgb_dir = os.path.join(base_dir, f"{args.infer_set}_{cpt_name}_{tag}_segs/seg_preds_rgb")

        os.makedirs(args.segs_rgb_dir, exist_ok=True)
        os.makedirs(args.cs_cam_dir, exist_ok=True)

    args.cam_dir = os.path.join(base_dir, f"{args.infer_set}_{cpt_name}_{tag}_img")
    args.log_dir = os.path.join(base_dir, f"{args.infer_set}_{cpt_name}_{tag}_results.log")

    os.makedirs(args.cam_dir, exist_ok=True)
    setup_logger(filename=args.log_dir)
    logging.info('Pytorch version: %s' % torch.__version__)
    logging.info("GPU type: %s"%(torch.cuda.get_device_name(0)))
    logging.info('\nargs: %s' % args)

    validate(args=args)