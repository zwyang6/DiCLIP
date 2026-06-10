import argparse
import os
import sys
sys.path.append("./")
import numpy as np
import torch
import torch.nn.functional as F
from datasets import coco
from datasets import dif as voc
from model.model_diclip import DiCLIP_model
from sklearn.cluster import KMeans

from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()

parser.add_argument("--model_path", default="./w_outputs/checkpoints/frozen_30000.pth", type=str, help="model_path")
parser.add_argument("--model", default="ExCEL_ViT-B/16", type=str, help="custom clip")
parser.add_argument("--dataset_name", default="pascal_voc", type=str, help="custom clip")
parser.add_argument("--attr_json", default="./attributes_text/descriptors_pascal_voc_gpt4.0_cluster_a_photo_of4.json", type=str, help="custom clip")
parser.add_argument("--num_attri", default=112, type=int, help="number of attribution tokens")
parser.add_argument("--embedding_dim", default=256, type=int, help="number of attribution tokens")
parser.add_argument("--in_channels", default=768, type=int, help="number of attribution tokens")
parser.add_argument("--resize_size", default=320, type=int, help="resize the long side")

parser.add_argument("--infer_set", default="dif_data", type=str, help="infer_set")
parser.add_argument("--num_cluster", default=5, type=int, help="resize the long side")
#! TO DO
####refine_raw_CAM 2 masks with multiscales, if False output cam_seeds, else output refined_masks
parser.add_argument("--data_folder", default='/data/PROJECTS/Diffusion_CLIP/DiT_base_diffusion/datasets/dif/', type=str, help="dataset folder")
parser.add_argument("--list_folder", default='datasets/dif', type=str, help="train/val/test list file")
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
        stage='generate_kv_cache',
        aug=False,
        ignore_index=args.ignore_index,
        num_classes=args.num_classes,
    )

    model = DiCLIP_model(
                        clip_model=args.model, embedding_dim=args.embedding_dim, in_channels=args.in_channels, \
                        dataset_name=args.dataset_name, \
                        num_classes=args.num_classes, num_atrr_clusters=args.num_attri, json_file=args.attr_json,\
                        img_size=args.resize_size, mode=args.infer_set, device='cuda')


    model.to(torch.device(args.local_rank))
    model.eval()

    model = DistributedDataParallel(model, device_ids=[args.local_rank],)
    n_gpus = dist.get_world_size()
    split_dataset = [torch.utils.data.Subset(val_dataset, np.arange(i, len(val_dataset), n_gpus)) for i in range (n_gpus)]
    val_loader = DataLoader(split_dataset[args.local_rank], batch_size=1, shuffle=False, num_workers=2, pin_memory=False)

    build_validation(model=model, num_cluster=args.num_cluster, val_loader=val_loader, device='cuda', args=args)
    torch.cuda.empty_cache()

    return True

def build_validation(model=None, num_cluster=None, val_loader=None, device='cuda', args=None):

    model.eval()
    cache_key={f'{cls}': [] for cls in voc.class_list[1:]}
    cache_value={f'{cls}': [] for cls in voc.class_list[1:]}

    with torch.no_grad():
        raw_pth_file = f'datasets/dif/diffusion_v2_1_kv_cache_raw.pth'
        if os.path.exists(raw_pth_file):
            cache_key, cache_value = torch.load(raw_pth_file)
            print(f'{raw_pth_file} loaded')
        else:
            for i, data in tqdm(enumerate(val_loader), total=len(val_loader), ncols=100, ascii=" >="):
                name, inputs, labels, cls_labels = data
                inputs = inputs.to(device, non_blocking=True)            
                inputs  = F.interpolate(inputs, size=[args.resize_size, args.resize_size], mode='bilinear', align_corners=False)

                cls_labels = cls_labels.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                labels_down = F.interpolate(labels.unsqueeze(0), size=[args.resize_size // 16, args.resize_size // 16], mode='nearest')[0]
                labels_re = F.interpolate(labels.unsqueeze(0), size=[args.resize_size, args.resize_size], mode='nearest')

                labels_re[labels_re>0] = 1
                feats = model(inputs, return_feats=True)
                # attr_maps_raw, attn_weights, diff_attn, diff_attn_down = model(inputs*labels_re)
                masks = labels_down.reshape(1,-1).unsqueeze(-1)
                masks_binary = masks > 0
                if masks_binary.sum() > 0:
                    cls_idx = cls_labels[0].argmax()
                    cls_name = voc.class_list[1:][cls_idx]
                    key = feats[:,1:,:]*masks_binary
                    key_norm = key.sum(1)/ masks_binary.sum(1)
                    cache_key[cls_name].append(key_norm)
                    cache_value[cls_name].append(cls_labels)
                else:
                    continue
            
            raw_kv_cache=[cache_key, cache_value]
            print(f'diffusion_v2_1_kv_cache_raw saved.')
            torch.save(raw_kv_cache, f'/data/PROJECTS/Diffusion_CLIP/DiT_base_diffusion/datasets/dif/diffusion_v2_1_kv_cache_raw.pth')

        cluster_keys = []
        cluster_values = []
        for i, cls_name in enumerate(voc.class_list[1:]):
            cls_key = torch.cat(cache_key[cls_name]).cpu().numpy()
            cls_value = torch.cat(cache_value[cls_name])
            if num_cluster is not None:
                tag = f'with_clustering_{num_cluster}_centroids_per_cls'
                kmeans = KMeans(n_clusters=num_cluster, random_state=0).fit(cls_key)
                cluster_embedding = kmeans.cluster_centers_
                cluster_embedding = torch.tensor(cluster_embedding)
                cluster_value = cls_value[:num_cluster]
            else:
                tag = 'no_clustering'
                cluster_embedding = cls_key
                cluster_value = cls_value

            cluster_keys.append(cluster_embedding)
            cluster_values.append(cluster_value)

        cache_keys = torch.cat(cluster_keys)
        cache_values = torch.cat(cluster_values)
        kv_cache = [cache_keys, cache_values]

        print(f'diffusion_v2_1_kv_cache_{tag} saved.')
        torch.save(kv_cache, f'/data/PROJECTS/Diffusion_CLIP/DiT_base_diffusion/datasets/dif/diffusion_v2_1_kv_cache_{tag}.pth')

    return kv_cache

if __name__ == "__main__":

    args = parser.parse_args()
    base_dir = args.model_path.split("checkpoints/")[0] + f'/{args.infer_set}/'
    cpt_name = args.model_path.split("checkpoints/")[-1].replace('.pth','')

    validate(args=args)