
import torch
import os
import matplotlib.pyplot as plt

import torch.nn.functional as F
import math
import numpy as np 

def visual_self_attn(attn, image, mask, name, save_root):

    img_size = image.shape[-2:]
    labels = F.interpolate(mask.unsqueeze(1).type(torch.float32), size=img_size, mode="nearest")[0]
    save_dir = f'{save_root}/diff_attn'
    os.makedirs(save_dir,exist_ok=True)
    anchor_point = find_foreground_center(labels)
    # attn = attn[[1, 2]].mean(0).float()
    attn = attn.float()
    save_attm_maps(image,attn,img_size,name[0], anchor_point,save_dir, mask)

def save_attm_maps(inputs, attn_, size, name, anchor=None, anchor_dir=None, mask=None):

    attn = compute_trans_mat(attn_)
    sizes = int(math.sqrt(attn.shape[0]))
    aggr_attn = aggregate_attn(attn,mask,size=sizes)
    agg_mask = F.interpolate(aggr_attn.unsqueeze(0).unsqueeze(0), size=size, mode="bilinear", align_corners=False)[0,0,...].cpu().numpy()

    sal_g = attn
    sal_g = sal_g + 0.1 
    sal_g[sal_g>=1]=1
    sal_g = sal_g ** (0.5)
    affinity_map = F.interpolate(sal_g.unsqueeze(0).unsqueeze(0), size=size, mode="bilinear", align_corners=False)[0,0,...].cpu().numpy()

    h = int(math.sqrt(attn.shape[-1]))
    ratio = size[0] // h
    idx = int(anchor[0]//ratio) * h + int(anchor[1]//ratio)

    idx_row = sal_g[idx,...]
    anchor_feat = idx_row.reshape(h,h).unsqueeze(0).unsqueeze(0)
    anchor_feat_resize = F.interpolate(anchor_feat, size=size, mode="bilinear", align_corners=False)[0,0,...].cpu().numpy()
    anchor_feat_resize -= anchor_feat_resize.min()
    anchor_feat_resize /= anchor_feat_resize.max()

    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    axs[0].imshow(inputs[0].permute(1,2,0).cpu().numpy())
    axs[0].scatter(anchor[1], anchor[0], color='orange', marker='*', s=200)  # 注意x, y的顺序
    axs[0].axis('off')  # 关闭坐标轴

    axs[1].imshow(agg_mask,cmap='viridis')
    axs[1].axis('off')  # 关闭坐标轴

    axs[2].imshow(anchor_feat_resize,cmap='viridis')
    axs[2].axis('off')  # 关闭坐标轴

    axs[3].imshow(affinity_map, cmap='YlGn')
    axs[3].axis('off')  # 关闭坐标轴

    plt.tight_layout()
    plt.savefig(os.path.join(anchor_dir, name + ".jpg"),dpi=200)
    plt.close()

    return 


def aggregate_attn(attn_maps, mask_, size=64):

    mask = F.interpolate(mask_.unsqueeze(1).type(torch.float32), size=(size,size), mode="nearest")[0]
    mask_flatten = mask.flatten(-2,-1)
    mask_flatten[mask_flatten>=1] = 1
    mask_flatten = mask_flatten.transpose(1,0)

    filtered = mask_flatten * attn_maps
    aggregate_attn = filtered.sum(0) / mask_flatten.sum(0)
    aggregate_attn = aggregate_attn.reshape(size,size)

    return aggregate_attn

def compute_trans_mat(attn_weight):
    trans_mat = attn_weight

    for _ in range(1):
        trans_mat = trans_mat / torch.sum(trans_mat, dim=0, keepdim=True)
        trans_mat = trans_mat / torch.sum(trans_mat, dim=1, keepdim=True)

    trans_mat = (trans_mat + trans_mat.transpose(1, 0)) / 2

    for _ in range(1):
        trans_mat = torch.matmul(trans_mat, trans_mat)

    return trans_mat

def find_foreground_center(mask):

    mask[mask==255] = 0
    mask = mask[0]

    unique_labels, counts = torch.unique(mask, return_counts=True)

    # 去掉背景（标签 0）
    foreground_labels = unique_labels[unique_labels != 0]
    foreground_counts = counts[unique_labels != 0]

    # 如果没有前景，返回None
    if foreground_counts.size(0) == 0:
        return tuple([10,10])

    # 找到拥有最多像素的前景标签
    max_label = foreground_labels[torch.argmax(foreground_counts)]

    # 找到该标签对应的所有坐标
    foreground_coords = torch.nonzero(mask == max_label)

    # 计算这些坐标的均值，得到重心
    center = foreground_coords.float().mean(dim=0)

    center = center.round().int()

    # 检查质心是否在前景内
    if mask[center[0], center[1]] == max_label:
        center = center  # 选择质心
    else:
        # 随机选择一个前景像素作为中心
        center = foreground_coords[torch.randint(0, foreground_coords.size(0), (1,))][0]
    
    # 返回重心的坐标（可以选择四舍五入取整，或保持浮点数形式）
    return center.tolist()