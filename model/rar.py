import torch
import torch.nn.functional as F
import math
from utils.camutils import cam_to_label_iseg
from functools import reduce
from utils import imutils


def recursive_refinement_with_diff_attn(cls_label, cam, diff_attn, mask_size,iter=10):
    """
    cam: 1 num_patches, num_cls
    diff_attn: num_heads, num_patches, num_patches
    """
    num_cls = cls_label.shape[-1]
    cls_lst = torch.where(cls_label)[0]
    N = cls_lst.shape[0]
    cross_att = cam
    or_s = int(math.sqrt(cross_att.shape[1]))
    tar_s = int(math.sqrt(diff_attn.shape[-1]))
    # cross_att = attr_maps_raw[:,:,cls_lst]
    cross_att  = F.interpolate(cross_att.permute(0,2,1).reshape(1,N,or_s,or_s), size=[tar_s,tar_s], mode='bilinear', align_corners=False).reshape(1,N,-1).permute(0,2,1)
    cross_attn = cross_att - cross_att.amin(dim=-2, keepdim=True)  # cross_att: 4096, 20
    cross_attn = cross_attn / cross_attn.sum(dim=-2, keepdim=True)  # 归一化
    
    trans_mat = diff_attn[[1,2]].mean(0).unsqueeze(0)
    # trans_mat = diff_attn.sum(0).unsqueeze(0)
    trans_mat /= torch.amax(trans_mat, dim=-2, keepdim=True)
    trans_mat += torch.where(trans_mat == 0, 0, 0.02 * (torch.log10(torch.e * trans_mat)))
    trans_mat = torch.clamp(trans_mat, min=0)

    trans_mat_p = trans_mat.clone()
    trans_mat_p /= trans_mat_p.sum(dim=-1, keepdim=True)

    for _ in range(iter):
        cross_attn = torch.bmm(trans_mat_p, cross_attn)
        # cross_attn = torch.where(cross_attn < cross_attn.amax(dim=-2, keepdim=True) * 0.1, 0, cross_attn)
        cross_attn -= cross_attn.amin(dim=-2, keepdim=True)
        cross_attn /= cross_attn.sum(dim=-2, keepdim=True)

    cross_att = cross_attn
    att_map = cross_att.unflatten(dim=-2, sizes=(tar_s, tar_s)).permute(0, 3, 1, 2)
    att_map = F.interpolate(att_map, size=[512,512], mode='bilinear', align_corners=False)
    att_map = att_map[0]
    att_map -= att_map.amin(dim=(-2, -1), keepdim=True)
    att_map /= att_map.amax(dim=(-2, -1), keepdim=True)

    final_attention_map = torch.zeros(num_cls, 512, 512).to(att_map.device)
    final_attention_map[cls_lst] += att_map

    final_attention_map = F.interpolate(final_attention_map[None], size=mask_size,
                            mode="bilinear", align_corners=False)[0]

    valid_lam, pred_mask = cam_to_label_iseg(final_attention_map[None].clone(),
                            cls_label=cls_label, bkg_thre=0.)
                            
    return valid_lam, pred_mask


def extract_diff_attn(img, diff_attn_extractor):

    img = imutils.denormalize_img(img) / 255.0
    diff_attn_extractor(img.to(img.device), "")
    diff_att_ori = torch.cat(
        [diff_attn_extractor.attention_maps_raw[idx][0] for idx in [-4]]).float()
        
    # diff_att = torch.cat(
    #     [diff_attn_extractor.attention_maps[-14][0]]).float()

    # diff_att /= torch.amax(diff_att, dim=-2, keepdim=True) + 1e-5
    # diff_att = torch.where(diff_att < 0.2, 0, diff_att)
    # diff_att /= diff_att.sum(dim=-1, keepdim=True) + 1e-5

    # diff_att = reduce(torch.matmul, diff_att[[1,5],...], torch.eye(diff_att.shape[-1], device=img.device))

    return diff_att_ori
                                    
                        