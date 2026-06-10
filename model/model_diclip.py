
import torch
import torch.nn as nn
import torch.nn.functional as F
from .segformer_head import SegFormerHead
import numpy as np
import os
from torchvision.transforms import Compose, Normalize
from .decoder.TransDecoder import DecoderTransformer
import clip
from datasets.clip_text import new_class_names, BACKGROUND_CATEGORY,new_class_names_coco, BACKGROUND_CATEGORY_COCO
from diffusion_model.stable_diffusion import diffusion
from utils import imutils
from functools import reduce
from model.featurecluster import attn_kl_cluster
from timm.models.layers import trunc_normal_

class KV_Adapter(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., kv_cache=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        cache_key, cache_value = kv_cache
        idx = cache_key.shape[0]
        self.fc1 = nn.Linear(in_features, hidden_features)
        k_prompt = trunc_normal_(torch.zeros(hidden_features,in_features))
        k_prompt[:idx] = cache_key
        self.fc1.weight = nn.Parameter(k_prompt.clone())
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        v_prompt = trunc_normal_(torch.zeros(hidden_features,out_features))
        v_prompt[:idx] = cache_value
        self.fc2.weight = nn.Parameter(v_prompt.t().clone())
        self.drop = nn.Dropout(0.1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class DiCLIP_model(nn.Module):
    def __init__(self,  clip_model=None, embedding_dim=256, in_channels=512, adapter_size=312, \
                        num_classes=21, cache_file='./gpt4.0_cluster_a_photo_of4.json',\
                        img_size=320, mode='train', device='cuda', diff_attention_layers=[-4, -14]):

        super().__init__()
        self.device = device
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim

        self.encoder, _ = clip.load(clip_model, device=device)
        self.encoder.visual.reload_self_attn(layers=4, feat_size=img_size//16, mode=mode)
        self.encoder.eval()
        self.in_channels = in_channels

        self.decoder_fts_fuse = SegFormerHead(in_channels=self.in_channels,embedding_dim=self.embedding_dim,
                                              num_classes=self.num_classes, index=12)
        self.decoder = DecoderTransformer(width=self.embedding_dim, layers=3, heads=8, output_dim=self.num_classes)

        text_prompts = new_class_names+BACKGROUND_CATEGORY if num_classes <= 21 else new_class_names_coco+BACKGROUND_CATEGORY_COCO
        self.integral_text_features = clip.encode_text_with_prompt_ensemble(self.encoder, text_prompts, device, prompt_templates=['a clean origami {}.'])
        self.kv_cache = torch.load(cache_file)
        hidden, out = self.kv_cache[1].shape
        self.dynamic_adapter = KV_Adapter(in_features=512,hidden_features=adapter_size,out_features=out,kv_cache=self.kv_cache)

        self.diff_attn_extractor = diffusion(attention_layers_to_use=diff_attention_layers,
                                        model='v2.1', time_step=45, device=self.device,
                                        dtype=torch.float16 if "cuda" in str(self.device) else torch.float32)

    def get_param_groups(self):

        param_groups = [[], [], [], []]  # backbone; backbone_norm; cls_head; seg_head;
        for param in list(self.dynamic_adapter.parameters()):
            param_groups[2].append(param)
        for param in list(self.decoder.parameters()):
            param_groups[3].append(param)
        for param in list(self.decoder_fts_fuse.parameters()):
            param_groups[3].append(param)

        return param_groups

    def extract_diff_attn(self,img, size):
        img = imutils.denormalize_img(img) / 255.0
        self.diff_attn_extractor(img.half().to(self.device), "")

        diff_att_ori = torch.cat(
            [self.diff_attn_extractor.attention_maps_raw[idx] for idx in [-4]]).float()
        diff_att = torch.cat(
            [self.diff_attn_extractor.attention_maps[idx] for idx in [-14]]).float()

        diff_att /= torch.amax(diff_att, dim=-2, keepdim=True) + 1e-5
        diff_att = torch.where(diff_att < 0.2, 0, diff_att)
        diff_att /= diff_att.sum(dim=-1, keepdim=True) + 1e-5
        diff_atts = reduce(torch.matmul, diff_att[:,[1,2],...].permute(1,0,2,3), torch.eye(diff_att.shape[-1], device=diff_att.device))

        temp_cluster = []
        for diff_att in diff_atts:
            ## clustering
            cluster_att = attn_kl_cluster(diff_att,num_cluster=9,iter=5,size=size,)
            temp_cluster.append(cluster_att)
        cluster_att = torch.stack(temp_cluster, dim=0)
        ## contrain diff_attn with cluster
        temp = diff_atts
        for _ in range(3):
            temp = torch.matmul(cluster_att.float(), temp)
            temp /= (cluster_att.sum(-1, keepdim=True)+ 1e-6)
        diff_att_refined = temp.unsqueeze(1)

        return diff_att_refined, diff_att_ori

    def diff_knowledge_inject_cam(self,kv_cache,image_features):

        text_features, value = kv_cache[0].to(image_features.device), kv_cache[1].to(image_features.device)
        prob = image_features[:, :1, :] @ text_features.t()
        prob = (prob * 2).softmax(-1)
        w = prob / prob.mean(-1, keepdim=True)

        # element-wise multiplied features
        b, n_t, n_i, c = image_features.shape[0], text_features.shape[0], image_features.shape[1], image_features.shape[2]
        feats = image_features.reshape(b, n_i, 1, c) * text_features.reshape(1, 1, n_t, c)
        feats *= w.unsqueeze(-1)
        redundant_feats = feats.mean(2, keepdim=True) # along cls dim
        feats = feats - redundant_feats
        # sum the element-wise multiplied features as cosine similarity
        similarity = feats.sum(-1)
        similarity = F.relu(similarity)
        # calculate the intra-class difference
        text_features_21 = torch.cat([self.integral_text_features[(self.num_classes-1):].mean(0, keepdim=True),self.integral_text_features[:(self.num_classes-1)]])
        values = (torch.matmul(text_features, text_features_21.transpose(1,0)))
        values[value==0] = float('-inf')
        value = values.softmax(0)

        similarity = similarity[:,1:,:] @ value.unsqueeze(0)
        diff_maps = (similarity - similarity.min(1, keepdim=True)[0]) / (similarity.max(1, keepdim=True)[0] - similarity.min(1, keepdim=True)[0])

        # denoise with bkg cam
        fore_maps = diff_maps[:,:,1:]
        bkg_maps = diff_maps[:,:,0].unsqueeze(-1)
        fuse = fore_maps * (1-bkg_maps)
        fuse = F.relu(fuse)
        fuse_map = (fuse - fuse.min(1, keepdim=True)[0]) / (fuse.max(1, keepdim=True)[0] - fuse.min(1, keepdim=True)[0])

        return fuse_map

    def forward(self, img, return_dynamic=False, eval_seg=False):

        [diff_attn, diff_att_ori] = self.extract_diff_attn(img,size=int(img.shape[-1]//16)) if not eval_seg else [None, None]

        b, c, h, w = img.shape
        self.encoder.eval()
        image_features, attn_weights, all_feats = clip.generate_clip_fts(img, self.encoder, return_weights=True, ex_feats=diff_attn)
        attr_maps_raw = clip.clip_feature_surgery(image_features, self.integral_text_features)[:,1:,:self.num_classes-1]
        diff_maps = self.diff_knowledge_inject_cam(self.kv_cache, image_features)
        fuse =  0.5*diff_maps + attr_maps_raw
        diff_maps = (fuse - fuse.min(1, keepdim=True)[0]) / (fuse.max(1, keepdim=True)[0] - fuse.min(1, keepdim=True)[0])

        all_img_tokens =  all_feats[:, :, 1:, ...]
        all_img_tokens = all_img_tokens.permute(0, 1, 3, 2)
        all_img_tokens = all_img_tokens.reshape(12, b, all_img_tokens.size(-2), h//16, w //16) #(11, b, c, h, w)

        fts = self.decoder_fts_fuse(all_img_tokens)
        attn_fts = fts.clone()
        seg, seg_attn_weight_list = self.decoder(fts)
        if eval_seg:
            return seg

        dynamic_maps = self.dynamic_adapter(image_features[:,1:,:])
        if return_dynamic:
            fuse = 0.1*F.relu(dynamic_maps[:,:,1:]) + attr_maps_raw
            diff_maps = (fuse - fuse.min(1, keepdim=True)[0]) / (fuse.max(1, keepdim=True)[0] - fuse.min(1, keepdim=True)[0])

        f_b, f_c, f_h, f_w = fts.shape
        dynamic_maps_pred = dynamic_maps.permute(0,2,1).reshape(b,self.num_classes,f_h, f_w)
        
        f_b, f_c, f_h, f_w = attn_fts.shape
        attn_fts_flatten = attn_fts.reshape(f_b, f_c, f_h*f_w)
        attn_fts_flatten = F.normalize(attn_fts_flatten, dim=1)
        attn_pred = attn_fts_flatten.transpose(2, 1).bmm(attn_fts_flatten)
        attn_pred = (attn_pred - torch.mean(attn_pred) * 1.) * 3.0
        attn_pred = torch.sigmoid(attn_pred)

        return diff_maps.detach(), attn_weights, diff_att_ori, seg, attn_pred, dynamic_maps_pred