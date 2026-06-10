import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from voc_generation import *
from diffusers import StableDiffusionPipeline,  DDIMScheduler
g_cpu = torch.Generator(4307)    # 设置随机种子以保持一致性
dtype = torch.float16
diffusion_device = "cuda:0"
device = torch.device(diffusion_device) if torch.cuda.is_available() else torch.device('cpu')

# embs = [
#     "plane", "bicycle", "bird", "boat", "bottle", "buses", "car",
#     "cat", "chair", "cow", "table", "dog", "horse", "motorbike",
#     "people", "plant", "sheep", "sofa", "train", "monitor"
# ]


embs = [
    "boat"
]


model_key = "runwayml/stable-diffusion-v1-5"
# model_key = "stabilityai/stable-diffusion-2-1-base"
# model_key = '/home/jaye/.cache/huggingface/hub/models--stabilityai--stable-diffusion-2-1-base/snapshots/5ede9e4bf3e3fd1cb0ef2f7a3fff13ee514fdf06'
ldm_stable = StableDiffusionPipeline.from_pretrained(model_key, torch_dtype=dtype).to(device)
# 生成知识库图片
import time 

stime = time.time()
image = generate_images(
    categories=embs,
    ldm_stable=ldm_stable,
    device=device,
    dtype=dtype,
    num_images_per_category=50
)

etime = time.time()
elasped_time = etime - stime
print(f"elapsed time: {elasped_time:.6f} s for one class")
print(f"For VOC: {elasped_time*20:.6f} s is estimated")