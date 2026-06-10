import os
import torch
import numpy as np
import random
import shutil
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# 预定义类别
embs = [
    "plane", "bicycle", "bird", "boat", "bottle", "buses", "car",
    "cat", "chair", "cow", "table", "dog", "horse", "motorbike",
    "people", "plant", "sheep", "sofa", "train", "monitor"
]

# 定义所有需要用到的路径
IMAGE_ROOT = "/data/PROJECTS/Diffusion_CLIP/sd_generates_img/generated_dir/img_root"
DATASET_ROOT = "/data/PROJECTS/Diffusion_CLIP/sd_generates_img/generated_dir/data_root"
IMAGE_OUTPUT = os.path.join(DATASET_ROOT, "images")
DIFFUSION_DATA_TXT = os.path.join(DATASET_ROOT, "diffusion_data.txt")
LABELS_NPY = os.path.join(DATASET_ROOT, "cls_labels_onehot.npy")

# 确保输出目录存在
os.makedirs(IMAGE_OUTPUT, exist_ok=True)

# 定义预处理方法
preprocess = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
])

def reorganize_images(image_root, output_root, K=50, dpi=200):
    """重新组织并编号图像"""
    print("Step 1: Reorganizing images...")
    img_count = 0
    for class_idx, class_name in enumerate(embs):
        class_path = os.path.join(image_root, class_name)
        if not os.path.exists(class_path):
            continue
        
        img_files = sorted(os.listdir(class_path))[:K]
        for img_name in tqdm(img_files, desc=f"Processing {class_name}", unit="image"):
            img_path = os.path.join(class_path, img_name)
            try:
                img = Image.open(img_path).convert("RGB")
                img_tensor = preprocess(img)
                img_pil = transforms.ToPILImage()(img_tensor)
                
                save_name = f"3001_{str(img_count).zfill(6)}.jpg"
                save_path = os.path.join(output_root, save_name)
                img_pil.save(save_path, dpi=(dpi, dpi))
                img_count += 1
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
    
    return img_count

def collect_image_names(image_folder, output_file):
    """收集图像文件名"""
    print("\nStep 2: Collecting image names...")
    image_files = sorted([f for f in os.listdir(image_folder) if f.endswith('.jpg')])
    
    with open(output_file, 'w') as f:
        for image_file in image_files:
            file_prefix = image_file.split('.')[0]
            f.write(f"{file_prefix}\n")
    
    return image_files

def create_label_dict(image_files, num_classes=20, images_per_class=50):
    """创建标签字典"""
    print("\nStep 3: Creating label dictionary...")
    labels_dict = {}
    
    for idx, class_name in enumerate(embs):
        # 为当前类别创建one-hot标签
        one_hot_label = np.zeros(num_classes, dtype=np.float32)
        one_hot_label[idx] = 1.0
        
        # 获取当前类别的图片
        class_images = image_files[idx * images_per_class: (idx + 1) * images_per_class]
        
        # 将图片名称与one-hot标签关联
        for img_name in class_images:
            img_name_no_ext = os.path.splitext(img_name)[0]
            labels_dict[img_name_no_ext] = one_hot_label
    
    return labels_dict

def shuffle_data(input_txt):
    """打乱数据顺序"""
    print("\nStep 4: Shuffling data...")
    with open(input_txt, 'r') as f:
        lines = f.readlines()
    
    # 打乱行顺序
    random.shuffle(lines)
    
    # 写回文件
    with open(input_txt, 'w') as f:
        f.writelines(lines)

def main():
    # 1. 重新组织图像
    total_images = reorganize_images(IMAGE_ROOT, IMAGE_OUTPUT)
    print(f"\nTotal images processed: {total_images}")
    
    # 2. 收集图像文件名
    image_files = collect_image_names(IMAGE_OUTPUT, DIFFUSION_DATA_TXT)
    print(f"Image names collected in: {DIFFUSION_DATA_TXT}")
    
    # 3. 创建标签字典并保存
    labels_dict = create_label_dict(image_files)
    np.save(LABELS_NPY, labels_dict)
    print(f"Label dictionary saved to: {LABELS_NPY}")
    
    # 4. 打乱数据顺序
    shuffle_data(DIFFUSION_DATA_TXT)
    print(f"Data shuffled in: {DIFFUSION_DATA_TXT}")
    
    print("\nAll processing completed successfully!")

if __name__ == "__main__":
    main()