import shutil
import torch
import os
from PIL import Image

def generate_with_noise(ldm_stable, prompt, device, dtype, num_inference_steps=100, guidance_scale=7.5):
    # 创建随机噪声
    noise = torch.randn([1, 4, 64, 64]).to(device)
    noise = noise if dtype == torch.float32 else noise.half()
    
    # 生成图片
    with torch.no_grad():
        image = ldm_stable(
            prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            noise=noise
        ).images[0]
    
    return image

def setup_directories(base_path, categories):
    """创建输出目录结构"""
    # 如果输出根目录已存在，先删除它
    if os.path.exists(base_path):
        shutil.rmtree(base_path)
    
    # 创建根目录
    os.makedirs(base_path)
    
    # 为每个类别创建子目录
    for category in categories:
        category_path = os.path.join(base_path, category)
        os.makedirs(category_path)
    
    return base_path

def generate_images1(categories, ldm_stable, device, dtype,  num_images_per_category=5):
    """为每个类别生成图片并保存到相应目录"""
    # 设置输出根目录
    output_base = "output_generated"
    setup_directories(output_base, categories)
    

    
    # 设置随机种子
    g_cpu = torch.Generator()
    g_cpu.manual_seed(4307)
    
    # 为每个类别生成图片
    for category in categories:
        print(f"\nGenerating images for category: {category}")
        category_dir = os.path.join(output_base, category)
        
        for i in range(num_images_per_category):
            # 生成提示词
            prompt = f"a photograph of {category}"
            if category == "people":  # 特殊处理people类别
                prompt = "a photograph of a person"
            elif category == "buses":  # 特殊处理buses类别
                prompt = "a photograph of a bus"
                
            # 创建随机噪声
            noise = torch.randn([1, 4, 64, 64], generator=g_cpu).to(device)
            noise = noise if dtype == torch.float32 else noise.half()
            
            # 生成图片
            with torch.no_grad():
                image = ldm_stable(
                    [prompt],
                    num_inference_steps=100,
                    guidance_scale=10,
                    noise=noise
                ).images[0]
                
            # 保存图片
            output_path = os.path.join(category_dir, f"{category}_{i+1}.png")
            image.save(output_path)
            print(f"Saved {output_path}")



def generate_images(categories, ldm_stable, device, dtype, num_images_per_category=5):
    """为每个类别生成图片并保存到相应目录"""
    # 设置输出根目录
    output_base = "/data/PROJECTS/Diffusion_CLIP/sd_generates_img/output_generated/"
    setup_directories(output_base, categories)
    
    # 设置随机种子
    g_cpu = torch.Generator()
    g_cpu.manual_seed(4307)
    
    # 为每个类别生成图片
    for category in categories:
        print(f"\nGenerating images for category: {category}")
        category_dir = os.path.join(output_base, category)
        
        # 优化提示词
        prompt = (
            f"a realistic photograph of a fully visible, entire {category} "
            f"with natural colors, centered in the image, "
            f"with a clear and distinct background, high color contrast, "
            f"and the {category} should not occupy the entire image frame, "
            f"ensuring the {category} is completely visible without any part being cropped."
        )
        if category == "people":  # 特殊处理people类别
            prompt = (
                "a realistic photograph of a fully visible, entire person, "
                "centered in the image with natural colors, "
                "a clear and distinct background, and high contrast between the subject and background. "
                "The person should be fully visible, not cropped, and should not occupy the whole image frame."
            )
        elif category == "buses":  # 特殊处理buses类别
            prompt = (
                "a realistic photograph of a fully visible, entire bus, "
                "centered in the image with natural colors, "
                "a clear and distinct background, and high contrast between the bus and background. "
                "The bus should be fully visible, not cropped, and should not occupy the whole image frame."
            )

        for i in range(num_images_per_category):

            # 创建随机噪声
            noise = torch.randn([1, 4, 64, 64], generator=g_cpu).to(device)
            noise = noise if dtype == torch.float32 else noise.half()
            
            # 生成图片
            with torch.no_grad():
                image = ldm_stable(
                    [prompt],
                    num_inference_steps=45,
                    guidance_scale=10,
                    noise=noise,
                    width=384,
                    height=384
                ).images[0]
            
            # 转换为 512x512 的分辨率
            image_resized = image.resize((512, 512), Image.LANCZOS)
            
            # 保存图片
            output_path = os.path.join(category_dir, f"{category}_{i+1}.png")
            image_resized.save(output_path)
            print(f"Saved {output_path}")
