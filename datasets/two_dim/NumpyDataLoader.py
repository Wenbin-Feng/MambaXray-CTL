import numpy as np
from batchgenerators.transforms import  MirrorTransform,Compose
from batchgenerators.transforms.crop_and_pad_transforms import CenterCropTransform, RandomCropTransform
from batchgenerators.transforms.spatial_transforms import ResizeTransform, SpatialTransform
from batchgenerators.transforms.utility_transforms import NumpyToTensor
from batchgenerators.transforms.color_transforms import BrightnessTransform, GammaTransform
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform, GaussianBlurTransform
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import os
import fnmatch

class SimCLRTransform:
    """SimCLR 数据增强"""
    def __init__(self, size=224):
        self.transform = Compose([
        BrightnessTransform(mu=1, sigma=0.1, p_per_sample=0.5),
        GammaTransform(gamma_range=(0.9, 1.1), p_per_sample=0.5),
        GaussianNoiseTransform(noise_variance=(0, 0.01), p_per_sample=0.5),
        SpatialTransform(
            patch_size=(224, 224),
            random_crop=False,
            do_elastic_deform=True,
            alpha=(0., 50.),
            sigma=(40., 60.),
            do_rotation=True,
            p_rot_per_sample=0.3,
            angle_z=(-10 * np.pi / 180, 10 * np.pi / 180),
            scale=(0.9, 1.1),
            p_scale_per_sample=0.3,
            border_mode_data="nearest",
            border_mode_seg="nearest"
        ),
        NumpyToTensor(),
    ])
    
         
    def __call__(self, x):
        # 使用关键字参数传递数据
        x = x[None, ...]  # 增加批次维度，形状变为 (1, c, h, w)
        transformed_xi = self.transform(data=x)
        xi = transformed_xi['data'][0].cpu().numpy()
        
        transformed_xj = self.transform(data=x)
        xj = transformed_xj['data'][0].cpu().numpy()
        
            # 转回 uint8
        xi = np.clip(xi, 0, 255).astype(np.uint8)
        xj = np.clip(xj, 0, 255).astype(np.uint8)
           # 调整通道顺序 [C, H, W] -> [H, W, C]
        
        return xi, xj
    
def load_dataset(base_dir, pattern='*.png'):
    """加载数据集文件路径"""
    fls = []
    for root, dirs, files in os.walk(base_dir):
        for filename in sorted(fnmatch.filter(files, pattern)):
            img_file = os.path.join(root, filename)
            fls.append(img_file)
    return fls

class ImageDataset(Dataset):
    def __init__(self, base_dir, transform=None, pattern='*.png'):
        """
        Args:
            base_dir (str): 数据集根目录
            transform (callable, optional): 数据增强函数
            pattern (str, optional): 图像文件匹配模式
        """
        self.image_paths = load_dataset(base_dir, pattern)
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 加载图像
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        image = np.array(image).astype(np.float32).transpose(2, 0, 1) / 255.0
        if self.transform:
            xi, xj = self.transform(image)
            return xi, xj
        else:
            return image, image