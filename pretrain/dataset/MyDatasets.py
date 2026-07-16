# from PIL import Image
# from torch.utils.data import Dataset
# import pickle
# import os
# import torch
# from typing import Tuple, Optional, Union
# class MyDatasets(Dataset):
#     def __init__(self, data_path,transform = None, target_transform = None):
#         dataset_info = pickle.load(open(data_path, 'rb+'))
#         self.image_path = dataset_info.image_path
#         self.impress_saw = dataset_info.report
#         self.transform=transform
#     def __getitem__(self, index)-> Tuple[torch.Tensor, ...]:
#         impress_saw=self.impress_saw[index]
#         img = Image.open(self.image_path[index])
#         if self.transform is not None:
#             img = self.transform(img) 
#         return img,impress_saw
#     def __len__(self):
#         return len(self.image_path)


# 修改为适配包含多个子文件夹的图像目录

from PIL import Image
from torch.utils.data import Dataset
import os
import torch
from typing import Tuple

class MyDatasets(Dataset):
    def __init__(self, data_path, transform=None):
        """
        初始化数据集，遍历指定目录下的所有子文件夹，收集所有图像路径。

        Args:
            data_path (str): 包含多个子文件夹的根目录路径。
            transform (callable, optional): 数据增强和预处理的变换函数。
        """
        self.image_paths = []
        supported_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        
        # 遍历根目录下的所有子文件夹
        for root, _, files in os.walk(data_path):
            for file in files:
                if any(file.lower().endswith(ext) for ext in supported_extensions):
                    self.image_paths.append(os.path.join(root, file))
        
        self.transform = transform
        print(f"Found {len(self.image_paths)} images in {data_path}")

    def __getitem__(self, index) -> torch.Tensor:
        """
        根据索引获取图像，并应用预处理变换。

        Args:
            index (int): 图像的索引。

        Returns:
            torch.Tensor: 预处理后的图像张量。
        """
        img_path = self.image_paths[index]
        img = Image.open(img_path).convert('RGB')  # 确保图像是RGB格式
        if self.transform:
            img = self.transform(img)
            
        return img

    def __len__(self) -> int:
        """
        返回数据集的大小。

        Returns:
            int: 图像的总数量。
        """
        return len(self.image_paths)