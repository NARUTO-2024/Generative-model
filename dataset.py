import os
import torch
from torch import Tensor
from pathlib import Path
from typing import List, Optional, Sequence, Union, Any, Callable
from torchvision.datasets.folder import default_loader
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import CelebA
import zipfile

# 自定义的基础数据集类
class MyDataset(Dataset):
    def __init__(self):
        pass  # 初始化逻辑尚未定义

    def __len__(self):
        pass  # 返回数据集长度的逻辑尚未定义

    def __getitem__(self, idx):
        pass  # 根据索引返回数据的逻辑尚未定义


# 自定义的CelebA数据集类，继承自CelebA
class MyCelebA(CelebA):
    """
    为解决PyTorch CelebA数据集类的问题而工作。

    下载和解压
    URL : https://drive.google.com/file/d/1m8-EBPgi5MRubrm6iQjafK2QMHDBMSfJ/view?usp=sharing
    """

    def _check_integrity(self) -> bool:
        return True  # 始终返回True，表示没有完整性检查


# 自定义的Oxford Pets数据集类，继承自Dataset
class OxfordPets(Dataset):
    """
    URL = https://www.robots.ox.ac.uk/~vgg/data/pets/
    """

    def __init__(self,
                 data_path: str,  #数据集的根目录路径
                 split: str,  #“train”或“val”
                 transform: Callable,  #数据预处理转换函数，通常用于图像增强
                 **kwargs):  #用于接收其他额外的参数
        # 设置数据目录
        self.data_dir = Path(data_path) / "OxfordPets"  #在根目录下找到该子目录
        self.transforms = transform

        # 加载图像路径，按名称排序
        imgs = sorted([f for f in self.data_dir.iterdir() if f.suffix == '.jpg'])  #遍历“OxfordPets”目录中的所有文件，只选择扩展名为“.jpg”的文件，并按名称进行排序

        # 根据选定的拆分（训练或验证）来选择图像
        self.imgs = imgs[:int(len(imgs) * 0.75)] if split == "train" else imgs[int(len(imgs) * 0.75):]  #若为“train”，选择前75%的图像；若为“val”，选择最后25%的图像

    def __len__(self):
        return len(self.imgs)  # 返回图像数量

    def __getitem__(self, idx):
        img = default_loader(self.imgs[idx])  # 加载指定索引的图像

        if self.transforms is not None:
            img = self.transforms(img)  # 应用变换

        return img, 0.0  # 返回图像和虚拟标签（避免出错）


# 自定义的VAEDataset类，继承自LightningDataModule
class VAEDataset(LightningDataModule):
    """
    PyTorch Lightning数据模块 

    参数:
        data_dir: 数据集的根目录。
        train_batch_size: 训练时使用的批大小。
        val_batch_size: 验证时使用的批大小。
        patch_size: 从原始图像中裁剪的尺寸。
        num_workers: 加载数据项时创建的并行工作数（参考PyTorch的DataLoader文档）。
        pin_memory: 是否将已准备的项加载到固定内存，这可以提高GPU性能。
    """

    def __init__(
            self,
            data_path: str,
            train_batch_size: int = 8,
            val_batch_size: int = 8,
            patch_size: Union[int, Sequence[int]] = (256, 256),   #是一个类型注解，代表可以接受整数和整数序列
            num_workers: int = 0,
            pin_memory: bool = False,
            **kwargs,
    ):
        super().__init__()  # 调用父类构造函数

        # 初始化参数
        self.data_dir = data_path
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.patch_size = patch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def setup(self, stage: Optional[str] = None) -> None:  #类型注解，可以是train，val或none
        # =========================  OxfordPets 数据集  =========================

        # 设置训练数据集的变换
        train_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),  #随机水平翻转图像
            transforms.CenterCrop(self.patch_size),  #使用中心裁剪方法从图像中裁剪出指定大小的图像块
            transforms.ToTensor(),  #将图像转换为张量格式，将图像的像素值（通常是0到255之间的整数）转换为0到1之间的浮点数
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  #归一化方式（将图像的像素值（通常是0到255之间的整数）转换为0到1之间的浮点数）为正太，前面数组为RGB三个通道的mean，后面数组为RGB三个通道的方差
        ])

        # 设置验证数据集的变换
        val_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.CenterCrop(self.patch_size),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

        # 创建训练和验证数据集对象
        self.train_dataset = OxfordPets(
            self.data_dir,  #数据根目录
            split='train',
            transform=train_transforms,
        )

        self.val_dataset = OxfordPets(
            self.data_dir,
            split='val',
            transform=val_transforms,
        )

        # =========================  CelebA 数据集  =========================

        # 训练数据集的变换
        train_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.CenterCrop(148),
            transforms.Resize(self.patch_size),
            transforms.ToTensor(),
        ])

        # 验证数据集的变换
        val_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.CenterCrop(148),
            transforms.Resize(self.patch_size),
            transforms.ToTensor(),
        ])

        self.train_dataset = MyCelebA(  # 使用自定义的CelebA数据集
            self.data_dir,
            split='train',
            transform=train_transforms,
            download=False,
        )

        self.val_dataset = MyCelebA(  # 使用自定义的CelebA数据集
            self.data_dir,
            split='test',
            transform=val_transforms,
            download=False,
        )
        # ===============================================================

    def train_dataloader(self) -> DataLoader:
        # 返回训练数据的DataLoader
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=True,  # 启用洗牌
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        # 返回验证数据的DataLoader
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,  # 不启用洗牌
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        # 返回测试数据的DataLoader
        return DataLoader(
            self.val_dataset,
            batch_size=144,  # 使用的测试批大小
            num_workers=self.num_workers,
            shuffle=True,  # 启用洗牌
            pin_memory=self.pin_memory,
        )