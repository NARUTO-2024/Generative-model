import os
import math
import torch
from torch import optim
from models import BaseVAE  # 导入基础变分自编码器模型
from models.types_ import *  # 导入模型所需的类型定义
from utils import data_loader  # 导入数据加载工具
import pytorch_lightning as pl  # 导入 PyTorch Lightning
from torchvision import transforms  # 导入图像变换工具
import torchvision.utils as vutils  # 导入图像处理工具
from torchvision.datasets import CelebA  # 导入 CelebA 数据集
from torch.utils.data import DataLoader  # 导入数据加载器


class VAEXperiment(pl.LightningModule):  # 定义一个变分自编码器实验类，继承自 PyTorch Lightning 模块

    def __init__(self,
                 vae_model: BaseVAE,  # 传入变分自编码器模型
                 params: dict) -> None:  # 传入超参数字典
        super(VAEXperiment, self).__init__()  # 调用父类构造函数

        self.model = vae_model  # 保存传入的模型
        self.params = params  # 保存传入的参数
        self.curr_device = None  # 当前设备（GPU/CPU）
        self.hold_graph = False  # 是否保持计算图的标志，计算图是用来记录模型前向传播过程的基础数据结构，通常用于计算梯度。
        try:
            self.hold_graph = self.params['retain_first_backpass']  # 从参数中获取保持计算图的设置
        except:
            pass  # 如果没有该参数，不执行任何操作

    def forward(self, input: Tensor, **kwargs) -> Tensor:  # 前向传播
        return self.model(input, **kwargs)  # 调用模型的前向传播

    def training_step(self, batch, batch_idx, optimizer_idx=0):  # 定义训练步骤
        real_img, labels = batch  # 解包输入图像和标签
        self.curr_device = real_img.device  # 更新当前设备

        results = self.forward(real_img, labels=labels)  # 前向传播，得到结果
        train_loss = self.model.loss_function(*results,  # 计算损失
                                              M_N=self.params['kld_weight'],  # 使用 KL 散度权重
                                              optimizer_idx=optimizer_idx,  # 当前优化器索引
                                              batch_idx=batch_idx)  # 当前批次索引

        self.log_dict({key: val.item() for key, val in train_loss.items()}, sync_dist=True)  # 记录训练损失

        return train_loss['loss']  # 返回总损失

    def validation_step(self, batch, batch_idx, optimizer_idx=0):  # 定义验证步骤
        real_img, labels = batch  # 解包输入图像和标签
        self.curr_device = real_img.device  # 更新当前设备

        results = self.forward(real_img, labels=labels)  # 前向传播，得到结果
        val_loss = self.model.loss_function(*results,  # 计算验证损失
                                            M_N=1.0,  # 使用默认的 KL 散度权重
                                            optimizer_idx=optimizer_idx,  # 当前优化器索引
                                            batch_idx=batch_idx)  # 当前批次索引

        self.log_dict({f"val_{key}": val.item() for key, val in val_loss.items()}, sync_dist=True)  # 记录验证损失

    def on_validation_end(self) -> None:  # 在验证结束时调用
        self.sample_images()  # 进行图像采样

    def sample_images(self):  # 定义采样图像的方法
        # 从测试数据集中获取样本图像和标签
        test_input, test_label = next(iter(self.trainer.datamodule.test_dataloader()))  #iter是把数据加载器转换成迭代器，可以逐批获取数据
        test_input = test_input.to(self.curr_device)  # 将测试输入转移到当前设备
        test_label = test_label.to(self.curr_device)  # 将测试标签转移到当前设备

        recons = self.model.generate(test_input, labels=test_label)  # 生成重建图像
        vutils.save_image(recons.data,  # 保存重建图像
                          os.path.join(self.logger.log_dir,  # 指定文件路径
                                       "Reconstructions",
                                       f"recons_{self.logger.name}_Epoch_{self.current_epoch}.png"),  # 重建图像文件名
                          normalize=True,  # 归一化图像
                          nrow=12)  # 每行显示12张图像

        try:
            samples = self.model.sample(144,  # 生成144个样本
                                        self.curr_device,
                                        labels=test_label)  # 使用标签进行采样
            vutils.save_image(samples.cpu().data,  # 保存生成样本图像
                              os.path.join(self.logger.log_dir,  # 指定文件路径
                                           "Samples",
                                           f"{self.logger.name}_Epoch_{self.current_epoch}.png"),  # 样本图像文件名
                              normalize=True,  # 归一化图像
                              nrow=12)  # 每行显示12张图像
        except Warning:  # 捕获异常
            pass  # 忽略警告

    def configure_optimizers(self):  # 配置优化器和学习率调度器
        optims = []  # 初始化优化器列表
        scheds = []  # 初始化调度器列表

        # 使用 Adam 优化器
        optimizer = optim.Adam(self.model.parameters(),
                               lr=self.params['LR'],  # 从参数中获取学习率
                               weight_decay=self.params['weight_decay'])  # 权重衰减
        optims.append(optimizer)  # 将优化器添加到列表中

        # 检查是否需要二个优化器（用于对抗训练）
        try:
            if self.params['LR_2'] is not None:  # 如果存在第二个学习率
                optimizer2 = optim.Adam(getattr(self.model, self.params['submodel']).parameters(),
                                        lr=self.params['LR_2'])  # 创建第二个优化器
                optims.append(optimizer2)  # 将第二个优化器添加到列表中
        except:
            pass  # 忽略异常

        try:
            if self.params['scheduler_gamma'] is not None:  # 如果存在学习率调度器
                scheduler = optim.lr_scheduler.ExponentialLR(optims[0],
                                                             gamma=self.params['scheduler_gamma'])  # 创建调度器
                scheds.append(scheduler)  # 将调度器添加到列表中

                # 检查是否需要为第二个优化器添加调度器
                try:
                    if self.params['scheduler_gamma_2'] is not None:
                        scheduler2 = optim.lr_scheduler.ExponentialLR(optims[1],
                                                                      gamma=self.params[
                                                                          'scheduler_gamma_2'])  # 创建第二个调度器
                        scheds.append(scheduler2)  # 将第二个调度器添加到列表中
                except:
                    pass
                return optims, scheds  # 返回优化器和调度器
        except:
            return optims  # 仅返回优化器