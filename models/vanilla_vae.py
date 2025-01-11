import torch
from models import BaseVAE
from torch import nn
from torch.nn import functional as F
from .types_ import *

class VanillaVAE(BaseVAE):  # 定义一个名为 VanillaVAE 的类，继承自 BaseVAE

    def __init__(self,  # 初始化函数，设置模型参数
                 in_channels: int,  # 输入通道数
                 latent_dim: int,  # 潜在空间维度
                 hidden_dims: List = None,  # 隐藏层维度列表
                 **kwargs) -> None:  # 其他参数
        super(VanillaVAE, self).__init__()  # 调用父类构造函数

        self.latent_dim = latent_dim  # 设置潜在空间维度

        modules = []  # 初始化模块列表
        if hidden_dims is None:  # 如果没有提供隐藏层维度，则使用默认值
            hidden_dims = [32, 64, 128, 256, 512]

        # 构建编码器
        for h_dim in hidden_dims:  # 遍历每个隐藏层维度
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,  # 创建卷积层
                              kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),  # 添加批归一化层
                    nn.LeakyReLU())  # 添加LeakyReLU激活函数
            )
            in_channels = h_dim  # 更新输入通道数

        self.encoder = nn.Sequential(*modules)  # 将所有编码器模块组合成一个序列
        self.fc_mu = nn.Linear(hidden_dims[-1]*4, latent_dim)  # 第一个全连接层，输出均值。*4是指将原本的四维张量的B C H W迁移为一维
        self.fc_var = nn.Linear(hidden_dims[-1]*4, latent_dim)  # 第二个全连接层，输出方差

        # 构建解码器
        modules = []  # 初始化解码器模块列表

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 4)  # 解码器输入层

        hidden_dims.reverse()  # 反转隐藏层维度列表，以便为解码器构建

        for i in range(len(hidden_dims) - 1):  # 构建解码器，每层使用反卷积
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       output_padding=1),  # 反卷积层
                    nn.BatchNorm2d(hidden_dims[i + 1]),  # 批归一化层
                    nn.LeakyReLU())  # LeakyReLU激活函数
            )

        self.decoder = nn.Sequential(*modules)  # 将所有解码器模块组合成一个序列

        # 定义最终解码层
        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(hidden_dims[-1],
                                               hidden_dims[-1],
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               output_padding=1),  # 最后的反卷积层
                            nn.BatchNorm2d(hidden_dims[-1]),  # 批归一化层
                            nn.LeakyReLU(),  # LeakyReLU激活函数
                            nn.Conv2d(hidden_dims[-1], out_channels=3,  # 最后一层卷积，将输出通道设为3（RGB图像）
                                      kernel_size=3, padding=1),
                            nn.Tanh())  # Tanh激活函数，用于输出像素值在[-1, 1]之间

    def encode(self, input: Tensor) -> List[Tensor]:  # 编码函数
        """
        通过编码器网络对输入进行编码，并返回潜在编码
        :param input: (Tensor) 输入张量 [N x C x H x W]
        :return: (Tensor) 潜在编码列表
        """
        result = self.encoder(input)  # 通过编码器获取特征
        result = torch.flatten(result, start_dim=1)  # 将输出展平，从而得到[N, feature_dim]

        # 将结果分割为均值和方差组件
        mu = self.fc_mu(result)  # 计算潜在分布的均值
        log_var = self.fc_var(result)  # 计算潜在分布的对数方差

        return [mu, log_var]  # 返回均值和对数方差

    def decode(self, z: Tensor) -> Tensor:  # 解码函数
        """
        将给定的潜在编码映射到图像空间
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)  # 将潜在编码输入解码器
        result = result.view(-1, 512, 2, 2)  # 将结果重塑为适合解码器输入的形状
        result = self.decoder(result)  # 通过解码器获取生成图像
        result = self.final_layer(result)  # 通过最终解码层处理结果
        return result  # 返回生成的图像

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:  # 重参数化函数
        """
        使用重参数化技巧，从 N(mu, var) 采样
        参数来自 N(0,1)的分布
        :param mu: (Tensor) 潜在高斯分布的均值 [B x D]
        :param logvar: (Tensor) 潜在高斯分布的对数标准差 [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)  # 计算标准差
        eps = torch.randn_like(std)  # 从标准正态分布中采样噪声
        return eps * std + mu  # 根据重参数化公式返回采样结果

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:  # 前向传播方法
        mu, log_var = self.encode(input)  # 编码输入以获取均值和对数方差
        z = self.reparameterize(mu, log_var)  # 通过重参数化采样潜在编码
        return [self.decode(z), input, mu, log_var]  # 返回生成图像、输入、均值和对数方差

    def loss_function(self,  # 定义损失函数
                      *args,
                      **kwargs) -> dict:
        """
        计算 VAE 损失函数。
        KL(N(mu, sigma), N(0, 1)) = log(1/sigma) + (sigma^2 + mu^2) / 2 - 1/2
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]  # 生成的图像
        input = args[1]  # 原始输入
        mu = args[2]  # 均值
        log_var = args[3]  # 对数方差

        kld_weight = kwargs['M_N']  # 记录小批量样本对损失的影响
        recons_loss = F.mse_loss(recons, input)  # 计算重构损失，使用均方误差

        # 计算 KL 散度损失
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

        loss = recons_loss + kld_weight * kld_loss  # 总损失为重构损失和加权的 KL 散度损失
        return {'loss': loss, 'Reconstruction_Loss': recons_loss.detach(), 'KLD': -kld_loss.detach()}  # 返回损失字典

    def sample(self,  # 采样函数
               num_samples: int,
               current_device: int, **kwargs) -> Tensor:
        """
        从潜在空间采样并返回对应的图像空间映射
        :param num_samples: (Int) 样本数量
        :param current_device: (Int) 运行模型的设备
        :return: (Tensor)
        """
        z = torch.randn(num_samples, self.latent_dim)  # 从标准正态分布中采样潜在编码

        z = z.to(current_device)  # 将潜在编码移动到当前设备

        samples = self.decode(z)  # 解码潜在编码以生成样本
        return samples  # 返回生成的图像样本

    def generate(self, x: Tensor, **kwargs) -> Tensor:  # 生成函数
        """
        给定输入图像 x，返回重建的图像
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """
        return self.forward(x)[0]  # 调用前向传播，返回生成的重建图像