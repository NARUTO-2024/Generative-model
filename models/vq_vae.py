import torch
from models import BaseVAE
from torch import nn
from torch.nn import functional as F
from .types_ import *

class VectorQuantizer(nn.Module):
    """
    向量量化的实现
    参考文献：
    [1] https://github.com/deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py
    """
    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 beta: float = 0.25):
        super(VectorQuantizer, self).__init__()
        self.K = num_embeddings  # 嵌入数量
        self.D = embedding_dim   # 嵌入维度
        self.beta = beta         # 权重系数

        # 创建嵌入层并初始化权重
        self.embedding = nn.Embedding(self.K, self.D)  #创建一个嵌入矩阵，其中行数等于定义的嵌入数量，列数对应嵌入的维度。
        self.embedding.weight.data.uniform_(-1 / self.K, 1 / self.K)  #它将在给定范围内（从 −1/K 到 1/K）生成随机数并赋值给嵌入权重

    def forward(self, latents: Tensor) -> Tensor:  #表示这个方法的返回类型是一个 Tensor，是一个pytorch张量
        # 将输入的潜变量从 (B, D, H, W) 转置为 (B, H, W, D)
        latents = latents.permute(0, 2, 3, 1).contiguous()  #contiguous() 是一个方法，用于确保一个张量在内存中是连续存放的。因为重塑操作类似于（view）不允许向量不连续存放
        latents_shape = latents.shape
        flat_latents = latents.view(-1, self.D)  # 展平为 [BHW, D]

        # 计算潜变量与嵌入权重之间的L2距离
        dist = torch.sum(flat_latents ** 2, dim=1, keepdim=True) + \
               torch.sum(self.embedding.weight ** 2, dim=1) - \
               2 * torch.matmul(flat_latents, self.embedding.weight.t())  # [BHW x K]
        #dist=∑(xi^2)+∑(ek^2)−2(xi⋅ek)
        # 获取最近邻的编码索引
        encoding_inds = torch.argmin(dist, dim=1).unsqueeze(1)  # [BHW, 1]  找到最小距离的索引，并将结果进行形状调整

        # 将编码索引转换为独热编码形式
        device = latents.device
        encoding_one_hot = torch.zeros(encoding_inds.size(0), self.K, device=device)
        encoding_one_hot.scatter_(1, encoding_inds, 1)  # [BHW x K]

        # 量化潜变量
        quantized_latents = torch.matmul(encoding_one_hot, self.embedding.weight)  # [BHW, D]
        quantized_latents = quantized_latents.view(latents_shape)  # [B x H x W x D]

        # 计算VQ损失
        commitment_loss = F.mse_loss(quantized_latents.detach(), latents)  # 提交损失
        embedding_loss = F.mse_loss(quantized_latents, latents.detach())   # 嵌入损失

        vq_loss = commitment_loss * self.beta + embedding_loss  # 总VQ损失

        # 将残差加回潜变量
        quantized_latents = latents + (quantized_latents - latents).detach()

        return quantized_latents.permute(0, 3, 1, 2).contiguous(), vq_loss  # 返回量化的潜变量和VQ损失

class ResidualLayer(nn.Module):
    """残差层的实现"""

    def __init__(self,
                 in_channels: int,
                 out_channels: int):
        super(ResidualLayer, self).__init__()
        self.resblock = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)
        )

    def forward(self, input: Tensor) -> Tensor:
        return input + self.resblock(input)  # 输入与残差块的输出相加

class VQVAE(BaseVAE):
    """变分量化自编码器VQ-VAE的实现"""

    def __init__(self,
                 in_channels: int,
                 embedding_dim: int,
                 num_embeddings: int,
                 hidden_dims: List = None,
                 beta: float = 0.25,
                 img_size: int = 64,
                 **kwargs) -> None:
        super(VQVAE, self).__init__()

        self.embedding_dim = embedding_dim  # 嵌入维度
        self.num_embeddings = num_embeddings  # 嵌入数量
        self.img_size = img_size              # 图像大小
        self.beta = beta                      # 权重系数

        modules = []
        if hidden_dims is None:
            hidden_dims = [128, 256]

        # 构建编码器
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim, kernel_size=4, stride=2, padding=1),
                    nn.LeakyReLU()
                )
            )
            in_channels = h_dim

        modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU()
            )
        )

        for _ in range(6):
            modules.append(ResidualLayer(in_channels, in_channels))  # 添加6个残差层  in_channels = 3
        modules.append(nn.LeakyReLU())

        modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels, embedding_dim, kernel_size=1, stride=1),
                nn.LeakyReLU()
            )
        )

        self.encoder = nn.Sequential(*modules)  # 将编码器模块列表转为序列

        self.vq_layer = VectorQuantizer(num_embeddings, embedding_dim, self.beta)  # 向量量化层

        # 构建解码器
        modules = []
        modules.append(
            nn.Sequential(
                nn.Conv2d(embedding_dim, hidden_dims[-1], kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU()
            )
        )

        for _ in range(6):
            modules.append(ResidualLayer(hidden_dims[-1], hidden_dims[-1]))  # 添加6个残差层

        modules.append(nn.LeakyReLU())

        hidden_dims.reverse()  # 反转隐藏维度列表

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=4,
                                       stride=2,
                                       padding=1),
                    nn.LeakyReLU())
            )

        modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(hidden_dims[-1],
                                   out_channels=3,
                                   kernel_size=4,
                                   stride=2, padding=1),
                nn.Tanh()))  # 最后一层使用Tanh作为激活函数

        self.decoder = nn.Sequential(*modules)  # 将解码器模块列表转为序列

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        编码输入，通过编码器网络并返回潜变量编码。
        :param input: (Tensor) 指定输入张量 [N x C x H x W]
        :return: (Tensor) 潜变量编码的列表
        """
        result = self.encoder(input)  # 通过编码器获取编码
        return [result]

    def decode(self, z: Tensor) -> Tensor:
        """
        将给定的潜变量编码映射到图像空间。
        :param z: (Tensor) [B x D x H x W]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder(z)  # 解码潜变量
        return result

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        # 前向传播
        encoding = self.encode(input)[0]  # 编码输入
        quantized_inputs, vq_loss = self.vq_layer(encoding)  # 量化输入并计算VQ损失
        return [self.decode(quantized_inputs), input, vq_loss]  # 返回解码结果、输入和VQ损失

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        """
        计算损失函数
        :param args:
        :param kwargs:
        :return: 损失字典
        """
        recons = args[0]  # 重构的输出
        input = args[1]  # 原始输入
        vq_loss = args[2]  # VQ损失

        recons_loss = F.mse_loss(recons, input)  # 计算重构损失

        loss = recons_loss + vq_loss  # 总损失
        return {'loss': loss,
                'Reconstruction_Loss': recons_loss,
                'VQ_Loss': vq_loss}  # 返回损失字典

    def sample(self,
               num_samples: int,
               current_device: Union[int, str], **kwargs) -> Tensor:
        raise Warning('VQVAE采样器尚未实现。')   # 警告采样器未实现

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        给定输入图像x，返回重构的图像
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """
        return self.forward(x)[0]  # 返回通过前向传递获得的重构图像