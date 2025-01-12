import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from main import instantiate_from_config

# 导入模型组件
from taming.modules.diffusionmodules.model import Encoder, Decoder
from taming.modules.vqvae.quantize import VectorQuantizer2 as VectorQuantizer
from taming.modules.vqvae.quantize import GumbelQuantize
from taming.modules.vqvae.quantize import EMAVectorQuantizer


# 定义 VQModel 类，继承 PyTorch Lightning 的 LightningModule
class VQModel(pl.LightningModule):
    def __init__(self,
                 ddconfig,  # 数据配置
                 lossconfig,  # 损失函数配置
                 n_embed,  # 嵌入（向量量化）总数
                 embed_dim,  # 嵌入维度
                 ckpt_path=None,  # 检查点路径
                 ignore_keys=[],  # 忽略的键，防止加载时出错
                 image_key="image",  # 图像字典中的键
                 colorize_nlabels=None,  # 颜色化标签数（用于分割模型等）
                 monitor=None,  # 监控指标
                 remap=None,  # 重映射
                 sane_index_shape=False,  # 控制向量量化返回索引的形状
                 ):
        super().__init__()
        # 初始化模型的各个组件
        self.image_key = image_key  # 图像关键字
        self.encoder = Encoder(**ddconfig)  # 编码器   **ddconfig是一个字典，包含了编码器配置的参数。初始化了一个编码器对象
        self.decoder = Decoder(**ddconfig)  # 解码器
        self.loss = instantiate_from_config(lossconfig)  # 初始化损失函数
        # 初始化向量量化模块
        self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25,
                                        remap=remap, sane_index_shape=sane_index_shape)
        # 初始化卷积层
        self.quant_conv = torch.nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)  # 量化的卷积层
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)  # 后处理卷积层

        # 从检查点初始化
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

        # 如果需要颜色化，会注册颜色化参数
        if colorize_nlabels is not None:
            assert type(colorize_nlabels) == int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))

        if monitor is not None:
            self.monitor = monitor  # 设置监控指标

    def init_from_ckpt(self, path, ignore_keys=list()):  #init_from_ckpt方法（该方法需要在类的定义中实现）将从给定的检查点路径加载模型的状态字典，并可能使用ignore_keys参数来忽略某些键，这些键可能对应于不应该从检查点加载的模型参数或模块。
        # 从检查点加载模型权重
        sd = torch.load(path, map_location="cpu")["state_dict"]  # 加载状态字典
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):  # 如果当前键在忽略列表中，则删除
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)  # 加载状态字典
        print(f"Restored from {path}")

    def encode(self, x):
        # 编码输入图像
        h = self.encoder(x)  # 通过编码器
        h = self.quant_conv(h)  # 量化卷积
        quant, emb_loss, info = self.quantize(h)  # 量化及计算嵌入损失  quant可以用于解码器生成图像，而emb_loss可以用于优化编码器和解码器的参数
        return quant, emb_loss, info  # 返回量化结果及嵌入损失

    def decode(self, quant):
        # 解码量化的表示
        quant = self.post_quant_conv(quant)  # 后处理卷积
        dec = self.decoder(quant)  # 通过解码器
        return dec  # 返回解码结果

    def decode_code(self, code_b):
        # 根据编码的索引解码（反向编码）
        quant_b = self.quantize.embed_code(code_b)  # 嵌入编码
        dec = self.decode(quant_b)  # 解码嵌入
        return dec

    def forward(self, input):
        # 前向传播，包括编码和解码过程
        quant, diff, _ = self.encode(input)  # 编码过程
        dec = self.decode(quant)  # 解码过程
        return dec, diff

    def get_input(self, batch, k):
        # 获取模型输入
        x = batch[k]  # 从batch中提取图像
        if len(x.shape) == 3:
            x = x[..., None]  # 如果只有3个维度，添加一个维度
        # 调整图像维度为（N, C, H, W）
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
        return x.float()  # 返回转为浮点数的张量

    def training_step(self, batch, batch_idx, optimizer_idx):
        # 训练步骤
        x = self.get_input(batch, self.image_key)  # 获取输入
        xrec, qloss = self(x)  # 前向计算，获取重建图像和量化损失

        if optimizer_idx == 0:
            # autoencode
            # 计算自编码损失
            aeloss, log_dict_ae = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")

            # 记录损失
            self.log("train/aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return aeloss  # 返回自编码损失

        if optimizer_idx == 1:
            # 在此处实现判别器的训练步骤（不过这个类其实没有使用判别器）
            discloss, log_dict_disc = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
                                                last_layer=self.get_last_layer(), split="train")
            self.log("train/discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return discloss  # 返回判别器损失

    def validation_step(self, batch, batch_idx):
        # 验证步骤
        x = self.get_input(batch, self.image_key)  # 获取验证输入
        xrec, qloss = self(x)  # 前向传播
        # 计算损失
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="val")  #并且使用参数 0 表示在损失计算中使用自编码器相关的损失。val代表这是在验证过程中，不会更新权重

        discloss, log_dict_disc = self.loss(qloss, x, xrec, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")
        rec_loss = log_dict_ae["val/rec_loss"]  # 记录重建损失
        self.log("val/rec_loss", rec_loss,
                 prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("val/aeloss", aeloss,
                 prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log_dict(log_dict_ae)  # 日志记录自编码损失
        self.log_dict(log_dict_disc)  # 日志记录判别器损失
        return self.log_dict

    def configure_optimizers(self):
        # 配置优化器
        lr = self.learning_rate  # 获取学习率
        opt_ae = torch.optim.Adam(list(self.encoder.parameters()) +
                                  list(self.decoder.parameters()) +
                                  list(self.quantize.parameters()) +
                                  list(self.quant_conv.parameters()) +
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr, betas=(0.5, 0.9))  # 自编码器优化器
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))  # 判别器优化器
        return [opt_ae, opt_disc], []  # 返回优化器列表

    def get_last_layer(self):
        # 获取最后一层的权重
        return self.decoder.conv_out.weight

    def log_images(self, batch, **kwargs):
        # 日志记录输入和重建图像
        log = dict()
        x = self.get_input(batch, self.image_key)  # 获取输入
        x = x.to(self.device)  # 移动到正确的设备
        xrec, _ = self(x)
        if x.shape[1] > 3:
            # 颜色化
            assert xrec.shape[1] > 3
            x = self.to_rgb(x)
            xrec = self.to_rgb(xrec)
        log["inputs"] = x  # 记录输入图像
        log["reconstructions"] = xrec  # 记录重建图像
        return log

    def to_rgb(self, x):
        # 颜色化函数
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)  # 使用颜色化权重进行卷积
        x = 2. * (x - x.min()) / (x.max() - x.min()) - 1.  # 将张量值归一化到[-1,1]范围
        return x


# 定义 VQSegmentationModel 类
class VQSegmentationModel(VQModel):
    def __init__(self, n_labels, *args, **kwargs):
        super().__init__(*args, **kwargs)  # 调用父类初始化
        self.register_buffer("colorize", torch.randn(3, n_labels, 1, 1))  # 注册颜色化参数

    def configure_optimizers(self):
        # 配置优化器
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.encoder.parameters()) +
                                  list(self.decoder.parameters()) +
                                  list(self.quantize.parameters()) +
                                  list(self.quant_conv.parameters()) +
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr, betas=(0.5, 0.9))  # 自编码器优化器
        return opt_ae  # 返回自编码器优化器

    def training_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)  # 获取输入
        xrec, qloss = self(x)  # 前向传播
        # 计算自编码损失
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, split="train")
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        return aeloss  # 返回损失

    def validation_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)  # 获取输入
        xrec, qloss = self(x)  # 前向传播
        # 计算损失
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, split="val")
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        total_loss = log_dict_ae["val/total_loss"]  # 获取总损失
        self.log("val/total_loss", total_loss,
                 prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        return aeloss  # 返回损失

    @torch.no_grad()
    def log_images(self, batch, **kwargs):
        # 日志记录输入和重建图像
        log = dict()
        x = self.get_input(batch, self.image_key)  # 获取输入
        x = x.to(self.device)  # 移动到正确的设备
        xrec, _ = self(x)  # 前向传播
        if x.shape[1] > 3:
            # 颜色化逻辑
            assert xrec.shape[1] > 3
            xrec = torch.argmax(xrec, dim=1, keepdim=True)  # 获取最大概率对应的类别
            xrec = F.one_hot(xrec, num_classes=x.shape[1])  # 转换为 one-hot 编码
            xrec = xrec.squeeze(1).permute(0, 3, 1, 2).float()  # 转换维度
            x = self.to_rgb(x)  # 颜色化输入
            xrec = self.to_rgb(xrec)  # 颜色化重建
        log["inputs"] = x  # 记录输入
        log["reconstructions"] = xrec  # 记录重建图像
        return log


# 定义 VQNoDiscModel 类，作为 VQModel 的简化版本，没有判别器
class VQNoDiscModel(VQModel):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 n_embed,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None
                 ):
        super().__init__(ddconfig=ddconfig, lossconfig=lossconfig, n_embed=n_embed, embed_dim=embed_dim,
                         ckpt_path=ckpt_path, ignore_keys=ignore_keys, image_key=image_key,
                         colorize_nlabels=colorize_nlabels)

    def training_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)  # 获取输入
        xrec, qloss = self(x)  # 前向传播
        # 计算自编码损失
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, self.global_step, split="train")
        output = pl.TrainResult(minimize=aeloss)  # 返回最小化的损失
        output.log("train/aeloss", aeloss,
                   prog_bar=True, logger=True, on_step=True, on_epoch=True)
        output.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        return output

    def validation_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)  # 获取输入
        xrec, qloss = self(x)  # 前向传播
        # 计算损失
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, self.global_step, split="val")
        rec_loss = log_dict_ae["val/rec_loss"]  # 获取重建损失
        output = pl.EvalResult(checkpoint_on=rec_loss)  # 返回重建损失
        output.log("val/rec_loss", rec_loss,
                   prog_bar=True, logger=True, on_step=True, on_epoch=True)
        output.log("val/aeloss", aeloss,
                   prog_bar=True, logger=True, on_step=True, on_epoch=True)
        output.log_dict(log_dict_ae)  # 日志记录自编码损失

        return output  # 返回损失

    def configure_optimizers(self):
        # 配置优化器
        optimizer = torch.optim.Adam(list(self.encoder.parameters()) +
                                     list(self.decoder.parameters()) +
                                     list(self.quantize.parameters()) +
                                     list(self.quant_conv.parameters()) +
                                     list(self.post_quant_conv.parameters()),
                                     lr=self.learning_rate, betas=(0.5, 0.9))  # 自编码器优化器
        return optimizer  # 返回优化器


# 定义 GumbelVQ 类，继承 VQModel
class GumbelVQ(VQModel):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 n_embed,
                 embed_dim,
                 temperature_scheduler_config,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 kl_weight=1e-8,
                 remap=None,
                 ):
        z_channels = ddconfig["z_channels"]  # 记录 z 通道数
        super().__init__(ddconfig,
                         lossconfig,
                         n_embed,
                         embed_dim,
                         ckpt_path=None,
                         ignore_keys=ignore_keys,
                         image_key=image_key,
                         colorize_nlabels=colorize_nlabels,
                         monitor=monitor,
                         )

        self.loss.n_classes = n_embed  # 设置损失类数
        self.vocab_size = n_embed  # 嵌入字典规模

        # 初始化 Gumbel 量化
        self.quantize = GumbelQuantize(z_channels, embed_dim,
                                       n_embed=n_embed,
                                       kl_weight=kl_weight, temp_init=1.0,
                                       remap=remap)  #Gumbel-Softmax 方法中的温度控制了连续分布向离散分布的转换。温度越高，采样分布越接近均匀分布；温度越低，分布越接近确定性的选择（即选择最高概率的嵌入）

        self.temperature_scheduler = instantiate_from_config(temperature_scheduler_config)  # 初始化温度调节器

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)  # 从检查点加载

    def temperature_scheduling(self):
        # 更新量化的温度值
        self.quantize.temperature = self.temperature_scheduler(self.global_step)

    def encode_to_prequant(self, x):
        # 编码到量化前的层
        h = self.encoder(x)
        h = self.quant_conv(h)
        return h

    def decode_code(self, code_b):
        # 解码编码（未实现）
        raise NotImplementedError

    def training_step(self, batch, batch_idx, optimizer_idx):
        self.temperature_scheduling()  # 更新温度
        x = self.get_input(batch, self.image_key)  # 获取输入
        xrec, qloss = self(x)  # 前向传播

        if optimizer_idx == 0:
            # autoencode
            # 计算自编码损失
            aeloss, log_dict_ae = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")

            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            self.log("temperature", self.quantize.temperature, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return aeloss  # 返回自编码损失

        if optimizer_idx == 1:
            # 在此处实现判别器的训练步骤
            discloss, log_dict_disc = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
                                                last_layer=self.get_last_layer(), split="train")
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return discloss  # 返回判别器损失

    def validation_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)  # 获取输入
        xrec, qloss = self(x, return_pred_indices=True)  # 前向传播
        # 计算损失
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="val")

        discloss, log_dict_disc = self.loss(qloss, x, xrec, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")
        rec_loss = log_dict_ae["val/rec_loss"]  # 记录重建损失
        self.log("val/rec_loss", rec_loss,
                 prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/aeloss", aeloss,
                 prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log_dict(log_dict_ae)  # 日志记录自编码损失
        self.log_dict(log_dict_disc)  # 日志记录判别器损失
        return self.log_dict

    def log_images(self, batch, **kwargs):
        # 日志记录输入和重建图像
        log = dict()
        x = self.get_input(batch, self.image_key)  # 获取输入
        x = x.to(self.device)  # 移动到正确的设备
        # 编码
        h = self.encoder(x)  # 编码
        h = self.quant_conv(h)  # 量化
        quant, _, _ = self.quantize(h)  # 量化
        # 解码
        x_rec = self.decode(quant)  # 解码
        log["inputs"] = x  # 记录输入
        log["reconstructions"] = x_rec  # 记录重建图像
        return log  # 返回日志


# 定义 EMAVQ 类，作为 VQModel 的一种变体
class EMAVQ(VQModel):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 n_embed,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 remap=None,
                 sane_index_shape=False):  # 控制向量量化返回索引的形状
        super().__init__(ddconfig,
                         lossconfig,
                         n_embed,
                         embed_dim,
                         ckpt_path=None,
                         ignore_keys=ignore_keys,
                         image_key=image_key,
                         colorize_nlabels=colorize_nlabels,
                         monitor=monitor,
                         )
        # 初始化 EMA 向量量化器
        self.quantize = EMAVectorQuantizer(n_embed=n_embed,
                                           embedding_dim=embed_dim,
                                           beta=0.25,
                                           remap=remap)

    def configure_optimizers(self):
        # 配置优化器
        lr = self.learning_rate
        # 移除 self.quantize 从参数列表，因为它通过 EMA 更新
        opt_ae = torch.optim.Adam(list(self.encoder.parameters()) +
                                  list(self.decoder.parameters()) +
                                  list(self.quant_conv.parameters()) +
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr, betas=(0.5, 0.9))  # 自编码器优化器
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))  # 判别器优化器
        return [opt_ae, opt_disc], []  # 返回优化器列表