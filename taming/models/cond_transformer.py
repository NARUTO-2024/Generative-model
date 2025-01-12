import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from main import instantiate_from_config
from taming.modules.util import SOSProvider


def disabled_train(self, mode=True):
    """重写model.train方法，确保训练/评估模式不再更改。"""
    return self  # 禁用更改模式，以避免影响训练行为


class Net2NetTransformer(pl.LightningModule):
    def __init__(self,
                 transformer_config,  # 转换器模型的配置
                 first_stage_config,  # 第一阶段模型（图像）的配置
                 cond_stage_config,  # 条件阶段模型（例如，深度图）的配置
                 permuter_config=None,  # 用于排列的配置，默认为身份映射
                 ckpt_path=None,  # 加载检查点的路径
                 ignore_keys=[],  # 加载权重时要忽略的键
                 first_stage_key="image",  # 第一阶段数据的键
                 cond_stage_key="depth",  # 条件阶段数据的键
                 downsample_cond_size=-1,  # 条件下采样的大小
                 pkeep=1.0,  # 在训练过程中保留原始索引的概率
                 sos_token=0,  # 开始序列标记，用于无条件生成
                 unconditional=False,  # 指示无条件训练的标志
                 ):
        super().__init__()
        self.be_unconditional = unconditional
        self.sos_token = sos_token
        self.first_stage_key = first_stage_key
        self.cond_stage_key = cond_stage_key

        # 根据配置初始化第一阶段模型
        self.init_first_stage_from_ckpt(first_stage_config)
        # 根据配置初始化条件阶段模型
        self.init_cond_stage_from_ckpt(cond_stage_config)

        # 如果没有提供排列器配置，则使用身份映射
        if permuter_config is None:
            permuter_config = {"target": "taming.modules.transformer.permuter.Identity"}

        # 实例化排列器和变换器模型
        self.permuter = instantiate_from_config(config=permuter_config)
        self.transformer = instantiate_from_config(config=transformer_config)

        # 如果提供了检查点路径，则加载模型权重
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

        self.downsample_cond_size = downsample_cond_size  # 条件下采样的大小
        self.pkeep = pkeep  # 在训练中保留的概率

    def init_from_ckpt(self, path, ignore_keys=list()):
        """从检查点加载模型权重，同时忽略某些键。"""
        sd = torch.load(path, map_location="cpu")["state_dict"]  # 加载状态字典
        for k in sd.keys():
            for ik in ignore_keys:
                if k.startswith(ik):
                    self.print("Deleting key {} from state_dict.".format(k))
                    del sd[k]  # 忽略指定的键
        self.load_state_dict(sd, strict=False)  # 加载参数
        print(f"Restored from {path}")

    def init_first_stage_from_ckpt(self, config):
        """从配置初始化第一阶段模型。"""
        model = instantiate_from_config(config)  # 从配置实例化模型
        model = model.eval()  # 设置为评估模式
        model.train = disabled_train  # 禁用模式更改
        self.first_stage_model = model  # 将模型分配给属性

    def init_cond_stage_from_ckpt(self, config):
        """从配置初始化条件阶段模型。"""
        if config == "__is_first_stage__":  # 如果指定为第一阶段
            print("Using first stage also as cond stage.")
            self.cond_stage_model = self.first_stage_model  # 使用第一阶段模型
        elif config == "__is_unconditional__" or self.be_unconditional:  # 无条件训练
            print(f"Using no cond stage. Assuming the training is intended to be unconditional. "
                  f"Prepending {self.sos_token} as a sos token.")
            self.be_unconditional = True
            self.cond_stage_key = self.first_stage_key  # 条件阶段键为第一阶段键
            self.cond_stage_model = SOSProvider(self.sos_token)  # 创建SOS提供者
        else:
            model = instantiate_from_config(config)  # 实例化条件模型
            model = model.eval()  # 设置为评估模式
            model.train = disabled_train  # 禁用模式更改
            self.cond_stage_model = model  # 将模型分配给属性

    def forward(self, x, c):
        # 一步产生logits（得分）
        _, z_indices = self.encode_to_z(x)  # 编码图像到z空间
        _, c_indices = self.encode_to_c(c)  # 编码条件到c空间  c代表隐含信息，即对图片z的某种约束

        if self.training and self.pkeep < 1.0:  # 若在训练且保持概率小于1.0
            mask = torch.bernoulli(self.pkeep * torch.ones(z_indices.shape,
                                                           device=z_indices.device))  # 创建掩码   生成一个与 z_indices 形状相同的随机掩码，掩码的值为0和1，其中1的概率由 pkeep 控制。
            mask = mask.round().to(dtype=torch.int64)  # 将掩码转换为整数
            r_indices = torch.randint_like(z_indices, self.transformer.config.vocab_size)  # 随机生成替代索引
            a_indices = mask * z_indices + (1 - mask) * r_indices  # 根据掩码选择索引
        else:
            a_indices = z_indices  # 不改变索引

        cz_indices = torch.cat((c_indices, a_indices), dim=1)  # 连接条件索引和样本索引

        # 目标包含所有序列元素（无需处理第一个元素，因为我们在条件化）
        target = z_indices
        # 进行预测
        logits, _ = self.transformer(cz_indices[:, :-1])  # 将cz_indices传入变换器
        # 截断条件输出 - 输出i对应p(z_i | z_{<i}, c)
        logits = logits[:, c_indices.shape[1] - 1:]

        return logits, target  # 返回logits和目标

    def top_k_logits(self, logits, k):
        v, ix = torch.topk(logits, k)  # 获取前k个值和其索引
        out = logits.clone()  # 克隆logits
        out[out < v[..., [-1]]] = -float('Inf')  # 将小于k值的logits设为负无穷
        return out  # 返回处理后的logits

    @torch.no_grad()
    def sample(self, x, c, steps, temperature=1.0, sample=False, top_k=None,
               callback=lambda k: None):
        x = torch.cat((c, x), dim=1)  # 将条件c和输入x拼接
        block_size = self.transformer.get_block_size()  # 获取变换器的块大小
        assert not self.transformer.training  # 确保模式为评估模式
        if self.pkeep <= 0.0:  # 如果保持概率为0
            # 一次性计算就可以，因为输入完全是噪声
            assert len(x.shape) == 2
            noise_shape = (x.shape[0], steps - 1)
            # noise = torch.randint(self.transformer.config.vocab_size, noise_shape).to(x)
            noise = c.clone()[:, x.shape[1] - c.shape[1]:-1]  # 复制条件张量作为噪声
            x = torch.cat((x, noise), dim=1)  # 将噪声拼接到x
            logits, _ = self.transformer(x)  # 将x传入变换器
            # 当前所有logits，按温度缩放
            logits = logits / temperature
            # 可选地裁剪概率到前k个选项
            if top_k is not None:
                logits = self.top_k_logits(logits, top_k)
            # 应用softmax以转换为概率
            probs = F.softmax(logits, dim=-1)
            # 从分布中进行采样或取最可能的
            if sample:
                shape = probs.shape
                probs = probs.reshape(shape[0] * shape[1], shape[2])  # 重新调整维度
                ix = torch.multinomial(probs, num_samples=1)  # 采样
                probs = probs.reshape(shape[0], shape[1], shape[2])  # 恢复维度
                ix = ix.reshape(shape[0], shape[1])  # 恢复维度
            else:
                _, ix = torch.topk(probs, k=1, dim=-1)  # 取最大概率的索引
            # 截断条件
            x = ix[:, c.shape[1] - 1:]  # 取出生成的部分
        else:  # 如果保持概率大于0
            for k in range(steps):  # 逐步生成
                callback(k)  # 运行回调函数
                assert x.size(1) <= block_size  # 确保模型能看到条件
                x_cond = x if x.size(1) <= block_size else x[:, -block_size:]  # 如果需要，裁剪上下文
                logits, _ = self.transformer(x_cond)  # 将x_cond传入变换器
                logits = logits[:, -1, :] / temperature  # 获取最后一步的logits并按温度缩放
                # 可选地裁剪概率到前k个选项
                if top_k is not None:
                    logits = self.top_k_logits(logits, top_k)
                # 应用softmax以转换为概率
                probs = F.softmax(logits, dim=-1)
                # 从分布中进行采样或取最可能的
                if sample:
                    ix = torch.multinomial(probs, num_samples=1)  # 采样
                else:
                    _, ix = torch.topk(probs, k=1, dim=-1)  # 取最大概率的索引
                # 将索引添加到序列中并继续
                x = torch.cat((x, ix), dim=1)
            # 截断条件
            x = x[:, c.shape[1]:]  # 只保留生成的部分
        return x  # 返回生成的序列

    @torch.no_grad()
    def encode_to_z(self, x):
        """将输入x编码到z空间。"""
        quant_z, _, info = self.first_stage_model.encode(x)  # 编码到z
        indices = info[2].view(quant_z.shape[0], -1)  # 从编码信息中获取索引
        indices = self.permuter(indices)  # 应用排列器
        return quant_z, indices  # 返回量化后的z和索引

    @torch.no_grad()
    def encode_to_c(self, c):
        """将输入c编码到条件空间。"""
        if self.downsample_cond_size > -1:
            c = F.interpolate(c, size=(self.downsample_cond_size, self.downsample_cond_size))  # 下采样
        quant_c, _, [_, _, indices] = self.cond_stage_model.encode(c)  # 编码条件
        if len(indices.shape) > 2:
            indices = indices.view(c.shape[0], -1)  # 处理索引维度
        return quant_c, indices  # 返回量化后的条件和索引

    @torch.no_grad()
    def decode_to_img(self, index, zshape):
        """将索引解码为图像。"""
        index = self.permuter(index, reverse=True)  # 恢复排列
        bhwc = (zshape[0], zshape[2], zshape[3], zshape[1])  # 获取解码形状
        quant_z = self.first_stage_model.quantize.get_codebook_entry(
            index.reshape(-1), shape=bhwc)  # 获取量化的z
        x = self.first_stage_model.decode(quant_z)  # 解码为图像
        return x  # 返回解码后的图像

    @torch.no_grad()
    def log_images(self, batch, temperature=None, top_k=None, callback=None, lr_interface=False, **kwargs):
        """记录生成的图像，用于可视化。"""
        log = dict()  # 存储日志的字典

        N = 4  # 记录的样本数量
        if lr_interface:
            x, c = self.get_xc(batch, N, diffuse=False, upsample_factor=8)  # 从批次中获取输入和条件
        else:
            x, c = self.get_xc(batch, N)  # 从批次中获取输入和条件
        x = x.to(device=self.device)  # 将输入移到当前设备
        c = c.to(device=self.device)  # 将条件移到当前设备

        quant_z, z_indices = self.encode_to_z(x)  # 编码图像
        quant_c, c_indices = self.encode_to_c(c)  # 编码条件

        # 创建一个“半”样本
        z_start_indices = z_indices[:, :z_indices.shape[1] // 2]  # 使用一半的索引进行样本生成
        index_sample = self.sample(z_start_indices, c_indices,
                                   steps=z_indices.shape[1] - z_start_indices.shape[1],
                                   temperature=temperature if temperature is not None else 1.0,
                                   sample=True,
                                   top_k=top_k if top_k is not None else 100,
                                   callback=callback if callback is not None else lambda k: None)
        x_sample = self.decode_to_img(index_sample, quant_z.shape)  # 解码样本

        # 生成样本
        z_start_indices = z_indices[:, :0]  # 使用空索引生成样本
        index_sample = self.sample(z_start_indices, c_indices,
                                   steps=z_indices.shape[1],
                                   temperature=temperature if temperature is not None else 1.0,
                                   sample=True,
                                   top_k=top_k if top_k is not None else 100,
                                   callback=callback if callback is not None else lambda k: None)
        x_sample_nopix = self.decode_to_img(index_sample, quant_z.shape)  # 解码无像素样本

        # 确定性样本
        z_start_indices = z_indices[:, :0]  # 使用空索引生成样本
        index_sample = self.sample(z_start_indices, c_indices,
                                   steps=z_indices.shape[1],
                                   sample=False,
                                   callback=callback if callback is not None else lambda k: None)
        x_sample_det = self.decode_to_img(index_sample, quant_z.shape)  # 解码确定性样本

        # 重构样本
        x_rec = self.decode_to_img(z_indices, quant_z.shape)  # 将z_indices解码为重构图像

        log["inputs"] = x  # 记录输入图像
        log["reconstructions"] = x_rec  # 记录重构图像

        if self.cond_stage_key in ["objects_bbox", "objects_center_points"]:
            figure_size = (x_rec.shape[2], x_rec.shape[3])
            dataset = kwargs["pl_module"].trainer.datamodule.datasets["validation"]  # 获取验证集数据集
            label_for_category_no = dataset.get_textual_label_for_category_no  # 获取类别标签
            plotter = dataset.conditional_builders[self.cond_stage_key].plot  # 获取条件构建器的绘制方法
            log["conditioning"] = torch.zeros_like(log["reconstructions"])  # 初始化条件记录
            for i in range(quant_c.shape[0]):
                log["conditioning"][i] = plotter(quant_c[i], label_for_category_no, figure_size)  # 记录条件图像
            log["conditioning_rec"] = log["conditioning"]  # 记录条件重构图像
        elif self.cond_stage_key != "image":  # 如果条件阶段不是图像
            cond_rec = self.cond_stage_model.decode(quant_c)  # 解码条件
            if self.cond_stage_key == "segmentation":
                # 从分割掩码中获取图像
                num_classes = cond_rec.shape[1]  # 获取类别数量

                c = torch.argmax(c, dim=1, keepdim=True)  # 获取条件的类别索引
                c = F.one_hot(c, num_classes=num_classes)  # 创建热编码
                c = c.squeeze(1).permute(0, 3, 1, 2).float()  # 调整维度
                c = self.cond_stage_model.to_rgb(c)  # 转换为RGB

                cond_rec = torch.argmax(cond_rec, dim=1, keepdim=True)  # 获取条件重构的类别索引
                cond_rec = F.one_hot(cond_rec, num_classes=num_classes)  # 创建热编码
                cond_rec = cond_rec.squeeze(1).permute(0, 3, 1, 2).float()  # 调整维度
                cond_rec = self.cond_stage_model.to_rgb(cond_rec)  # 转换为RGB
            log["conditioning_rec"] = cond_rec  # 记录条件重构
            log["conditioning"] = c  # 记录条件

        log["samples_half"] = x_sample  # 记录半样本
        log["samples_nopix"] = x_sample_nopix  # 记录无像素样本
        log["samples_det"] = x_sample_det  # 记录确定性样本
        return log  # 返回日志

    def get_input(self, key, batch):
        """从批次中获取输入数据。"""
        x = batch[key]  # 获取对应键的输入
        if len(x.shape) == 3:
            x = x[..., None]  # 如果维度为3，则添加一个维度
        if len(x.shape) == 4:
            x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)  # 调整维度
        if x.dtype == torch.double:
            x = x.float()  # 如果数据类型为double，则转换为float
        return x  # 返回处理后的输入

    def get_xc(self, batch, N=None):
        """从批次中获取图像和条件数据。"""
        x = self.get_input(self.first_stage_key, batch)  # 获取输入数据
        c = self.get_input(self.cond_stage_key, batch)  # 获取条件数据
        if N is not None:
            x = x[:N]  # 选择前N个样本
            c = c[:N]  # 选择前N个条件
        return x, c  # 返回图像和条件

    def shared_step(self, batch, batch_idx):
        """共享的前向传播步骤，用于计算损失。"""
        x, c = self.get_xc(batch)  # 获取图像和条件
        logits, target = self(x, c)  # 前向传播获得logits和目标
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1))  # 计算交叉熵损失
        return loss  # 返回损失

    def training_step(self, batch, batch_idx):
        """训练步骤，调用共享步骤并记录损失。"""
        loss = self.shared_step(batch, batch_idx)  # 计算损失
        self.log("train/loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)  # 记录损失
        return loss  # 返回损失

    def validation_step(self, batch, batch_idx):
        """验证步骤，调用共享步骤并记录损失。"""
        loss = self.shared_step(batch, batch_idx)  # 计算损失
        self.log("val/loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)  # 记录损失
        return loss  # 返回损失

    def configure_optimizers(self):
        """配置优化器和学习率调度器。"""
        # 将所有参数分为两个组：参与正则化的权重衰减和不参与的权重
        decay = set()  # 权重衰减参数集合
        no_decay = set()  # 不参与权重衰减参数集合
        whitelist_weight_modules = (torch.nn.Linear,)  # 允许权重衰减的模块
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)  # 不允许权重衰减的模块

        for mn, m in self.transformer.named_modules():  # 遍历变换器中的所有模块
            for pn, p in m.named_parameters():  # 遍历每个模块的所有参数
                fpn = '%s.%s' % (mn, pn) if mn else pn  # 完整参数名称

                if pn.endswith('bias'):
                    # 所有偏置不会被衰减
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # 白名单模块的权重将被权重衰减
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # 黑名单模块的权重不会被衰减
                    no_decay.add(fpn)

        # 将位置嵌入参数特殊处理为不衰减
        no_decay.add('pos_emb')

        # 验证是否考虑了所有参数
        param_dict = {pn: p for pn, p in self.transformer.named_parameters()}  # 获取所有参数
        inter_params = decay & no_decay  # 查找交集参数
        union_params = decay | no_decay  # 查找并集参数
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert len(
            param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params),)

        # 创建PyTorch优化器对象
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": 0.01},  # 权重衰减的参数
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},  # 不衰减的参数
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=self.learning_rate, betas=(0.9, 0.95))  # 使用AdamW优化器
        return optimizer  # 返回优化器