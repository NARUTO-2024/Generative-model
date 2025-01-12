from abc import abstractmethod
from functools import partial
import math
from typing import Iterable
import torch

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

try:
    import xformers
    import xformers.ops
    XFORMERS_IS_AVAILBLE = True
except:
    XFORMERS_IS_AVAILBLE = False

from ldm.modules.diffusionmodules.util import (
    checkpoint,
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    normalization,
    timestep_embedding,
)
from ldm.modules.attention import SpatialTransformer, SpatialTransformerV2
from ldm.modules.spade import SPADE

from basicsr.archs.stylegan2_arch import ConvLayer, EqualConv2d
# dummy replace
def convert_module_to_f16(x):
    pass

def convert_module_to_f32(x):
    pass

def exists(val):
    return val is not None

def cal_fea_cossim(fea_1, fea_2, save_dir=None):
    cossim_fuc = nn.CosineSimilarity(dim=-1, eps=1e-6)
    if save_dir is None:
        save_dir_1 = './cos_sim64_1_not.txt'
        save_dir_2 = './cos_sim64_2_not.txt'
    b, c, h, w = fea_1.size()
    fea_1 = fea_1.reshape(b, c, h*w)
    fea_2 = fea_2.reshape(b, c, h*w)
    cos_sim = cossim_fuc(fea_1, fea_2)
    cos_sim = cos_sim.data.cpu().numpy()
    with open(save_dir_1, "a") as my_file:
        my_file.write(str(np.mean(cos_sim[0])) + "\n")
    # with open(save_dir_2, "a") as my_file:
    #     my_file.write(str(np.mean(cos_sim[1])) + "\n")

## go
class AttentionPool2d(nn.Module):
    """
    Adapted from CLIP: https://github.com/openai/CLIP/blob/main/clip/model.py
    """

    def __init__(
        self,
        spacial_dim: int,
        embed_dim: int,
        num_heads_channels: int,
        output_dim: int = None,
    ):
        super().__init__()
        self.positional_embedding = nn.Parameter(th.randn(embed_dim, spacial_dim ** 2 + 1) / embed_dim ** 0.5)
        self.qkv_proj = conv_nd(1, embed_dim, 3 * embed_dim, 1)
        self.c_proj = conv_nd(1, embed_dim, output_dim or embed_dim, 1)
        self.num_heads = embed_dim // num_heads_channels
        self.attention = QKVAttention(self.num_heads)

    def forward(self, x):
        b, c, *_spatial = x.shape
        x = x.reshape(b, c, -1)  # NC(HW)
        x = th.cat([x.mean(dim=-1, keepdim=True), x], dim=-1)  # NC(HW+1)
        x = x + self.positional_embedding[None, :, :].to(x.dtype)  # NC(HW+1)
        x = self.qkv_proj(x)
        x = self.attention(x)
        x = self.c_proj(x)
        return x[:, :, 0]


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """

class TimestepBlockDual(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb, cond):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """

class TimestepBlock3cond(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb, s_cond, seg_cond):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb, context=None, struct_cond=None, seg_cond=None):
        # 遍历该模块中的每个子模块
        for layer in self:
            if isinstance(layer, TimestepBlock):
                # 如果子模块是 TimestepBlock, 则将 x 和 emb 作为参数传递
                x = layer(x, emb)
            elif isinstance(layer, SpatialTransformer) or isinstance(layer, SpatialTransformerV2):
                # 如果子模块是 SpatialTransformer 或其变种，则需要提供上下文信息
                assert context is not None
                x = layer(x, context)
            elif isinstance(layer, TimestepBlockDual):
                # 如果子模块是双时间步块，则需要提供结构条件
                assert struct_cond is not None
                x = layer(x, emb, struct_cond)
            elif isinstance(layer, TimestepBlock3cond):
                # 如果子模块接受三个条件，则需要提供分割条件
                assert seg_cond is not None
                x = layer(x, emb, struct_cond, seg_cond)
            else:
                # 对于其他层，仅仅以 x 为参数进行正向传播
                x = layer(x)
        return x  # 返回最终的输出


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None, padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=padding)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(
                x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
            )
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x

class TransposedUpsample(nn.Module):
    'Learned 2x upsampling without padding'
    def __init__(self, channels, out_channels=None, ks=5):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels

        self.up = nn.ConvTranspose2d(self.channels,self.out_channels,kernel_size=ks,stride=2)

    def forward(self,x):
        return self.up(x)


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None,padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(
                dims, self.channels, self.out_channels, 3, stride=stride, padding=padding
            )
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
        up=False,
        down=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )

        if self.out_channels % 32 == 0:
            self.out_layers = nn.Sequential(
                normalization(self.out_channels),
                nn.SiLU(),
                nn.Dropout(p=dropout),
                zero_module(
                    conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
                ),
            )
        else:
            self.out_layers = nn.Sequential(
                normalization(self.out_channels, self.out_channels),
                nn.SiLU(),
                nn.Dropout(p=dropout),
                zero_module(
                    conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
                ),
            )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.
        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(
            self._forward, (x, emb), self.parameters(), self.use_checkpoint
        )


    def _forward(self, x, emb):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h

class ResBlockDual(TimestepBlockDual):
    """
    一个残差块，可以选择性地改变通道数。
    :param channels: 输入通道的数量。
    :param emb_channels: 时间步嵌入的通道数量。
    :param dropout: dropout 的比例。
    :param out_channels: 如果指定，则输出通道的数量。
    :param use_conv: 如果为 True，并且指定了 out_channels，则使用空间卷积而不是较小的 1x1 卷积来改变跳跃连接中的通道。
    :param dims: 指定信号是1D、2D还是3D。
    :param use_checkpoint: 如果为 True，则在该模块上使用梯度检查点。
    :param up: 如果为 True，则使用该块进行上采样。
    :param down: 如果为 True，则使用该块进行下采样。
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        semb_channels,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
        up=False,
        down=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()  #不执行任何操作，仅仅返回输入的输出

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )

        # Here we use the built component of SPADE, rather than SFT. Should have no significant influence on the performance.
        self.spade = SPADE(self.out_channels, semb_channels)

        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb, s_cond):
        """
        以时间步嵌入为条件，对一个张量应用块。
        :param x：一个[N x C x ...]特征张量。
        :param emb: 一个 [N x emb_channels] 时间步嵌入张量。
        :return：一个 [N x C x ...] 输出张量。
        """
        return checkpoint(
            self._forward, (x, emb, s_cond), self.parameters(), self.use_checkpoint
        )


    def _forward(self, x, emb, s_cond):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(emb_out, 2, dim=1)  #emb_out在第一个维度上被平分，返回scale和shift
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        h = self.spade(h, s_cond)
        return self.skip_connection(x) + h


class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.
    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(
            self,
            channels,  # 输入特征图的通道数
            num_heads=1,  # 注意力头的数量
            num_head_channels=-1,  # 每个头的通道数，如果为-1，则直接用channels // num_heads
            use_checkpoint=False,  # 是否使用检查点
            use_new_attention_order=False,  # 是否使用新的注意力顺序
    ):
        super().__init__()  # 调用父类的构造函数

        self.channels = channels  # 保存输入的通道数

        # 根据 num_head_channels 指定每个头的通道数
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            # 确保通道数能够被每个头的通道数整除
            assert (
                    channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels  # 计算每个头的通道数

        self.use_checkpoint = use_checkpoint  # 是否使用检查点用于反向传播中的内存优化
        self.norm = normalization(channels)  # 规范化层
        self.qkv = conv_nd(1, channels, channels * 3, 1)  # Q, K, V的卷积层

        # 根据是否使用新注意力顺序选择注意力实现
        if use_new_attention_order:
            self.attention = QKVAttention(self.num_heads)  # 新版注意力实现
        else:
            self.attention = QKVAttentionLegacy(self.num_heads)  # 旧版注意力实现

        # 输出卷积，使用零初始化
        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x):
        # 前向传播函数，使用检查点来节省内存
        return checkpoint(self._forward, (x,), self.parameters(), True)
        # TODO: check checkpoint usage, is True # TODO: fix the .half call!!!
        # return pt_checkpoint(self._forward, x)  # 使用 PyTorch 的检查点

    def _forward(self, x):
        b, c, *spatial = x.shape  # 获取输入的批次大小、通道数和空间维度
        x = x.reshape(b, c, -1)  # 将输入重塑为2D，形状为 (batch_size, channels, num_pixels)

        qkv = self.qkv(self.norm(x))  # 经过规范化后的输入计算 Q, K, V
        h = self.attention(qkv)  # 通过注意力机制得到输出
        h = self.proj_out(h)  # 通过输出卷积层进行变换

        return (x + h).reshape(b, c, *spatial)  # 将结果按原形状返回，进行残差连接


def count_flops_attn(model, _x, y):
    """
    A counter for the `thop` package to count the operations in an
    attention operation.
    Meant to be used like:
        macs, params = thop.profile(
            model,
            inputs=(inputs, timestamps),
            custom_ops={QKVAttention: QKVAttention.count_flops},
        )
    """
    b, c, *spatial = y[0].shape
    num_spatial = int(np.prod(spatial))
    # We perform two matmuls with the same number of ops.
    # The first computes the weight matrix, the second computes
    # the combination of the value vectors.
    matmul_ops = 2 * b * (num_spatial ** 2) * c
    model.total_ops += th.DoubleTensor([matmul_ops])

class QKVAttentionLegacy(nn.Module):
    """
    A module which performs QKV attention. Matches legacy QKVAttention + input/ouput heads shaping
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.attention_op: Optional[Any] = None

    def forward(self, qkv):
        """
        Apply QKV attention.
        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        if XFORMERS_IS_AVAILBLE:
            q, k, v = map(
                lambda t:t.permute(0,2,1)
                .contiguous(),
                (q, k, v),
            )
            # actually compute the attention, what we cannot get enough of
            a = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None, op=self.attention_op)
            a = (
                a.permute(0,2,1)
                .reshape(bs, -1, length)
            )
        else:
            weight = th.einsum(
                "bct,bcs->bts", q * scale, k * scale
            )  # More stable with f16 than dividing afterwards
            weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
            a = th.einsum("bts,bcs->bct", weight, v)
            a = a.reshape(bs, -1, length)
        return a

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class QKVAttention(nn.Module):
    """
    一个模块，用于执行 QKV 注意力机制，并以不同的顺序进行拆分。
    """

    def __init__(self, n_heads):
        # 初始化函数，接收头的数量 n_heads
        super().__init__()
        self.n_heads = n_heads  # 记录头的数量
        self.attention_op: Optional[Any] = None  # 注意力运算的可选操作（初始化为空）

    def forward(self, qkv):
        """
        应用 QKV 注意力机制。
        :param qkv: 一个形状为 [N x (3 * H * C) x T] 的张量，包含 Q、K 和 V。
        :return: 一个形状为 [N x (H * C) x T] 的张量，经过注意力后的结果。
        """
        bs, width, length = qkv.shape  # 获取批次大小、宽度和长度
        assert width % (3 * self.n_heads) == 0  # 确保宽度可以被 (3 * n_heads) 整除
        ch = width // (3 * self.n_heads)  # 计算每个头的通道数

        # 将 qkv 张量拆分为 q、k 和 v
        q, k, v = qkv.chunk(3, dim=1)  # 在维度 1 上分割为 3 个张量
        scale = 1 / math.sqrt(math.sqrt(ch))  # 计算缩放因子

        # 如果 xformers 可用，使用高效的注意力计算
        if XFORMERS_IS_AVAILBLE:
            # Q、K、V张量进行维度重排列，从 [N, C, T]（批次大小 x 通道数 x 序列长度）转换为 [N, T, C]
            q, k, v = map(  #map() 将对 q、k 和 v 三个张量执行相同的操作
                lambda t: t.permute(0, 2, 1).contiguous(),
                (q, k, v),
            )
            # 实际计算注意力，这是一个重要而高效的操作
            a = xformers.ops.memory_efficient_attention(
                q, k, v, attn_bias=None, op=self.attention_op
            )
            # 转置输出并重新调整维度
            a = a.permute(0, 2, 1).reshape(bs, -1, length)
        else:
            # 如果 xformers 不可用，使用传统计算方法
            weight = th.einsum(
                "bct,bcs->bts",
                (q * scale).view(bs * self.n_heads, ch, length),  # 缩放 q
                (k * scale).view(bs * self.n_heads, ch, length),  # 缩放 k
            )  # 使用爱因斯坦求和表示法计算权重
            # 对权重进行 softmax 处理以规范化(), dim=-1).type(weight.dtype)
            # 计算最终的注意力输出
            weight = th.softmax(weight.float)
            a = th.einsum("bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length))
            # 重新调整输出维度
            a = a.reshape(bs, -1, length)

        return a  # 返回经过注意力计算后的结果张量

    @staticmethod
    def count_flops(model, _x, y):
        """静态方法，用于计算模型的浮点运算（FLOPs）"""
        return count_flops_attn(model, _x, y)  # 调用另一个函数计算 FLOPs

class UNetModel(nn.Module):
    """
    The full UNet model with attention and timestep embedding.
    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param use_new_attention_order: use a different attention pattern for potentially
                                    increased efficiency.
    """

    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=-1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        use_spatial_transformer=False,    # custom transformer support
        transformer_depth=1,              # custom transformer support
        context_dim=None,                 # custom transformer support
        n_embed=None,                     # custom support for prediction of discrete ids into codebook of first stage vq model
        legacy=True,
    ):
        super().__init__()
        if use_spatial_transformer:
            assert context_dim is not None, 'Fool!! You forgot to include the dimension of your cross-attention conditioning...'

        if context_dim is not None:
            assert use_spatial_transformer, 'Fool!! You forgot to use the spatial transformer for your cross-attention conditioning...'
            from omegaconf.listconfig import ListConfig
            if type(context_dim) == ListConfig:
                context_dim = list(context_dim)

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if num_heads == -1:
            assert num_head_channels != -1, 'Either num_heads or num_head_channels has to be set'

        if num_head_channels == -1:
            assert num_heads != -1, 'Either num_heads or num_head_channels has to be set'

        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.predict_codebook_ids = n_embed is not None

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1)
                )
            ]
        )
        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        #num_heads = 1
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=dim_head,
                            use_new_attention_order=use_new_attention_order,
                        ) if not use_spatial_transformer else SpatialTransformer(
                            ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        if legacy:
            #num_heads = 1
            dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=dim_head,
                use_new_attention_order=use_new_attention_order,
            ) if not use_spatial_transformer else SpatialTransformer(
                            ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim
                        ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=model_channels * mult,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        #num_heads = 1
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads_upsample,
                            num_head_channels=dim_head,
                            use_new_attention_order=use_new_attention_order,
                        ) if not use_spatial_transformer else SpatialTransformer(
                            ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim
                        )
                    )
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, model_channels, out_channels, 3, padding=1)),
        )
        if self.predict_codebook_ids:
            self.id_predictor = nn.Sequential(
            normalization(ch),
            conv_nd(dims, model_channels, n_embed, 1),
            #nn.LogSoftmax(dim=1)  # change to cross_entropy and produce non-normalized logits
        )

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)

    def forward(self, x, timesteps=None, context=None, y=None,**kwargs):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param context: conditioning plugged in via crossattn
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"
        hs = []
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)

        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)

        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb, context)
            hs.append(h)
        h = self.middle_block(h, emb, context)
        for module in self.output_blocks:
            h = th.cat([h, hs.pop()], dim=1)
            h = module(h, emb, context)
        h = h.type(x.dtype)
        if self.predict_codebook_ids:
            return self.id_predictor(h)
        else:
            return self.out(h)


class UNetModelDualcondV2(nn.Module):
    """
    完整的带有注意力机制和时间步嵌入的 U-Net 模型。
    :param in_channels: 输入 Tensor 的通道数。
    :param model_channels: 模型的基本通道数。
    :param out_channels: 输出 Tensor 的通道数。
    :param num_res_blocks: 每次下采样的残差块数量。
    :param attention_resolutions: 注意力机制应用的下采样率集合。
    :param dropout: dropout 概率。
    :param channel_mult: U-Net 每一层的通道扩展倍数。
    :param conv_resample: 如果为 True，使用学习到的卷积进行上采样和下采样。
    :param dims: 决定信号是 1D、2D 还是 3D。
    :param num_classes: 如果指定（整数），则模型为条件模型，有 num_classes 个类别。
    :param use_checkpoint: 使用梯度检查点以减少内存使用。
    :param num_heads: 每个注意力层中的注意力头数。
    :param num_heads_channels: 如果指定，则忽略 num_heads，使用每个注意力头的固定通道宽度。
    :param num_heads_upsample: 设置上采样的不同头数量，已弃用。
    :param use_scale_shift_norm: 使用类似 FiLM 的条件机制。
    :param resblock_updown: 对上采样/下采样使用残差块。
    :param use_new_attention_order: 使用不同的注意力模式以提高效率。
    """

    def __init__(
            self,
            image_size,
            in_channels,
            model_channels,
            out_channels,
            num_res_blocks,
            attention_resolutions,
            dropout=0,
            channel_mult=(1, 2, 4, 8),
            conv_resample=True,
            dims=2,
            num_classes=None,
            use_checkpoint=False,
            use_fp16=False,
            num_heads=-1,
            num_head_channels=-1,
            num_heads_upsample=-1,
            use_scale_shift_norm=False,
            resblock_updown=False,
            use_new_attention_order=False,
            use_spatial_transformer=False,  # 支持自定义变换器
            transformer_depth=1,  # 支持自定义变换器
            context_dim=None,  # 支持自定义变换器的上下文维度
            n_embed=None,  # 自定义支持，预测离散 ID 到第一个阶段 VQ 模型的代码簿
            legacy=True,
            disable_self_attentions=None,
            num_attention_blocks=None,
            disable_middle_self_attn=False,
            use_linear_in_transformer=False,
            semb_channels=None
    ):
        # 初始化父类 nn.Module
        super().__init__()

        # 处理 spatial transformer 的上下文维度
        if use_spatial_transformer:
            assert context_dim is not None, 'Fool!! You forgot to include the dimension of your cross-attention conditioning...'

        # 确保 context_dim 以适当形式存在
        if context_dim is not None:
            assert use_spatial_transformer, 'Fool!! You forgot to use the spatial transformer for your cross-attention conditioning...'
            from omegaconf.listconfig import ListConfig
            if type(context_dim) == ListConfig:
                context_dim = list(context_dim)

        # 处理上采样的头数
        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        # 确保群体头数或每头通道数被设置
        if num_heads == -1:
            assert num_head_channels != -1, 'Either num_heads or num_head_channels has to be set'

        if num_head_channels == -1:
            assert num_heads != -1, 'Either num_heads or num_head_channels has to be set'

        # 初始化输入参数
        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels

        # 初始化每个下采样层的残差块数目
        if isinstance(num_res_blocks, int):
            self.num_res_blocks = len(channel_mult) * [num_res_blocks]  #创建一个channel_mult长度的数组，其中的每一个值都是num_res_blocks
        else:
            if len(num_res_blocks) != len(channel_mult):
                raise ValueError("provide num_res_blocks either as an int (globally constant) or "
                                 "as a list/tuple (per-level) with the same length as channel_mult")
            self.num_res_blocks = num_res_blocks

        # 检查自注意力禁用选项
        if disable_self_attentions is not None:
            assert len(disable_self_attentions) == len(channel_mult)

        # 检查注意力块数
        if num_attention_blocks is not None:
            assert len(num_attention_blocks) == len(self.num_res_blocks)
            assert all(
                map(lambda i: self.num_res_blocks[i] >= num_attention_blocks[i], range(len(num_attention_blocks))))
            print(f"Constructor of UNetModel received num_attention_blocks={num_attention_blocks}. "
                  f"This option has LESS priority than attention_resolutions {attention_resolutions}, "
                  f"i.e., in cases where num_attention_blocks[i] > 0 but 2**i not in attention_resolutions, "
                  f"attention will still not be set.")

        # 初始化其他模型参数
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.predict_codebook_ids = n_embed is not None

        # 定义时间嵌入层
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),  #igmoid Linear Unit 激活函数，也被称为 Swish 激活函数
            linear(time_embed_dim, time_embed_dim),
        )

        # 对类别标记进行嵌入处理
        if self.num_classes is not None:
            if isinstance(self.num_classes, int):
                self.label_emb = nn.Embedding(num_classes, time_embed_dim)  #nn.Embedding 接收一个整数索引（例如，单词的索引，或类别标签的索引），输出该索引对应的嵌入向量，嵌入层的权重（即每个类别对应的向量）是可学习的参数。
            elif self.num_classes == "continuous":
                print("setting up linear c_adm embedding layer")
                self.label_emb = nn.Linear(1, time_embed_dim)
            else:
                raise ValueError()

        # 输入块（U-Net 的下采样部分）
        # 创建一个 ModuleList，里面存放模型的输入块
        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1)  # 初始卷积层，使用 conv_nd 创建一个卷积层  ，TimestepEmbedSequential是一个选择机制，在这里只是简单的构建。使用这个类是需要输入x emb以及其他信息的
                )
            ]
        )
        self._feature_size = model_channels  # 初始化特征大小
        input_block_chans = [model_channels]  # 记录输入块的通道数，初始化为 model_channels
        ch = model_channels  # 当前通道数
        ds = 1  # 下采样因子，初始化为 1

        # 构建 U-Net 的下采样路径
        for level, mult in enumerate(channel_mult):  # channel_mult 如 (1, 2, 4, 8)
            for nr in range(self.num_res_blocks[level]):  # 基于每层的残差块数
                layers = [
                    ResBlockDual(  # 使用自定义的双残差块
                        ch,  # 输入通道数
                        time_embed_dim,  # 时间嵌入的维度
                        dropout,  # dropout 的比率
                        semb_channels=semb_channels,  # 结构嵌入的通道数
                        out_channels=mult * model_channels,  # 输出通道数
                        dims=dims,  # 维度信息
                        use_checkpoint=use_checkpoint,  # 是否使用检查点
                        use_scale_shift_norm=use_scale_shift_norm,  # 使用尺度-偏移归一化
                    )
                ]
                ch = mult * model_channels  # 更新当前通道数

                # 检查是否需要添加注意力机制
                if ds in attention_resolutions:  # 如果当前下采样因子在关注分辨率中
                    if num_head_channels == -1:
                        dim_head = ch // num_heads  # 计算头的维度
                    else:
                        num_heads = ch // num_head_channels  # 每个头的通道数
                        dim_head = num_head_channels  # 每个头的通道数

                    # 处理遗留的注意力机制参数
                    if legacy:
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels

                    # 是否禁用自注意力机制
                    if exists(disable_self_attentions):
                        disabled_sa = disable_self_attentions[level]
                    else:
                        disabled_sa = False

                    # 如果没有规定自注意力块或当前块在允许添加的块数内，则添加注意力块
                    if not exists(num_attention_blocks) or nr < num_attention_blocks[level]:
                        layers.append(
                            AttentionBlock(  # 添加注意力块
                                ch,
                                use_checkpoint=use_checkpoint,  # 是否使用检查点
                                num_heads=num_heads,  # 注意力头的数量
                                num_head_channels=dim_head,  # 每个头的通道数
                                use_new_attention_order=use_new_attention_order,  # 是否使用新的注意力顺序
                            ) if not use_spatial_transformer else SpatialTransformerV2(
                                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                                disable_self_attn=disabled_sa, use_linear=use_linear_in_transformer,
                                use_checkpoint=use_checkpoint
                            )
                        )

                # 构建当前的输入块并添加到 input_blocks
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch  # 更新特征大小
                input_block_chans.append(ch)  # 添加当前通道数到记录

            # 如果不是最后一级，添加下采样层
            if level != len(channel_mult) - 1:
                out_ch = ch  # 当前通道数作为输出通道数
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlockDual(
                            ch,
                            time_embed_dim,
                            dropout,
                            semb_channels=semb_channels,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,  # 表示要下采样
                        )
                        if resblock_updown  # 根据条件判断是否使用 ResBlockDual
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch  # 否则使用 Downsample
                        )
                    )
                )
                ch = out_ch  # 更新当前通道数
                input_block_chans.append(ch)  # 记录当前通道数
                ds *= 2  # 更新下采样因子
                self._feature_size += ch  # 更新特征大小
        # 构建中间块
        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels

        if legacy:
            dim_head = ch // num_heads if use_spatial_transformer else num_head_channels

        self.middle_block = TimestepEmbedSequential(
            ResBlockDual(
                ch,
                time_embed_dim,
                dropout,
                semb_channels=semb_channels,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(  # 中间的注意力块
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=dim_head,
                use_new_attention_order=use_new_attention_order,
            ) if not use_spatial_transformer else SpatialTransformerV2(
                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                disable_self_attn=disable_middle_self_attn, use_linear=use_linear_in_transformer,
                use_checkpoint=use_checkpoint
            ),
            ResBlockDual(
                ch,
                time_embed_dim,
                dropout,
                semb_channels=semb_channels,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += ch

        # 构建 U-Net 的上采样路径
        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(self.num_res_blocks[level] + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlockDual(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        semb_channels=semb_channels,
                        out_channels=model_channels * mult,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = model_channels * mult

                # 处理注意力机制
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels

                    if legacy:
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels

                    if exists(disable_self_attentions):
                        disabled_sa = disable_self_attentions[level]
                    else:
                        disabled_sa = False

                    if not exists(num_attention_blocks) or i < num_attention_blocks[level]:
                        layers.append(
                            AttentionBlock(
                                ch,
                                use_checkpoint=use_checkpoint,
                                num_heads=num_heads_upsample,
                                num_head_channels=dim_head,
                                use_new_attention_order=use_new_attention_order,
                            ) if not use_spatial_transformer else SpatialTransformerV2(
                                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                                disable_self_attn=disabled_sa, use_linear=use_linear_in_transformer,
                                use_checkpoint=use_checkpoint
                            )
                        )
                # 构建上采样
                if level and i == self.num_res_blocks[level]:
                    out_ch = ch
                    layers.append(
                        ResBlockDual(
                            ch,
                            time_embed_dim,
                            dropout,
                            semb_channels=semb_channels,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    ds //= 2

                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        # 定义输出层
        self.out = nn.Sequential(
            normalization(ch),  # 正则化层
            nn.SiLU(),  # 激活函数
            zero_module(conv_nd(dims, model_channels, out_channels, 3, padding=1)),  # 输出卷积
        )

        # 如果需要预测代码簿 ID，则定义 ID 预测层
        if self.predict_codebook_ids:
            self.id_predictor = nn.Sequential(
                normalization(ch),
                conv_nd(dims, model_channels, n_embed, 1),  # 最后的卷积层
                # nn.LogSoftmax(dim=1)  # 可以用于交叉熵损失
            )

    def convert_to_fp16(self):
        """
        将模型的主体转换为 float16 类型。
        """
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        将模型的主体转换为 float32 类型。
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)

    def forward(self, x, timesteps=None, context=None, struct_cond=None, y=None, **kwargs):
        """
        将输入批次应用于模型。
        :param x: 输入的 [N x C x ...] 张量。
        :param timesteps: 一维的时间步长批次。
        :param context: 通过交叉注意获得的条件。
        :param y: [N] 张量，表示类别标签，如果是条件模型。
        :return: 输出的 [N x C x ...] 张量。
        """
        assert (y is not None) == (
                self.num_classes is not None  #y和num_class必须是none
        ), "must specify y if and only if the model is class-conditional"

        hs = []  # 保存每个输入块的输出
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)  # 获取时间步嵌入
        emb = self.time_embed(t_emb)  # 嵌入操作

        # 如果模型是条件的，添加类别嵌入
        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)  #通过检查 y 的形状是否等于 x 的第一维，确保每个输入样本都有对应的标签
            emb = emb + self.label_emb(y)

        h = x.type(self.dtype)  # 将输入转化为模型的 dtype
        # 通过输入块进行前向传播
        for module in self.input_blocks:
            h = module(h, emb, context, struct_cond)  #在每次循环中，当前的 module 被调用，传入以下参数
            hs.append(h)

        # 通过中间块
        h = self.middle_block(h, emb, context, struct_cond)

        # 通过输出块
        for module in self.output_blocks:
            h = th.cat([h, hs.pop()], dim=1)  # 拼接特征
            h = module(h, emb, context, struct_cond)

        h = h.type(x.dtype)  # 转换回输入的 dtype


        if self.predict_codebook_ids:
            return self.id_predictor(h)  # 返回 ID 预测
        else:
            return self.out(h)  # 返回模型的最终输出



class EncoderUNetModelWT(nn.Module):
    """
    The half UNet model with attention and timestep embedding.
    For usage, see UNet.
    """

    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        *args,
        **kwargs
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1)
                )
            ]
        )
        self._feature_size = model_channels
        input_block_chans = []
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=num_head_channels,
                use_new_attention_order=use_new_attention_order,
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        input_block_chans.append(ch)
        self._feature_size += ch
        self.input_block_chans = input_block_chans

        self.fea_tran = nn.ModuleList([])

        for i in range(len(input_block_chans)):
            self.fea_tran.append(
                ResBlock(
                    input_block_chans[i],
                    time_embed_dim,
                    dropout,
                    out_channels=out_channels,
                    dims=dims,
                    use_checkpoint=use_checkpoint,
                    use_scale_shift_norm=use_scale_shift_norm,
                )
            )

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)

    def forward(self, x, timesteps):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :return: an [N x K] Tensor of outputs.
        """
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        result_list = []
        results = {}
        h = x.type(self.dtype)
        for module in self.input_blocks:
            last_h = h
            h = module(h, emb)
            if h.size(-1) != last_h.size(-1):
                result_list.append(last_h)
        h = self.middle_block(h, emb)
        result_list.append(h)

        assert len(result_list) == len(self.fea_tran)

        for i in range(len(result_list)):
            results[str(result_list[i].size(-1))] = self.fea_tran[i](result_list[i], emb)

        return results
