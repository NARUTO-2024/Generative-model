import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange
import math
import warnings
from torch.nn.init import _calculate_fan_in_and_fan_out


# 截断正态初始化，用于权重初始化
def _no_grad_trunc_normal_(tensor, mean, std, a, b):  #将（tensor)一个张量初始化为具有指定均值、标准差和截断区间的截断正态分布
    #mean 是正态分布的均值，std 是标准差，a 和 b 是截断区间的上下限
    def norm_cdf(x):  #用于计算标准正态分布的累积分布函数（CDF）
        return (1. + math.erf(x / math.sqrt(2.))) / 2.   #  标准正态分布随机变量小于或等于x的概率

    # 如果均值距离 [a, b] 区间大于2个标准差，发出警告   保证采样的值大部分落在截断区间内
    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)   #stacklevel=2 指的是调用当前函数的函数，


    with torch.no_grad():  #一个上下文管理器，用于确保在其中的代码块执行时，不会计算梯度
        l = norm_cdf((a - mean) / std)   #标准正态分布随机变量小于或等于a的概率为l
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)  # tensor.uniform_()：于在指定的区间内生成均匀分布的随机数，并原地（in-place）更新张量的值。  将[0,1]映射至[-1,1],由于逆误差函数需要
        tensor.erfinv_()  # 应用逆误差函数  将均匀分布的样本转换为截断正态分布的样本
        tensor.mul_(std * math.sqrt(2.))  # 乘以标准差和 sqrt(2)
        tensor.add_(mean)  # 加上均值
        tensor.clamp_(min=a, max=b)  # 将值限制在 [a, b] 范围内
        return tensor


# 截断正态分布的封装函数
def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


# 用于权重初始化的 方差缩放初始化
def variance_scaling_(tensor, scale=1.0, mode='fan_in', distribution='normal'):
    #缩放因子，默认值为1.0，用于调整权重的方差
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)  #已有的函数，计算传入张量的扇入（输入神经元的数量）和扇出（输出神经元的数量）

    if mode == 'fan_in':
        denom = fan_in
    elif mode == 'fan_out':
        denom = fan_out
    elif mode == 'fan_avg':
        denom = (fan_in + fan_out) / 2

    variance = scale / denom

    if distribution == "truncated_normal":
        trunc_normal_(tensor, std=math.sqrt(variance) / .87962566103423978)
    elif distribution == "normal":
        tensor.normal_(std=math.sqrt(variance))  #从指定均值和方差的正态分布中生成随机数，并赋值给张量
    elif distribution == "uniform":
        bound = math.sqrt(3 * variance)
        tensor.uniform_(-bound, bound)   #从均匀分布中抽样数值
    else:
        raise ValueError(f"invalid distribution {distribution}")  #如果提供的 distribution 不是上述三种之一，则抛出一个 ValueError 异常，提示分布类型无效。


# LeCun 正态初始化
def lecun_normal_(tensor):
    variance_scaling_(tensor, mode='fan_in', distribution='truncated_normal')


# 对给定层进行预归一化（pre-normalization）的封装类
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn  #fn是一个函数对象，表示这层要执行的操作，比如一个自注意力层或一个前馈神经网络层
        self.norm = nn.LayerNorm(dim)  #使用张量的维度dim实例化一个LayerNorm层
        #LayerNorm层会对输入数据的每个特征维度进行归一化处理，即计算每个特征维度上的均值和标准差，然后应用一个变换来保持特征值的均值和方差。
    def forward(self, x, *args, **kwargs):
        x = self.norm(x)  #输出x是一个归一化处理后的张量
        return self.fn(x, *args, **kwargs)  #*args 是一个包含位置参数的元组，而 **kwargs 是一个包含关键字参数的字典。
#这个函数为一层的操作，先对张量进行LN归一化，再进行fn操作，然后返回

# GELU（高斯误差线性单元）激活函数的实现
class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)  #返回GELU函数的值，是一种非线性激活函数


# 特定参数的卷积层
def conv(in_channels, out_channels, kernel_size, bias=False, padding=1, stride=1):
    return nn.Conv2d(  #创建二维卷积层
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias, stride=stride)
#in_channels：输入通道的数量。
#out_channels：输出通道的数量，即卷积层将生成的特征图的数量。
#kernel_size：卷积核的大小，可以是一个整数或者一个包含两个整数的元组。
#bias：布尔值，指定是否在卷积层中添加偏置项。
#padding：整数，指定在输入边界周围填充多少个零。
#stride：整数或包含两个整数的元组，指定卷积核在输入上滑动的步长。
#二维卷积层格式一般是一个四维张量(batch_size, in_channels, height, width)  batch_size 是输入数据中的样本数量  in_channels 是输入数据的通道数

# 用于移位张量数据的函数，通常用于对齐特征图
def shift_back(inputs, step=2):  # 输入 [bs,28,256,310] 输出 [bs, 28, 256, 256]     每个通道的宽度就被缩放和移动到了指定的 out_col 值。  裁去多余的值，直至正方形
    [bs, nC, row, col] = inputs.shape  #inputs.shape解析输入的格式，并将分离的inputs赋值给四个变量
    down_sample = 256 // row   #计算每个宽度方向上的采样点数量。
    step = float(step) / float(down_sample * down_sample)  #计算缩放和移动的步长
    out_col = row  #设置输出宽度 out_col 为原始宽度 row
    for i in range(nC):  #遍历输入张量的每个通道
        inputs[:, i, :, :out_col] = \
            inputs[:, i, :, int(step * i):int(step * i) + out_col]  #\用于续行
        # :：这个 : 表示不限制这个维度上的数据，因此会包含所有数据。
        # #i：这个索引 i 表示选择第 i 个通道的数据。
        # #:out_col：这个 :out_col 表示选择所有宽度（width）的数据，直到 out_col 这个位置。   最后是正方形
    return inputs[:, :, :, :out_col]


# 多尺度多头注意力机制（MS-MSA）类的定义
class MS_MSA(nn.Module):  #S-MSA
    def __init__(self, dim, dim_head, heads):
        super().__init__()
        self.num_heads = heads  # 注意力头的数量
        self.dim_head = dim_head  # 每个注意力头的维度  只能是dim，否则无法进行矩阵乘法运算
        self.to_q = nn.Linear(dim, dim_head * heads, bias=False)  # 线性层：生成 Q  self.to_q 用于将输入数据的维度从 dim 映射到 dim_head * heads
        self.to_k = nn.Linear(dim, dim_head * heads, bias=False)  # 线性层：生成 K
        self.to_v = nn.Linear(dim, dim_head * heads, bias=False)  # 线性层：生成 V
        #Linear创建一个线性层，基础神经网络
        self.rescale = nn.Parameter(torch.ones(heads, 1, 1))  # 缩放参数  缩放操作的目的是确保注意力得分（attention scores）的分布不会因为键张量被除以 dim_head 而变得过于集中。
        #torch.ones(heads, 1, 1)创建了一个[heads,1,1]维度的张量
        #Parameter创建一个参数
        self.proj = nn.Linear(dim_head * heads, dim, bias=True)  # 投影层   将输入数据的维度从 dim_head * heads 映射到 dim
        self.pos_emb = nn.Sequential(  # 位置编码   nn.Sequential(...)：这是PyTorch中用于创建序列模块的函数调用。序列模块是一个包含多个层的模块，它们按照定义的顺序依次执行。
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
            GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
        )#不使用直接的位置编码公式，而使用两个卷积层和一个激活函数来拟合位置编码  这种学习的位置编码被称为学习位置编码（Learned Positional Encoding）
        self.dim = dim  #输入数据的维度=通道数c

    def forward(self, x_in):
        """
        x_in: [b, h, w, c]
        return out: [b, h, w, c]
        """
        b, h, w, c = x_in.shape
        x = x_in.reshape(b, h * w, c)
        q_inp = self.to_q(x)  #输入 x_in 是一个形状为 [b, h*w, c] 的三维张量  输出 x_q 是一个形状为 [b, h * w, dim_head * heads] 的三维张量 线性层中乘的[1,heads]的w是q
        k_inp = self.to_k(x)
        v_inp = self.to_v(x)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads),
                      (q_inp, k_inp, v_inp))   #把 q_inp 的形式从 [b, h * w, dim_head * heads] ，变为 [b,  heads ， h * w ，dim_head]

        # q, k, v 的形状变为 [b, heads, hw, c]
        q = q.transpose(-2, -1)  #-2 表示最后一个维度，-1 表示倒数第二个维度  q 的原始形状 [b,  heads ， h * w ，dim_head] 将被转换为 [b,  heads ，dim_head， h * w ]
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)

        # 归一化 q 和 k  F 在这里就是 torch.nn.functional
        q = F.normalize(q, dim=-1, p=2)  #对 h*w 进行归一化处理，即对每张图片 b 上的每个多注意力头 heads 所采集到的每个颜色通道 c 的矩阵 h*w 的值归一化（矩阵的每个值归一化映射）
        k = F.normalize(k, dim=-1, p=2)  #q：输入张量 dim=-1：指定归一化操作的维度。-1 表示最后一个维度 p=2：指定用于计算的范数类型。p=2 表示使用L2范数。

        # 计算注意力分数
        attn = (k @ q.transpose(-2, -1))  # @：PyTorch中的矩阵乘法运算符  k：[b,  heads ，dim_head， h * w ]@ q：[b,  heads ， h * w ，dim_head]  attn：[b,  heads ，dim_head， dim_head（得分） ]
        attn = attn * self.rescale  # self.rescale 为可训练参数，格式为 [ heads , a , b ] 与 attn 格式为 [b,  heads ，dim_head， dim_head ] 进行 * (标量乘法运算)，将dim_head中的每一个值与a，b相乘，输出为 attn [b,  heads ，dim_head ， dim_head ]
        attn = attn.softmax(dim=-1)  # 通过 softmax 对 [b,  heads ，dim_head ， dim_head ] 最后一个维度进行归一化，将注意力得分转换为注意力权重，使得所有注意力权重之和为1。

        # 应用注意力权重到 v 上
        x = attn @ v  #attn：[b, heads ，dim_head ， dim_head ]与 v:[b,  heads ，dim_head， h * w ]相乘，结果为x：[b, heads ，dim_head， h * w ]
        x = x.permute(0, 3, 1, 2)  # x：[b ， h * w , heads ，dim_head]
        x = x.reshape(b, h * w, self.num_heads * self.dim_head)  #x：[b ， h * w , heads*dim_head]
        out_c = self.proj(x).view(b, h, w, c)  # 投影并重塑形状   proj为一个线性层，out_c：[b ， h ， w , dim_head]
        out_p = self.pos_emb(v_inp.reshape(b, h, w, c).permute(0, 3, 1, 2)).permute(0, 2, 3, 1)  # 位置编码  给v加上位置编码向量，最后[b ， h ， w , dim_head]
        out = out_c + out_p  # 将通道和位置结果相加

        return out  # [ b , h , w , c ]


# 前馈网络（FeedForward）层的定义
class FeedForward(nn.Module):  #FFN
    def __init__(self, dim, mult=4):  #dim为输入数据的维度
        super().__init__()
        self.net = nn.Sequential(  #nn.Sequential实现不同层的顺序连接
            nn.Conv2d(dim, dim * mult, 1, 1, bias=False),  #输入通道数为 dim ，输出通道数为 dim * mult,卷积核为 1x1 ，步长为1
            GELU(),
            nn.Conv2d(dim * mult, dim * mult, 3, 1, 1, bias=False, groups=dim * mult), #输入通道数为 dim * mult ，输出通道数为 dim * mult,卷积核为 3x3 ，步长为1 ，填充为 1
            #输入channel通道被分为groups个组，每个组独立地进行卷积操作。在这里，每个输入通道都有一个对应的3x3卷积核，不同的组之间不共享权重。
            GELU(),
            nn.Conv2d(dim * mult, dim, 1, 1, bias=False), #输入通道数为 dim * mult ，输出通道数为 dim ,卷积核为 1x1 ，步长为1
        )

    def forward(self, x):
        """
        x: [b, h, w, c]
        return out: [b, h, w, c]
        """
        out = self.net(x.permute(0, 3, 1, 2))  # 将输入从 [b, h, w, c] 转换为 [b, c, h, w]
        return out.permute(0, 2, 3, 1)  # 将输出从 [b, c, h, w] 转换为 [b, h, w, c]


# 多层多头自注意力块（MSAB）类的定义
class MSAB(nn.Module):  #SAB
    def __init__(self, dim, dim_head, heads, num_blocks):
        super().__init__()
        self.blocks = nn.ModuleList([])  #用于创建一个模块列表。与普通的Python列表不同，nn.ModuleList 能够识别并正确处理其中包含的 nn.Module 子类实例。

        # 创建包含给定数量块的模块列表
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([
                MS_MSA(dim=dim, dim_head=dim_head, heads=heads),  # 多尺度多头自注意力层
                PreNorm(dim, FeedForward(dim=dim))  # 前馈网络
            ]))

    def forward(self, x):
        """
        x: [b, c, h, w]
        return out: [b, c, h, w]
        """
        x = x.permute(0, 2, 3, 1)  # 将输入从 [b, c, h, w] 转换为 [b, h, w, c]

        for (attn, ff) in self.blocks:  #遍历self.blocks中的（自注意力层和前馈网络层）
            #attn表示MS_MSA层，ff表示PreNorm层
            x = attn(x) + x  # 应用自注意力层，“+x”残差连接
            x = ff(x) + x  # 应用前馈网络，“+x”残差连接

        out = x.permute(0, 3, 1, 2)  # 将输出从 [b, h, w, c] 转换为 [b, c, h, w]
        return out


# 主模型类 MST 的定义
class MST(nn.Module):  #SST
    def __init__(self, in_dim=31, out_dim=31, dim=31, stage=2, num_blocks=[2, 4, 4]):  #dim=31, stage=2, num_blocks=[1, 1, 1]
        super(MST, self).__init__()
        self.dim = dim  #输入特征的维度
        self.stage = stage  #模型的阶段数

        # 输入投影  embedding
        self.embedding = nn.Conv2d(in_dim, self.dim, 3, 1, 1, bias=False)  #输入特征从31维转为31维，卷积核大小为3x3，步长为1，填充为1

        # 编码器部分  从 32 升维至 32*8
        self.encoder_layers = nn.ModuleList([])
        dim_stage = dim  #用于表示当前阶段的特征维度

        for i in range(stage):  #循环stage个阶段
            self.encoder_layers.append(nn.ModuleList([
                MSAB(
                    dim=dim_stage, num_blocks=num_blocks[i], dim_head=dim, heads=dim_stage // dim),  #SAB   dim, dim_head, heads, num_blocks
                nn.Conv2d(dim_stage, dim_stage * 2, 4, 2, 1, bias=False),  #DownSample
            ]))
            dim_stage *= 2

        # 瓶颈部分
        self.bottleneck = MSAB(
            dim=dim_stage, dim_head=dim, heads=dim_stage // dim, num_blocks=num_blocks[-1])  #-1为最后一个数值

        # 解码器部分 从 31*8 降维至 32/8
        self.decoder_layers = nn.ModuleList([])
        for i in range(stage):
            self.decoder_layers.append(nn.ModuleList([
                nn.ConvTranspose2d(dim_stage, dim_stage // 2, stride=2, kernel_size=2, padding=0, output_padding=0),
                nn.Conv2d(dim_stage, dim_stage // 2, 1, 1, bias=False),
                MSAB(
                    dim=dim_stage // 2, num_blocks=num_blocks[stage - 1 - i], dim_head=dim,
                    heads=(dim_stage // 2) // dim),
            ]))
            dim_stage //= 2

        # 输出投影
        self.mapping = nn.Conv2d(self.dim, out_dim, 3, 1, 1, bias=False)

        # 激活函数
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.apply(self._init_weights)  #apply为内置函数，使用此函数来对数据对象进行批量处理

    # 初始化权重
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):  #isinstance() 函数来判断一个对象是否是一个已知的类型,
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    # 前向传播
    def forward(self, x):
        """
        x: [b, c, h, w]
        return out:[b, c, h, w]
        """

        # 输入投影
        fea = self.embedding(x)

        # 编码器部分
        fea_encoder = []
        for (MSAB, FeaDownSample) in self.encoder_layers:
            fea = MSAB(fea)
            fea_encoder.append(fea)
            fea = FeaDownSample(fea)

        # 瓶颈部分
        fea = self.bottleneck(fea)

        # 解码器部分
        for i, (FeaUpSample, Fution, LeWinBlcok) in enumerate(self.decoder_layers):  #enumerate函数获取当前元素的索引i  self.decoder_layers是一个列表，其中每个元素都是一个包含三个部分的元组，分别是FeaUpSample（特征上采样操作）、Fution（特征融合操作）和LeWinBlcok（可能是一个窗口注意力块或其他特征处理块）。
            fea = FeaUpSample(fea)
            fea = Fution(torch.cat([fea, fea_encoder[self.stage - 1 - i]], dim=1))  #torch.cat函数将当前上采样后的特征fea与fea_encoder列表中相应索引的特征（通过self.stage - 1 - i计算索引）沿着通道维度（dim=1）进行拼接。拼接后的特征然后传递给Fution模块进行特征融合，这通常用于合并来自不同层的特征信息。
            fea = LeWinBlcok(fea)

        # 输出映射
        out = self.mapping(fea) + x  #回复原始维度，并进行残差处理

        return out


# 扩展版 MST_Plus_Plus 模型的定义
class MST_Plus_Plus(nn.Module):
    def __init__(self, in_channels=3, out_channels=31, n_feat=31, stage=3):  #in_channels（输入通道数，默认为3），out_channels（输出通道数，默认为31），n_feat（特征图通道数，默认为31），stage（N1，默认为3）。
        super(MST_Plus_Plus, self).__init__()
        self.stage = stage
        self.conv_in = nn.Conv2d(in_channels, n_feat, kernel_size=3, padding=(3 - 1) // 2, bias=False)

        # 创建包含多层 MST 模块的序列
        modules_body = [MST(dim=31, stage=2, num_blocks=[1, 1, 1]) for _ in range(stage)]
        self.body = nn.Sequential(*modules_body)  #*为解包语法

        self.conv_out = nn.Conv2d(n_feat, out_channels, kernel_size=3, padding=(3 - 1) // 2, bias=False)

    # 前向传播
    def forward(self, x):
        """
        x: [b, c, h, w]
        return out:[b, c, h, w]
        """
        b, c, h_inp, w_inp = x.shape
        hb, wb = 8, 8
        pad_h = (hb - h_inp % hb) % hb
        pad_w = (wb - w_inp % wb) % wb
        x = F.pad(x, [0, pad_w, 0, pad_h], mode='reflect')  #这意味着填充的值是输入张量的边界值。填充的顺序是 [left, right, top, bottom]，在宽度方向上右侧填充 pad_w 个像素，在高度方向上下方填充 pad_h 个像素。
        x = self.conv_in(x)
        h = self.body(x)
        h = self.conv_out(h)
        h += x  #残差处理
        return h[:, :, :h_inp, :w_inp]