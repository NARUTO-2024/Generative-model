import einops
import torch
import torch as th
import torch.nn as nn

from ..ldm.modules.diffusionmodules.util import (
    conv_nd,
    linear,
    zero_module,
    timestep_embedding,
)

from einops import rearrange, repeat
from torchvision.utils import make_grid
from ..ldm.modules.attention import SpatialTransformer
from ..ldm.modules.diffusionmodules.openaimodel import UNetModel, TimestepEmbedSequential, ResBlock, Downsample, AttentionBlock
from ..ldm.models.diffusion.ddpm import LatentDiffusion
from ..ldm.util import log_txt_as_img, exists, instantiate_from_config
from ..ldm.models.diffusion.ddim import DDIMSampler


class ControlledUnetModel(UNetModel):  # ControlledUnetModel 继承自 UNetModel
    def forward(self, x, timesteps=None, context=None, control=None, only_mid_control=False, **kwargs):  #(x=x_noisy, timesteps=t, context=cond_txt, control=None, only_mid_control=self.only_mid_control)
        # 定义前向传播方法

        hs = []  # 用于存储每一层的输出特征图
        with torch.no_grad():  # 禁用梯度计算，节省显存，常用于评估或推理阶段
            # 获取时间步的嵌入
            t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
            # 获取时间步的嵌入并通过嵌入层进行处理
            emb = self.time_embed(t_emb)
            # 将输入张量转换为指定的数据类型
            h = x.type(self.dtype)

            # 输入块的前向传播
            for module in self.input_blocks:
                h = module(h, emb, context)  # 逐层处理输入，通过输入模块
                hs.append(h)  # 保存每一层的特征图

            # 中间块的前向传播
            h = self.middle_block(h, emb, context)

        # 如果有控制信号，则在当前特征图上添加控制信号
        if control is not None:
            h += control.pop()  # 从控制列表中取出并应用控制信号

        # 输出块的前向传播
        for i, module in enumerate(self.output_blocks):
            # 判断是否只使用中间控制或控制信号是否为 None
            if only_mid_control or control is None:
                h = torch.cat([h, hs.pop()], dim=1)  # 如果没有控制，拼接当前特征图和输入块的特征图
            else:
                h = torch.cat([h, hs.pop() + control.pop()], dim=1)  # 否则拼接当前特征图和控制信号

            h = module(h, emb, context)  # 通过输出模块处理拼接后的特征图

        h = h.type(x.dtype)  # 将输出特征图转换为与输入相同的数据类型
        return self.out(h)  # 返回最终的输出


class ControlNet(nn.Module):
    def __init__(
            self,
            image_size,
            in_channels,
            model_channels,
            hint_channels,
            num_res_blocks,
            attention_resolutions,
            dropout=0,
            channel_mult=(1, 2, 4, 8),
            conv_resample=True,
            dims=2,
            use_checkpoint=False,
            use_fp16=False,
            num_heads=-1,
            num_head_channels=-1,
            num_heads_upsample=-1,
            use_scale_shift_norm=False,
            resblock_updown=False,
            use_new_attention_order=False,
            use_spatial_transformer=False,  # custom transformer support
            transformer_depth=1,  # custom transformer support
            context_dim=None,  # custom transformer support
            n_embed=None,  # custom support for prediction of discrete ids into codebook of first stage vq model
            legacy=True,
            disable_self_attentions=None,
            num_attention_blocks=None,
            disable_middle_self_attn=False,
            use_linear_in_transformer=False,
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

        self.dims = dims
        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        if isinstance(num_res_blocks, int):
            self.num_res_blocks = len(channel_mult) * [num_res_blocks]
        else:
            if len(num_res_blocks) != len(channel_mult):
                raise ValueError("provide num_res_blocks either as an int (globally constant) or "
                                 "as a list/tuple (per-level) with the same length as channel_mult")
            self.num_res_blocks = num_res_blocks
        if disable_self_attentions is not None:
            # should be a list of booleans, indicating whether to disable self-attention in TransformerBlocks or not
            assert len(disable_self_attentions) == len(channel_mult)
        if num_attention_blocks is not None:
            assert len(num_attention_blocks) == len(self.num_res_blocks)
            assert all(map(lambda i: self.num_res_blocks[i] >= num_attention_blocks[i], range(len(num_attention_blocks))))
            print(f"Constructor of UNetModel received num_attention_blocks={num_attention_blocks}. "
                  f"This option has LESS priority than attention_resolutions {attention_resolutions}, "
                  f"i.e., in cases where num_attention_blocks[i] > 0 but 2**i not in attention_resolutions, "
                  f"attention will still not be set.")

        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
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

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1)
                )
            ]
        )
        self.zero_convs = nn.ModuleList([self.make_zero_conv(model_channels)])

        self.input_hint_block = TimestepEmbedSequential(
            conv_nd(dims, hint_channels, 16, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 16, 16, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 16, 32, 3, padding=1, stride=2),
            nn.SiLU(),
            conv_nd(dims, 32, 32, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 32, 96, 3, padding=1, stride=2),
            nn.SiLU(),
            conv_nd(dims, 96, 96, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 96, 256, 3, padding=1, stride=2),
            nn.SiLU(),
            zero_module(conv_nd(dims, 256, model_channels, 3, padding=1))
        )

        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for nr in range(self.num_res_blocks[level]):
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
                        # num_heads = 1
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    if exists(disable_self_attentions):
                        disabled_sa = disable_self_attentions[level]
                    else:
                        disabled_sa = False

                    if not exists(num_attention_blocks) or nr < num_attention_blocks[level]:
                        layers.append(
                            AttentionBlock(
                                ch,
                                use_checkpoint=use_checkpoint,
                                num_heads=num_heads,
                                num_head_channels=dim_head,
                                use_new_attention_order=use_new_attention_order,
                            ) if not use_spatial_transformer else SpatialTransformer(
                                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                                disable_self_attn=disabled_sa, use_linear=use_linear_in_transformer,
                                use_checkpoint=use_checkpoint
                            )
                        )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self.zero_convs.append(self.make_zero_conv(ch))
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
                self.zero_convs.append(self.make_zero_conv(ch))
                ds *= 2
                self._feature_size += ch

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        if legacy:
            # num_heads = 1
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
            ) if not use_spatial_transformer else SpatialTransformer(  # always uses a self-attn
                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                disable_self_attn=disable_middle_self_attn, use_linear=use_linear_in_transformer,
                use_checkpoint=use_checkpoint
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
        self.middle_block_out = self.make_zero_conv(ch)
        self._feature_size += ch

    def make_zero_conv(self, channels):
        return TimestepEmbedSequential(zero_module(conv_nd(self.dims, channels, channels, 1, padding=0)))

    def forward(self, x, hint, timesteps, context, **kwargs):
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)

        guided_hint = self.input_hint_block(hint, emb, context)

        outs = []

        h = x.type(self.dtype)
        for module, zero_conv in zip(self.input_blocks, self.zero_convs):
            if guided_hint is not None:
                h = module(h, emb, context)
                h += guided_hint
                guided_hint = None
            else:
                h = module(h, emb, context)
            outs.append(zero_conv(h, emb, context))

        h = self.middle_block(h, emb, context)
        outs.append(self.middle_block_out(h, emb, context))

        return outs


class ControlLDM(LatentDiffusion):

    """
    ControlLDM是通过控制模型增强的Latent Diffusion Model，用于条件图像生成。
    """

    def __init__(self, control_stage_config, control_key, only_mid_control, *args, **kwargs):
        # 初始化父类LatentDiffusion
        super().__init__(*args, **kwargs)

        # 实例化控制模型，使用从配置文件中提取的控制阶段配置
        self.control_model = instantiate_from_config(control_stage_config)  # target: cldm.cldm.ControlNet

        # 存储控制标签键
        self.control_key = control_key  #control_key: "hint"

        # 是否仅在中间阶段进行控制
        self.only_mid_control = only_mid_control  #only_mid_control: False

        # 初始化控制尺度
        self.control_scales = [1.0] * 13

    @torch.no_grad()
    def get_input(self, batch, k, bs=None, *args, **kwargs):
        # 获取输入数据，包括潜在变量和条件
        x, c = super().get_input(batch, self.first_stage_key, *args, **kwargs)

        # 获取控制信息
        control = batch[self.control_key]

        # 如果bs非空，则截取控制信息
        if bs is not None:
            control = control[:bs]

        # 将控制信息转换为适当的格式并移动到设备
        control = control.to(self.device)
        control = einops.rearrange(control, 'b h w c -> b c h w')  # 调整维度顺序
        control = control.to(memory_format=torch.contiguous_format).float()  # 将其转为连续格式

        return x, dict(c_crossattn=[c], c_concat=[control])  # 返回潜在变量和条件信息

    def apply_model(self, x_noisy, t, cond, *args, **kwargs):
        # 应用模型进行噪声预测
        assert isinstance(cond, dict)  # 确保条件是字典类型
        diffusion_model = self.model.diffusion_model  # 获取扩散模型  self.model.diffusion_model的结果是生成一个名为 ControlledUnetModel 的对象

        # 拼接条件文本
        cond_txt = torch.cat(cond['c_crossattn'], 1)

        # 根据条件拼接和控制信息应用扩散模型
        if cond['c_concat'] is None:
            # 没有控制条件的情况
            eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=None,
                                  only_mid_control=self.only_mid_control)
        else:
            # 使用控制模型进行预测
            control = self.control_model(x=x_noisy, hint=torch.cat(cond['c_concat'], 1), timesteps=t, context=cond_txt)
            control = [c * scale for c, scale in zip(control, self.control_scales)]  # 应用控制尺度
            eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=control,
                                  only_mid_control=self.only_mid_control)

        return eps  # 返回预测的噪声

    @torch.no_grad()
    def get_unconditional_conditioning(self, N):
        # 获取无条件的条件信息
        return self.get_learned_conditioning([""] * N)

    @torch.no_grad()
    def log_images(self, batch, N=4, n_row=2, sample=False, ddim_steps=50, ddim_eta=0.0, return_keys=None,
                   quantize_denoised=True, inpaint=True, plot_denoise_rows=False, plot_progressive_rows=True,
                   plot_diffusion_rows=False, unconditional_guidance_scale=9.0, unconditional_guidance_label=None,
                   use_ema_scope=True,
                   **kwargs):
        # 日志记录图像生成过程的图像
        use_ddim = ddim_steps is not None  # 判断是否使用DDIM采样

        log = dict()  # 初始化日志字典
        z, c = self.get_input(batch, self.first_stage_key, bs=N)  # 获取输入
        c_cat, c = c["c_concat"][0][:N], c["c_crossattn"][0][:N]  # 获取条件信息
        N = min(z.shape[0], N)  # 更新N的值
        n_row = min(z.shape[0], n_row)  # 更新n_row的值

        # 解码潜在变量生成重建图像
        log["reconstruction"] = self.decode_first_stage(z)
        log["control"] = c_cat * 2.0 - 1.0  # 对控制图像进行缩放
        log["conditioning"] = log_txt_as_img((512, 512), batch[self.cond_stage_key], size=16)  # 文本条件日志

        if plot_diffusion_rows:
            # 日志记录扩散过程的图像
            diffusion_row = list()
            z_start = z[:n_row]
            for t in range(self.num_timesteps):
                if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                    t = repeat(torch.tensor([t]), '1 -> b', b=n_row).to(self.device).long()  # 重复时间步数
                    noise = torch.randn_like(z_start)  # 生成随机噪声
                    z_noisy = self.q_sample(x_start=z_start, t=t, noise=noise)  # 采样带噪声的潜在变量
                    diffusion_row.append(self.decode_first_stage(z_noisy))  # 解码并记录

            # 生成扩散日志网格
            diffusion_row = torch.stack(diffusion_row)  # 叠加扩散图像
            diffusion_grid = rearrange(diffusion_row, 'n b c h w -> b n c h w')  # 调整维度顺序
            diffusion_grid = rearrange(diffusion_grid, 'b n c h w -> (b n) c h w')  # 展平维度
            diffusion_grid = make_grid(diffusion_grid, nrow=diffusion_row.shape[0])  # 创建网格
            log["diffusion_row"] = diffusion_grid  # 记录扩散行

        if sample:
            # 获取去噪样本
            samples, z_denoise_row = self.sample_log(cond={"c_concat": [c_cat], "c_crossattn": [c]},
                                                     batch_size=N, ddim=use_ddim,
                                                     ddim_steps=ddim_steps, eta=ddim_eta)
            x_samples = self.decode_first_stage(samples)  # 解码生成样本
            log["samples"] = x_samples  # 记录样本
            if plot_denoise_rows:
                # 记录去噪图像的网格
                denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
                log["denoise_row"] = denoise_grid

        if unconditional_guidance_scale > 1.0:
            # 应用无条件指导
            uc_cross = self.get_unconditional_conditioning(N)
            uc_cat = c_cat  # torch.zeros_like(c_cat)  # 这里可以根据需要选择
            uc_full = {"c_concat": [uc_cat], "c_crossattn": [uc_cross]}  # 整合无条件信息

            # 获取带无条件指导的样本
            samples_cfg, _ = self.sample_log(cond={"c_concat": [c_cat], "c_crossattn": [c]},
                                             batch_size=N, ddim=use_ddim,
                                             ddim_steps=ddim_steps, eta=ddim_eta,
                                             unconditional_guidance_scale=unconditional_guidance_scale,
                                             unconditional_conditioning=uc_full,
                                             )
            x_samples_cfg = self.decode_first_stage(samples_cfg)  # 解码样本
            log[f"samples_cfg_scale_{unconditional_guidance_scale:.2f}"] = x_samples_cfg  # 记录样本

        return log  # 返回日志字典

    @torch.no_grad()
    def sample_log(self, cond, batch_size, ddim, ddim_steps, **kwargs):
        # 进行抽样并记录日志
        ddim_sampler = DDIMSampler(self)  # 创建DDIM采样器
        b, c, h, w = cond["c_concat"][0].shape  # 获取条件维度
        shape = (self.channels, h // 8, w // 8)  # 计算输出形状
        samples, intermediates = ddim_sampler.sample(ddim_steps, batch_size, shape, cond, verbose=False, **kwargs)
        return samples, intermediates  # 返回样本和中间结果

    def configure_optimizers(self):
        # 配置优化器
        lr = self.learning_rate  # 获取学习率
        params = list(self.control_model.parameters())  # 获取控制模型参数
        if not self.sd_locked:
            params += list(self.model.diffusion_model.output_blocks.parameters())  # 添加扩散模型参数
            params += list(self.model.diffusion_model.out.parameters())  # 添加扩散模型输出参数

        # 创建AdamW优化器
        opt = torch.optim.AdamW(params, lr=lr)
        return opt  # 返回优化器

    def low_vram_shift(self, is_diffusing):
        # 根据是否正在扩散，调整模型位置以节省显存
        if is_diffusing:
            self.model = self.model.cuda()  # 将模型移动到GPU
            self.control_model = self.control_model.cuda()  # 将控制模型移动到GPU
            self.first_stage_model = self.first_stage_model.cpu()  # 将第一阶段模型移回CPU
            self.cond_stage_model = self.cond_stage_model.cpu()  # 将条件阶段模型移回CPU
        else:
            self.model = self.model.cpu()  # 将模型移回CPU
            self.control_model = self.control_model.cpu()  # 将控制模型移回CPU
            self.first_stage_model = self.first_stage_model.cuda()  # 将第一阶段模型移回GPU
            self.cond_stage_model = self.cond_stage_model.cuda()  # 将条件阶段模型移回GPU