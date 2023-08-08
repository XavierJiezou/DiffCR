import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import abstractmethod
from timm.models.layers import DropPath
from functools import partial
from mmcv.cnn import build_norm_layer
from mmcv.cnn import build_conv_layer
from mmcv.cnn.bricks.registry import NORM_LAYERS


act_module = nn.GELU
BaseBlock = None
ls_init_value = 1e-6


class EmbedBlock(nn.Module):
    """
    Any module where forward() takes embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` embeddings.
        """


class EmbedSequential(nn.Sequential, EmbedBlock):
    """
    A sequential module that passes embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, EmbedBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


class LayerNorm(nn.Module):

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first", "channels_first_v2"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x
        elif self.data_format == "channels_first_v2":
            return F.layer_norm(x.permute(0, 2, 3, 1), self.normalized_shape, self.weight, self.bias, self.eps).permute(0, 3, 1, 2)

# 通道注意力模块


class ChannelAttention(nn.Module):
    def __init__(self, in_channels) -> None:
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels*4, bias=False),
            act_module(),
            nn.Linear(in_channels*4, in_channels, bias=False),
        )

        self.alpha = nn.Parameter(torch.ones(
            (in_channels)), requires_grad=True)
        self.betas = nn.Parameter(torch.ones(
            (in_channels)), requires_grad=True)

    def forward(self, x):
        b, c, _, _ = x.size()
        avg_out = self.alpha*self.avg_pool(x).view(b, c)
        max_out = self.betas*self.max_pool(x).view(b, c)
        out = self.fc(avg_out+max_out).view(b, c, 1, 1)
        return x*out.expand_as(x)


class Block(EmbedBlock):

    def __init__(self, dim, time_emb_dim, drop_path=0., norm_cfg=None, **kwargs):
        super().__init__()

        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7,
                                padding=3, groups=dim)  # depthwise conv
        self.norm = build_norm_layer(norm_cfg, dim)[1]
        # pointwise/1x1 convs, implemented with linear layers
        # self.pwconv1 = nn.Linear(dim, 4 * dim)
        # self.act = act_module()
        # self.pwconv2 = nn.Linear(4 * dim, dim)
        self.ca = ChannelAttention(dim)
        self.gamma = nn.Parameter(
            ls_init_value * torch.ones((1, dim, 1, 1)), requires_grad=True) if ls_init_value > 0 else None
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.time_emb = nn.Sequential(
            act_module(),
            nn.Linear(time_emb_dim, dim),
        )

    def forward(self, x, t):
        input = x
        x = self.dwconv(x) + self.time_emb(t)[..., None, None]
        x = self.norm(x)    # input (N, C, *)

        # x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.ca(x)
        # x = self.pwconv1(x)
        # x = self.act(x)
        # x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        # x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class DBlock(nn.Module):

    def __init__(self, dim, drop_path=0., dilation=3, norm_cfg=None, **kwargs):
        super().__init__()

        self.dwconv1 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=7,
                      padding=3, dilation=1, groups=dim),
            build_norm_layer(norm_cfg, dim)[1],
            act_module())

        self.dwconv2 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=7, padding=3 *
                      dilation, dilation=dilation, groups=dim),
            build_norm_layer(norm_cfg, dim)[1],
            act_module())

        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = act_module()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(
            ls_init_value * torch.ones((dim)), requires_grad=True) if ls_init_value > 0 else None
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):

        input = x
        x = self.dwconv1(x) + x
        x = self.dwconv2(x) + x

        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class Bottleneck(nn.Module):

    def __init__(self, dim, drop_path=0., norm_cfg=None, **kwargs):
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Sequential(
            build_conv_layer(None, dim, dim, kernel_size=3,
                             stride=1, padding=1, bias=False),
            build_norm_layer(norm_cfg, dim)[1],
            act_module(),
        )

        self.conv2 = nn.Sequential(
            build_conv_layer(None, dim, dim * 4, kernel_size=1,
                             stride=1, bias=False),
            build_norm_layer(norm_cfg, dim * 4)[1],
            act_module(),
        )

        self.conv3 = nn.Sequential(
            build_conv_layer(None, dim * 4, dim, kernel_size=1, bias=False),
            build_norm_layer(norm_cfg, dim)[1],
        )

        self.act = act_module()
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.act(self.drop_path(x) + input)
        return x


class DBottleneck(nn.Module):

    def __init__(self, dim, drop_path=0., dilation=3, norm_cfg=None, **kwargs):
        super(DBottleneck, self).__init__()

        self.conv1 = nn.Sequential(
            build_conv_layer(None, dim, dim, kernel_size=3,
                             stride=1, padding=1, bias=False),
            build_norm_layer(norm_cfg, dim)[1],
            act_module(),
        )

        self.dwconv1 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=7, padding=3 *
                      dilation, dilation=dilation, groups=dim),
            build_norm_layer(norm_cfg, dim)[1],
            act_module())

        self.conv2 = nn.Sequential(
            build_conv_layer(None, dim, dim * 4, kernel_size=1,
                             stride=1, bias=False),
            build_norm_layer(norm_cfg, dim * 4)[1],
            act_module(),
        )

        self.conv3 = nn.Sequential(
            build_conv_layer(None, dim * 4, dim, kernel_size=1, bias=False),
            build_norm_layer(norm_cfg, dim)[1],
        )

        self.act = act_module()
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.conv1(x) + x
        x = self.dwconv1(x) + x
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.act(self.drop_path(x) + input)
        return x


class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4):
        super().__init__()

        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_first")

        self.fc1 = nn.Conv2d(dim, int(dim * mlp_ratio), 1)
        self.pos = nn.Conv2d(int(dim * mlp_ratio), int(dim * mlp_ratio),
                             3, padding=1, groups=int(dim * mlp_ratio))
        self.fc2 = nn.Conv2d(int(dim * mlp_ratio), dim, 1)
        self.act = act_module()

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = x + self.act(self.pos(x))
        x = self.fc2(x)

        return x


class ConvMod(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_first")
        self.a = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            act_module(),
            nn.Conv2d(dim, dim, 11, padding=5, groups=dim)  # TODO
        )

        self.v = nn.Conv2d(dim, dim, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.norm(x)
        a = self.a(x)  # A=QK^T
        x = a * self.v(x)
        x = self.proj(x)

        return x


class Conv2Former(nn.Module):
    def __init__(self, dim, mlp_ratio=4., drop_path=0.):
        super().__init__()

        self.attn = ConvMod(dim)
        self.mlp = MLP(dim, mlp_ratio)
        layer_scale_init_value = 1e-6
        self.layer_scale_1 = nn.Parameter(  # a
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(  # b
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = x + \
            self.drop_path(
                self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.attn(x))
        x = x + \
            self.drop_path(
                self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(x))
        return x


class Encoder(nn.Module):

    def __init__(self, dims=[96, 192, 384, 768], blocks=[1, 1, 1, 1], time_emb_dim=512, dp_rates=0., norm_cfg=None):
        super().__init__()

        assert isinstance(dp_rates, list)
        cum_sum = np.array([0] + blocks[:-1]).cumsum()

        self.encoder = nn.ModuleList([
            EmbedSequential(*[BaseBlock(dims[0], time_emb_dim, dp_rates[_],
                                        norm_cfg=norm_cfg, widx=_) for _ in range(blocks[0])]),

            EmbedSequential(*[BaseBlock(dims[1], time_emb_dim, dp_rates[cum_sum[1]+_],
                                        norm_cfg=norm_cfg, widx=_) for _ in range(blocks[1])]),
            EmbedSequential(*[BaseBlock(dims[2], time_emb_dim, dp_rates[cum_sum[2]+_],
                                        norm_cfg=norm_cfg, widx=_) for _ in range(blocks[2])]),
            EmbedSequential(*[BaseBlock(dims[3], time_emb_dim, dp_rates[cum_sum[3]+_],
                                        norm_cfg=norm_cfg, widx=_) for _ in range(blocks[3])]),
            EmbedSequential(*[BaseBlock(dims[4], time_emb_dim, dp_rates[cum_sum[4]+_],
                                        norm_cfg=norm_cfg, widx=_) for _ in range(blocks[4])]),
        ])

        self.encoder_downsample = nn.ModuleList([
            nn.Sequential(nn.Conv2d(dims[0], dims[1], kernel_size=2, stride=2), build_norm_layer(
                norm_cfg, dims[1])[1]),
            nn.Sequential(nn.Conv2d(dims[1], dims[2], kernel_size=2, stride=2), build_norm_layer(
                norm_cfg, dims[2])[1]),
            nn.Sequential(nn.Conv2d(dims[2], dims[3], kernel_size=2, stride=2), build_norm_layer(
                norm_cfg, dims[3])[1]),
            nn.Sequential(nn.Conv2d(dims[3], dims[4], kernel_size=2, stride=2), build_norm_layer(
                norm_cfg, dims[4])[1]),
        ])

        self.attention = Conv2Former(dims[4], drop_path=dp_rates[-1])

        self.pooling_conv = nn.ModuleList([
            nn.Sequential(nn.Conv2d(dims[0], dims[4], kernel_size=1, stride=1), build_norm_layer(
                norm_cfg, dims[4])[1]),
            nn.Sequential(nn.Conv2d(dims[1], dims[4], kernel_size=1, stride=1), build_norm_layer(
                norm_cfg, dims[4])[1]),
            nn.Sequential(nn.Conv2d(dims[2], dims[4], kernel_size=1, stride=1), build_norm_layer(
                norm_cfg, dims[4])[1]),
            nn.Sequential(nn.Conv2d(dims[3], dims[4], kernel_size=1, stride=1), build_norm_layer(
                norm_cfg, dims[4])[1]),
        ])

    def forward(self, x, t):
        if isinstance(x, tuple):
            x = x[0]
        c3 = self.encoder[0](x, t)
        c4 = self.encoder[1](self.encoder_downsample[0](c3), t)
        c5 = self.encoder[2](self.encoder_downsample[1](c4), t)
        c6 = self.encoder[3](self.encoder_downsample[2](c5), t)
        c7 = self.encoder[4](self.encoder_downsample[3](c6), t)
        # import pdb; pdb.set_trace()
        global_f = F.adaptive_avg_pool2d(self.pooling_conv[0](c3), output_size=(c7.shape[-2], c7.shape[-1])) \
            + F.adaptive_avg_pool2d(self.pooling_conv[1](
                c4), output_size=(c7.shape[-2], c7.shape[-1])) + F.adaptive_avg_pool2d(self.pooling_conv[2](c5), output_size=(c7.shape[-2], c7.shape[-1])) + F.adaptive_avg_pool2d(self.pooling_conv[3](c6), output_size=(c7.shape[-2], c7.shape[-1]))+c7
        global_f = self.attention(global_f)
        return c3, c4, c5, c6, c7, global_f


class LAlayerUpsample(nn.Module):
    def __init__(self, inp: int, oup: int, kernel: int = 1, norm_cfg=None) -> None:
        super().__init__()
        norm = build_norm_layer(norm_cfg, oup)[1]
        groups = 1
        if inp == oup:
            groups = inp
        self.local_embedding = nn.Sequential(
            nn.Conv2d(oup, oup, kernel, groups=oup,
                      padding=int((kernel - 1) / 2), bias=False),
            norm
        )
        self.global_embedding = nn.Sequential(
            nn.Conv2d(inp, oup, kernel, padding=int(
                (kernel - 1) / 2), bias=False),
            norm
        )
        self.global_act = nn.Sequential(
            nn.Conv2d(inp, oup, kernel, padding=int(
                (kernel - 1) / 2), bias=False),
            norm
        )
        self.act = nn.Sigmoid()

    def forward(self, x_l, x_g):
        """
        x_g: global features
        x_l: local features
        """
        B, N, H, W = x_l.shape
        local_feat = self.local_embedding(x_l)

        global_act = self.global_act(x_g)
        sig_act = F.interpolate(self.act(global_act), size=(H, W))

        global_feat = self.global_embedding(x_g)
        global_feat = F.interpolate(global_feat, size=(H, W))

        out = local_feat * sig_act + global_feat
        return out


class LALayerG(nn.Module):
    def __init__(self, inp: int, oup: int, kernel: int = 1, norm_cfg=None) -> None:
        super().__init__()
        norm = build_norm_layer(norm_cfg, inp)[1]
        groups = 1
        if inp == oup:
            groups = inp
        self.local_embedding = nn.Sequential(
            nn.Conv2d(inp, inp, kernel, groups=inp,
                      padding=int((kernel - 1) / 2), bias=False),
            norm
        )
        self.global_embedding = nn.Sequential(
            nn.Conv2d(oup, inp, kernel, groups=groups,
                      padding=int((kernel - 1) / 2), bias=False),
            norm
        )
        self.global_act = nn.Sequential(
            nn.Conv2d(oup, inp, kernel, groups=groups,
                      padding=int((kernel - 1) / 2), bias=False),
            norm
        )
        self.act = nn.Sigmoid()

    def forward(self, x_l, x_g):
        """
        x_g: global features
        x_l: local features
        """
        B, N, H, W = x_l.shape
        # import pdb; pdb.set_trace()
        local_feat = self.local_embedding(x_l)
        global_act = self.global_act(x_g)
        sig_act = F.interpolate(self.act(global_act), size=(H, W))

        global_feat = self.global_embedding(x_g)
        global_feat = F.interpolate(global_feat, size=(H, W))

        out = local_feat * sig_act + global_feat
        return out


class Decoder(nn.Module):

    def __init__(self, dims=[96, 192, 384, 768], blocks=[1, 1, 1, 1], dp_rates=0., norm_cfg=None):
        super().__init__()

        self.decoder_conv = nn.ModuleList([
            LALayerG(dims[0], dims[0], norm_cfg=norm_cfg),
            LALayerG(dims[1], dims[0], norm_cfg=norm_cfg),
            LALayerG(dims[2], dims[0], norm_cfg=norm_cfg),
            LALayerG(dims[3], dims[0], norm_cfg=norm_cfg),
            LALayerG(dims[4], dims[0], norm_cfg=norm_cfg),
        ])

        self.decoder_up = nn.ModuleList([
            LAlayerUpsample(dims[0], dims[1], norm_cfg=norm_cfg),
            LAlayerUpsample(dims[1], dims[2], norm_cfg=norm_cfg),
            LAlayerUpsample(dims[2], dims[3], norm_cfg=norm_cfg),
            LAlayerUpsample(dims[3], dims[4], norm_cfg=norm_cfg),
        ])

    def forward(self, x):
        c3, c4, c5, c6, c7, global_f = x
        c7 = self.decoder_conv[0](c7, global_f)
        c6 = self.decoder_conv[1](c6, global_f)
        c5 = self.decoder_conv[2](c5, global_f)
        c4 = self.decoder_conv[3](c4, global_f)
        c3 = self.decoder_conv[4](c3, global_f)

        c6 = self.decoder_up[0](c6, c7)
        c5 = self.decoder_up[1](c5, c6)
        c4 = self.decoder_up[2](c4, c5)
        c3 = self.decoder_up[3](c3, c4)
        return c3, c4, c5, c6, c7


class Ours(nn.Module):
    def __init__(
        self,
        inp_channels=3,
        out_channels=3,
        encoder_dims=[64, 128, 256, 512, 1024],
        decoder_dims=[1024, 512, 256, 128, 64],
        encoder_blocks=[1, 1, 1, 1, 1],
        decoder_blocks=[1, 1, 1, 1, 1],
        drop_path_rate=0.1,
        norm_cfg=dict(type='LN', eps=1e-6, data_format="channels_first"),
        act_type='silu',
    ) -> None:

        NORM_LAYERS.register_module('LN', force=True, module=LayerNorm)
        global act_module, BaseBlock, ls_init_value
        act_module = {'gelu': nn.GELU, 'relu': nn.ReLU,
                      'silu': nn.SiLU}.get(act_type, None)
        BaseBlock = globals().get("Block", None)
        ls_init_value = 1e-6
        dp_rates = [i.item() for i in torch.linspace(
            0, drop_path_rate, sum(encoder_blocks))]

        super().__init__()

        self.inp = nn.Sequential(
            nn.Conv2d(inp_channels, encoder_dims[0]//2, 3, 1, 1),
            build_norm_layer(norm_cfg, encoder_dims[0]//2)[1],
            act_module(),
            nn.Conv2d(encoder_dims[0]//2, encoder_dims[0], 3, 1, 1),
            build_norm_layer(norm_cfg, encoder_dims[0])[1],
            act_module(),
        )

        self.encoder = Encoder(
            dims=encoder_dims,
            blocks=encoder_blocks,
            dp_rates=dp_rates,
            norm_cfg=norm_cfg,
        )

        self.decoder = Decoder(
            dims=decoder_dims,
            blocks=decoder_blocks,
            dp_rates=dp_rates,
            norm_cfg=norm_cfg,
        )

        self.out = nn.Sequential(
            build_norm_layer(norm_cfg, decoder_dims[-1])[1],
            act_module(),
            nn.Conv2d(decoder_dims[-1], out_channels, 3, 1, 1),
        )

    def forward(self, x):
        x = self.inp(x)
        x = self.encoder(x)
        # print(x[0].shape) # torch.Size([2, 64, 256, 256])
        # print(x[1].shape) # torch.Size([2, 128, 128, 128])
        # print(x[2].shape) # torch.Size([2, 256, 64, 64])
        # print(x[3].shape) # torch.Size([2, 512, 32, 32])
        # print(x[4].shape) # torch.Size([2, 512, 32, 32])
        x = self.decoder(x)
        # print(x[0].shape) # torch.Size([2, 64, 256, 256])
        # print(x[1].shape) # torch.Size([2, 128, 128, 128])
        # print(x[2].shape) # torch.Size([2, 256, 64, 64])
        # print(x[3].shape) # torch.Size([2, 512, 32, 32])
        return self.out(x[0])


def gamma_embedding(gammas, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    :param gammas: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0,
                                             end=half, dtype=torch.float32) / half
    ).to(device=gammas.device)
    args = gammas[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat(
            [embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class UNet(nn.Module):
    def __init__(
        self,
        inp_channels=12,
        out_channels=3,
        encoder_dims=[64, 128, 256, 512],
        decoder_dims=[512, 256, 128, 64],
        encoder_blocks=[1, 1, 1, 1],
        decoder_blocks=[1, 1, 1, 1],
        drop_path_rate=0.1,
        norm_type="ln",
        act_type='silu',
    ) -> None:

        NORM_LAYERS.register_module('ln', force=True, module=LayerNorm)
        global act_module, BaseBlock, ls_init_value
        act_module = {'gelu': nn.GELU, 'relu': nn.ReLU,
                      'silu': nn.SiLU}.get(act_type, None)
        BaseBlock = globals().get("Block", None)
        ls_init_value = 1e-6
        dp_rates = [i.item() for i in torch.linspace(
            0, drop_path_rate, sum(encoder_blocks))]
        norm_cfg = {
            "ln": dict(type='ln', eps=1e-6, data_format="channels_first"),
            "bn": dict(type='bn'),
        }.get(norm_type, None)

        super().__init__()

        self.inp = nn.Sequential(
            nn.Conv2d(inp_channels, encoder_dims[0]//2, 3, 1, 1),
            build_norm_layer(norm_cfg, encoder_dims[0]//2)[1],
            act_module(),
            nn.Conv2d(encoder_dims[0]//2, encoder_dims[0], 3, 1, 1),
            build_norm_layer(norm_cfg, encoder_dims[0])[1],
            act_module(),
        )

        self.emb = partial(gamma_embedding, dim=encoder_dims[0])
        self.map = nn.Sequential(
            nn.Linear(encoder_dims[0], encoder_dims[-1]),
            act_module(),
            nn.Linear(encoder_dims[-1], encoder_dims[-1]),
        )

        self.encoder = Encoder(
            dims=encoder_dims,
            blocks=encoder_blocks,
            time_emb_dim=encoder_dims[-1],
            dp_rates=dp_rates,
            norm_cfg=norm_cfg,
        )

        self.decoder = Decoder(
            dims=decoder_dims,
            blocks=decoder_blocks,
            dp_rates=dp_rates,
            norm_cfg=norm_cfg,
        )

        self.out = nn.Sequential(
            build_norm_layer(norm_cfg, decoder_dims[-1])[1],
            act_module(),
            nn.Conv2d(decoder_dims[-1], out_channels, 3, 1, 1),
        )

    def forward(self, x, gammas):
        x = self.inp(x)
        t = self.map(self.emb(gammas.view(-1, )))
        x = self.encoder(x, t)
        x = self.decoder(x)
        return self.out(x[0])


if __name__ == '__main__':
    inp = (torch.Tensor(2, 12, 256, 256), torch.ones(2,))
    net = UNet(
        inp_channels=12,
        out_channels=3,
        encoder_dims=[64, 128, 256, 512, 1024],
        decoder_dims=[1024, 512, 256, 128, 64],
        encoder_blocks=[1, 1, 1, 1, 1],
        decoder_blocks=[1, 1, 1, 1, 1],
        drop_path_rate=0.1,
        norm_type="ln",
        act_type='silu',
    )
    out = net(*inp)
    print(out.shape)
