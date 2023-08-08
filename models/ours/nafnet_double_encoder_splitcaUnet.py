import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import math
from abc import abstractmethod

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


class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None


class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class CondNAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel,
                               kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c,
                               kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
        # self.sca_avg = nn.Sequential(
        #     nn.AdaptiveAvgPool2d(1),
        #     nn.Conv2d(in_channels=dw_channel // 4, out_channels=dw_channel // 4, kernel_size=1, padding=0, stride=1,
        #               groups=1, bias=True),
        # )
        # self.sca_max = nn.Sequential(
        #     nn.AdaptiveMaxPool2d(1),
        #     nn.Conv2d(in_channels=dw_channel // 4, out_channels=dw_channel // 4, kernel_size=1, padding=0, stride=1,
        #               groups=1, bias=True),
        # )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel,
                               kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c,
                               kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(
            drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(
            drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros(
            (1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        # x_avg, x_max = x.chunk(2, dim=1)
        # x_avg = self.sca_avg(x_avg)*x_avg
        # x_max = self.sca_max(x_max)*x_max
        # x = torch.cat([x_avg, x_max], dim=1)
        x = self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma

class NAFBlock(EmbedBlock):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel,
                               kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c,
                               kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        # Simplified Channel Attention
        # self.sca = nn.Sequential(
        #     nn.AdaptiveAvgPool2d(1),
        #     nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
        #               groups=1, bias=True),
        # )
        self.sca_avg = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 4, out_channels=dw_channel // 4, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
        self.sca_max = nn.Sequential(
            nn.AdaptiveMaxPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 4, out_channels=dw_channel // 4, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel,
                               kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c,
                               kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(
            drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(
            drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros(
            (1, c, 1, 1)), requires_grad=True)
        self.time_emb = nn.Sequential(
            nn.SiLU(),
            nn.Linear(256, c),
        )

    def forward(self, inp, t):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x_avg, x_max = x.chunk(2, dim=1)
        x_avg = self.sca_avg(x_avg)*x_avg
        x_max = self.sca_max(x_max)*x_max
        x = torch.cat([x_avg, x_max], dim=1)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        y = y+self.time_emb(t)[..., None, None]

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma


class UNet(nn.Module):

    def __init__(
        self,
        img_channel=3,
        width=64,
        middle_blk_num=1,
        enc_blk_nums=[1, 1, 1, 1],
        dec_blk_nums=[1, 1, 1, 1],
    ):
        super().__init__()

        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                               bias=True)
        self.cond_intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                               bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=3, kernel_size=3, padding=1, stride=1, groups=1,
                                bias=True)
        # self.inp_ending = nn.Conv2d(in_channels=img_channel, out_channels=3, kernel_size=3, padding=1, stride=1, groups=1,
        #                             bias=True)

        self.encoders = nn.ModuleList()
        self.cond_encoders = nn.ModuleList()

        self.decoders = nn.ModuleList()

        self.middle_blks = nn.ModuleList()

        self.ups = nn.ModuleList()

        self.downs = nn.ModuleList()
        self.cond_downs = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                EmbedSequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )
            self.cond_encoders.append(
                nn.Sequential(
                    *[CondNAFBlock(chan) for _ in range(num)]
                )
            )
            self.downs.append(
                nn.Conv2d(chan, 2*chan, 2, 2)
            )
            self.cond_downs.append(
                nn.Conv2d(chan, 2*chan, 2, 2)
            )
            chan = chan * 2

        self.middle_blks = \
            EmbedSequential(
                *[NAFBlock(chan) for _ in range(middle_blk_num)]
            )

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                EmbedSequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )

        self.padder_size = 2 ** len(self.encoders)
        self.emb = partial(gamma_embedding, dim=64)
        self.map = nn.Sequential(
            nn.Linear(64, 256),
            nn.SiLU(),
            nn.Linear(256, 256),
        )

    def forward(self, inp, gammas):
        t = self.map(self.emb(gammas.view(-1, )))
        inp = self.check_image_size(inp)

        x1, x2, x3, x = inp.chunk(4, dim=1)
        cond = torch.stack([x1, x2, x3], dim=1)
        b, n, c, h, w = cond.shape
        cond = cond.view(b*n, c, h, w)
        x = self.intro(x)
        cond = self.cond_intro(cond)


        encs = []

        for encoder, down, cond_encoder, cond_down in zip(self.encoders, self.downs, self.cond_encoders, self.cond_downs):
            x = encoder(x, t)
            cond = cond_encoder(cond)
            b, c, h, w = cond.shape
            tmp_cond = cond.view(b//3, 3, c, h, w).sum(dim=1)
            x = x + tmp_cond
            encs.append(x)
            x = down(x)
            cond = cond_down(cond)

        x = self.middle_blks(x, t)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x, t)

        x = self.ending(x)
        # x = x + self.inp_ending(inp)
    

        return x

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h %
                     self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w %
                     self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x


if __name__ == '__main__':
    # img_channel = 5
    # width = 32

    # # enc_blks = [2, 2, 4, 8]
    # # middle_blk_num = 12
    # # dec_blks = [2, 2, 2, 2]

    # enc_blks = [1, 1, 1, 28]
    # middle_blk_num = 1
    # dec_blks = [1, 1, 1, 1]

    # net = ours(
    #     img_channel=img_channel,
    #     width=width,
    #     middle_blk_num=middle_blk_num,
    #     enc_blk_nums=enc_blks,
    #     dec_blk_nums=dec_blks,
    # )

    # inp_shape = (2, 5, 64, 64)

    # out_shape = net(torch.Tensor(*inp_shape)).shape

    # print(out_shape)
    # print(UNet())
    print(UNet()(*(torch.Tensor(1, 12, 256, 256), torch.ones(1,))).shape)
    from thop import profile
    from thop import clever_format
    net = UNet()
    flops, params = profile(net, inputs=(torch.Tensor(1, 12, 256, 256), torch.ones(1,)))
    flops, params = clever_format([flops, params], "%.3f")
    print(flops, params) # 154.582G 36.579M