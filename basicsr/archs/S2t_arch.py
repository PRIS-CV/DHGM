import basicsr.archs.common as common
from ldm.ddpm import DDPM
# import basicsr.archs.attention as attention
import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers
from basicsr.utils.registry import ARCH_REGISTRY
from einops import rearrange
from basicsr.ops.guided_filter.guide_filter import ConvGuidedFilter
from basicsr.ops.dct.dct1d import dct, high_pass, idct


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

        self.kernel = nn.Sequential(
            nn.Linear(256, dim*2, bias=False),
        )

    def forward(self, x, k_v):
        b,c,h,w = x.shape
        k_v=self.kernel(k_v).view(-1,c*2,1,1)
        k_v1,k_v2=k_v.chunk(2, dim=1)
        x = x*k_v1+k_v2

        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        self.kernel_p = nn.Linear(256, dim, bias=False)
        self.kernel = nn.Linear(256, dim*2, bias=False)
        self.guide_filter = ConvGuidedFilter(dim=dim, radius=12)

    def forward(self, x, p, guide_p=None, dct_p=None):
        b,c,h,w = x.shape

        if c == 64:
            p1,p2,p3,p4 = p.view(-1,c*4,1,1).chunk(4, dim=1)
            if guide_p != None:
                q_pre = self.kernel_p(guide_p).view(-1,c,1,1)
                x = self.guide_filter(p1*x, x+(p2+p3+p4), q_pre) + (p1 * x + p2 * x + p3 * x + p4 * x  + (p1+p2+p3+p4))
            elif dct_p != None:
                q_pre = self.kernel_p(dct_p).view(-1,c,1,1)
                x = x*p1 + (p2+p3+p4) + (p2 * x + p3 * x + p4 * x + p1)
        elif c == 128:
            q_pre = self.kernel_p(p).view(-1,c,1,1)
            p1,p2 = p.view(-1,c*2,1,1).chunk(2, dim=1)
            x = (p1 * x + p2) + (p2 * x + p1)
        elif c == 256 or c == 384:
            q_pre = self.kernel_p(p).view(-1,c,1,1)
            p1,p2=self.kernel(p).view(-1,c*2,1,1).chunk(2, dim=1)
            x = (p1 * x + p2) + (p2 * x + p1)

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)
        q = q * q_pre

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, y, guide_p=None, dct_p=None):
        x = y[0]
        k_v = y[1]
        if guide_p != None:
            x = x + self.attn(self.norm1(x), k_v, guide_p=guide_p)
        elif dct_p != None:
            x = x + self.attn(self.norm1(x), k_v, dct_p=dct_p)
        else:
            x = x + self.attn(self.norm1(x), k_v)
        x = x + self.ffn(self.norm2(x), k_v)

        return [x, k_v]


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x


class BasicLayer(nn.Module):
    def __init__(self, dim, out_dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type, num_blocks):
        super().__init__()

        # build blocks
        self.blocks = nn.ModuleList(
            [TransformerBlock(dim=dim, num_heads=num_heads, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in
             range(num_blocks)])

        self.conv = nn.Conv2d(dim, out_dim, 3, 1, 1)

    def forward(self, y, guide_p=None, dct_p=None):
        x = y[0]
        res = x
        prior = y[1]
        if guide_p != None:
            for blk in self.blocks:
                x, k_v = blk([x, prior], guide_p=guide_p)
        elif dct_p != None:
            for blk in self.blocks:
                x, k_v = blk([x, prior], dct_p=dct_p)
        else:
            for blk in self.blocks:
                x, k_v = blk([x, prior])

        x = self.conv(res + x)
        return [x, k_v]


class IR(nn.Module):
    def __init__(self,
        inp_channels=3,
        out_channels=3,
        scale=2,
        dim = 64,
        num_blocks = [6,6,6,6],
        heads = [1,2,4,8],
        ffn_expansion_factor = 2,
        bias = False,
        LayerNorm_type = 'WithBias',   ## Other option 'BiasFree'
        ):

        super(IR, self).__init__()
        self.scale=scale
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.layers1 = nn.ModuleList()
        for i_layer in range(len(num_blocks)//2):
            layer = BasicLayer(dim=dim,
                               out_dim=dim,
                               num_heads=heads[i_layer],
                               ffn_expansion_factor=ffn_expansion_factor,
                               bias=bias,
                               LayerNorm_type=LayerNorm_type,
                               num_blocks=num_blocks[i_layer])
            self.layers1.append(layer)

        self.layers2 = nn.ModuleList()
        for i_layer in range(len(num_blocks)//2):
            layer = BasicLayer(dim=dim,
                               out_dim=dim,
                               num_heads=heads[i_layer],
                               ffn_expansion_factor=ffn_expansion_factor,
                               bias=bias,
                               LayerNorm_type=LayerNorm_type,
                               num_blocks=num_blocks[i_layer])
            self.layers2.append(layer)

        self.expand = (common.default_conv(dim, dim*2, 3))
        self.layers_large = nn.ModuleList()
        layer = BasicLayer(dim=dim*2,
                        out_dim=dim*4,
                        num_heads=2,
                        ffn_expansion_factor=ffn_expansion_factor,
                        bias=bias,
                        LayerNorm_type=LayerNorm_type,
                        num_blocks=6)
        self.layers_large.append(layer)
        layer = BasicLayer(dim=dim*4,
                        out_dim=dim*1,
                        num_heads=4,
                        ffn_expansion_factor=ffn_expansion_factor,
                        bias=bias,
                        LayerNorm_type=LayerNorm_type,
                        num_blocks=1)
        self.layers_large.append(layer)

        modules_tail = [common.Upsampler(common.default_conv, scale, int(dim), act=False),
                        common.default_conv(int(dim), out_channels, 3)]
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, inp_img, k_v, guide_p=None, dct_p=None):
        feat = inp_img

        inp_enc_level = self.patch_embed(feat)

        for layer in self.layers1:
            out_enc_level, _ = layer([inp_enc_level, k_v], guide_p=guide_p)

        for layer in self.layers2:
            out_enc_level, _ = layer([out_enc_level, k_v], dct_p=dct_p)

        out_enc_level = self.expand(out_enc_level)
        for layer in self.layers_large:
            out_enc_level, _ = layer([out_enc_level, k_v])


        out = self.tail(out_enc_level) + F.interpolate(inp_img, scale_factor=self.scale, mode='nearest')

        return out



class Encoder(nn.Module):
    def __init__(self,n_feats = 64, n_encoder_res = 6,scale=4):
        super(Encoder, self).__init__()
        self.scale=scale
        if scale == 2:
            E1=[nn.Conv2d(12, n_feats, kernel_size=3, padding=1),
                nn.LeakyReLU(0.1, True)]
        elif scale == 1:
            E1=[nn.Conv2d(48, n_feats, kernel_size=3, padding=1),
                nn.LeakyReLU(0.1, True)]
        else:
            E1=[nn.Conv2d(3, n_feats, kernel_size=3, padding=1),
                nn.LeakyReLU(0.1, True)]

        E2=[
            common.ResBlock(
                common.default_conv, n_feats, kernel_size=3
            ) for _ in range(n_encoder_res)
        ]
        E3=[
            nn.Conv2d(n_feats, n_feats * 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(n_feats * 2, n_feats * 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(n_feats * 2, n_feats * 4, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, True),
            nn.AdaptiveAvgPool2d(1),
        ]
        E=E1+E2+E3
        self.E = nn.Sequential(
            *E
        )
        self.mlp = nn.Sequential(
            nn.Linear(n_feats * 4, n_feats * 4),
            nn.LeakyReLU(0.1, True),
            nn.Linear(n_feats * 4, n_feats * 4),
            nn.LeakyReLU(0.1, True)
        )
        self.pixel_unshuffle = nn.PixelUnshuffle(4)
        self.pixel_unshufflev2 = nn.PixelUnshuffle(2)

    def forward(self, x):
        if self.scale == 2:
            feat = self.pixel_unshufflev2(x)
        elif self.scale == 1:
            feat = self.pixel_unshuffle(x)
        else:
            feat = x
        fea = self.E(feat).squeeze(-1).squeeze(-1)
        fea1 = self.mlp(fea)

        return fea1

class ResMLP(nn.Module):
    def __init__(self,n_feats = 512):
        super(ResMLP, self).__init__()
        self.resmlp = nn.Sequential(
            nn.Linear(n_feats , n_feats ),
            nn.LeakyReLU(0.1, True),
        )
    def forward(self, x):
        res=self.resmlp(x)
        return res

class denoise(nn.Module):
    def __init__(self,n_feats = 64, n_denoise_res = 5,timesteps=5):
        super(denoise, self).__init__()
        self.max_period=timesteps*10
        n_featsx4=4*n_feats
        resmlp = [
            nn.Linear(n_featsx4*2+1, n_featsx4),
            nn.LeakyReLU(0.1, True),
        ]
        for _ in range(n_denoise_res):
            resmlp.append(ResMLP(n_featsx4))
        self.resmlp=nn.Sequential(*resmlp)

    def forward(self,x, t,c):
        t=t.float()
        t =t/self.max_period
        t=t.view(-1,1)
        c = torch.cat([c,t,x],dim=1)

        fea = self.resmlp(c)

        return fea

@ARCH_REGISTRY.register()
class DHGMS2(nn.Module):
    def __init__(self,
        n_encoder_res=9,
        inp_channels=3,
        out_channels=3,
        scale=2,
        dim = 64,
        num_blocks = [6,6,6,6,6,6],
        heads = [1,1,1,1,1,1],
        ffn_expansion_factor = 2,
        bias = False,
        LayerNorm_type = 'WithBias',   ## Other option 'BiasFree'
        n_denoise_res = 1,
        linear_start= 0.1,
        linear_end= 0.99,
        timesteps = 4 ):
        super(DHGMS2, self).__init__()

        # Generator
        self.G = IR(
        inp_channels=inp_channels,
        out_channels=out_channels,
        scale = scale,
        dim = dim,
        num_blocks = num_blocks,
        heads = heads,
        ffn_expansion_factor = ffn_expansion_factor,
        bias = bias,
        LayerNorm_type = LayerNorm_type,   ## Other option 'BiasFree'
        )
        self.condition = Encoder(n_feats=64, n_encoder_res=n_encoder_res,scale = scale)

        self.denoise= denoise(n_feats=64, n_denoise_res=n_denoise_res,timesteps=timesteps)

        self.diffusion = DDPM(denoise=self.denoise, condition=self.condition ,n_feats=64,linear_start= linear_start,
  linear_end= linear_end, timesteps = timesteps)

    def forward(self, img, IPRS1=None):
        if self.training:
            IPRS2, pred_IPR_list=self.diffusion(img,IPRS1)

            dct_feature = dct(IPRS2, norm='ortho')
            dct_filter = high_pass(dct_feature, 64)
            dct_prior = idct(dct_filter, norm='ortho')

            # dct_prior = IPRS2
            guide_prior = IPRS2

            sr = self.G(img, IPRS1, guide_prior, dct_prior)

            return sr, pred_IPR_list
        else:
            IPRS2=self.diffusion(img)

            dct_feature = dct(IPRS2, norm='ortho')
            dct_filter = high_pass(dct_feature, 64)
            dct_prior = idct(dct_filter, norm='ortho')

            # dct_prior = IPRS2
            guide_prior = IPRS2

            sr = self.G(img, IPRS2, guide_prior, dct_prior)

            return sr
